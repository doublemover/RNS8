"""Source contracts for the maintained byte-LDS tiled policy family, not ISA proof."""
from __future__ import annotations

import json
from pathlib import Path
import re

from benchmark_request import file_digest
from compiled_binary_identity import canonical_digest

ROOT = Path(__file__).resolve().parents[1]
CORE = "src/backend_hip_direct/hip_direct_tiled_residue_core.cuh"
LOAD = "src/backend_common/residue_tile.hpp"
WRAPPERS = "src/backend_hip_direct/hip_direct_gemm_tiled_kernels.cuh"
SCHEDULED = "src/backend_hip_direct/hip_direct_gemm_grouped_scheduled_kernels.cuh"


def validate_policy_manifest(value: object) -> list[dict]:
    fields={"id","core","load","wrappers","tile","safe_signed_byte_slab","barriers","policies"}
    if type(value) is not dict or set(value)!=fields:
        raise ValueError("tiled residue dataflow fields differ from v2")
    if (value["id"]!="byte_lds_tiled_residue_v2" or value["core"]!=CORE or value["load"]!=LOAD or
            value["wrappers"]!=[WRAPPERS,SCHEDULED] or type(value["tile"]) is not list or
            any(type(item) is not int for item in value["tile"]) or value["tile"]!=[16,16,64,17] or
            type(value["safe_signed_byte_slab"]) is not int or value["safe_signed_byte_slab"]!=65536 or
            value["barriers"]!="all_admitted_CTA_threads_each_load_compute_tile"):
        raise ValueError("tiled residue dataflow identity/envelope is unsupported")
    rows=value["policies"]
    if type(rows) is not list or len(rows)!=17:
        raise ValueError("tiled residue family requires its 17 explicit wrapper policies")
    names=set()
    for row in rows:
        if (type(row) is not dict or set(row)!={"symbol","a","b","reducer","dot","region","addressing","source","schedule","outputs"} or
                type(row["symbol"]) is not str or not re.fullmatch(r"rns8_[A-Za-z0-9_]+",row["symbol"]) or
                row["symbol"] in names or type(row["region"]) is not bool or
                type(row["outputs"]) is not int or row["outputs"] not in (1,2)):
            raise ValueError("tiled residue wrapper policy is malformed/duplicated")
        for key,choices in (("a",("centered_i8","native_u8")),("b",("centered_i8","native_u8")),
                            ("reducer",("dynamic","fixed","default_ladder")),
                            ("dot",("configured","packed","selected_default_moduli","scalar_pair")),
                            ("addressing",("direct","task_pointers","prefix_planes","task_prefix_planes","uniform_inputs_prefix_outputs")),
                            ("source",(WRAPPERS,SCHEDULED)),("schedule",("none","uniform","entry","active_entry"))):
            if type(row[key]) is not str or row[key] not in choices:
                raise ValueError("unsupported tiled residue "+key)
        if row["b"]=="native_u8" and row["a"]!="native_u8":
            raise ValueError("no native-B-only tiled policy is implemented")
        if row["reducer"]=="default_ladder" and (row["a"]!="centered_i8" or row["b"]!="centered_i8"):
            raise ValueError("default ladder policy requires resident centered inputs")
        if row["dot"]=="packed" and (row["reducer"]!="fixed" or row["a"]!="centered_i8" or row["b"]!="centered_i8"):
            raise ValueError("explicit packed-dot policy requires fixed resident inputs")
        if row["dot"]=="selected_default_moduli" and (row["reducer"]!="default_ladder" or row["region"] or row["addressing"]!="prefix_planes"):
            raise ValueError("selected-modulus dot policy belongs only to ordinary grouped prefixes")
        if row["source"]==WRAPPERS:
            if row["schedule"]!="none" or row["outputs"]!=1 or row["dot"]=="scalar_pair" or row["addressing"]=="uniform_inputs_prefix_outputs":
                raise ValueError("ordinary wrapper cannot inherit scheduled/pair policy")
        elif (row["schedule"]=="none" or row["region"] or row["reducer"]!="default_ladder" or
              row["a"]!="centered_i8" or row["b"]!="centered_i8" or
              row["dot"]!=("scalar_pair" if row["outputs"]==2 else "configured") or
              (row["outputs"]==2 and row["schedule"]!="uniform") or
              row["addressing"]!={"uniform":"uniform_inputs_prefix_outputs","entry":"direct","active_entry":"prefix_planes"}[row["schedule"]]):
            raise ValueError("scheduled wrapper has an incompatible tile policy")
        names.add(row["symbol"])
    return [dict(row) for row in rows]


def wrapper_bodies(text: str) -> dict[str,str]:
    bodies={}
    for match in re.finditer(r"__global__\s+void\s+__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s*(rns8_\w+)\(",text):
        start=text.index("{",match.end());position=start+1;depth=1
        while depth and position<len(text):
            depth+=(text[position]=="{")-(text[position]=="}")
            position+=1
        if depth:
            raise ValueError("unterminated tiled entrypoint body")
        body=text[match.start():position]
        if re.search(r"__shared__\s+int8_t\s+a_tile",body):
            name=match.group(1)
            if name in bodies:raise ValueError("duplicate tiled wrapper body")
            bodies[name]=body
    return bodies


def inspect_source(value: object, *, root: Path=ROOT) -> dict:
    rows=validate_policy_manifest(value)
    stored=json.loads((root/"metadata/kernels.yaml").read_text(encoding="utf-8"))
    if canonical_digest(stored["direct_residue_tile_dataflow"])!=canonical_digest(value):
        raise ValueError("selected tiled policy is not the retained registry content")
    sources={name:wrapper_bodies((root/name).read_text(encoding="utf-8")) for name in (WRAPPERS,SCHEDULED)}
    for name,bodies in sources.items():
        if set(bodies)!={row["symbol"] for row in rows if row["source"]==name}:
            raise ValueError("tiled source and policy denominator differ: "+name)
    for row in rows:
        body=sources[row["source"]][row["symbol"]]
        compact=re.sub(r"\s+","",body)
        fixed=row["reducer"]=="fixed"
        def operand(letter):
            if row[letter.lower()]=="centered_i8":
                value="tile_"+letter if row["schedule"] in ("entry","active_entry") else letter
                return f"rns8::detail::CenteredResidueTileInput{{{value},ld{letter.lower()}}}"
            return f"rns8::detail::NativeByteResidueTileInput<{('Modulus' if fixed else '0')}>{{{letter},ld{letter.lower()},uint32_t({('Modulus' if fixed else 'modulus')})}}"
        reducer=("Rns8FixedTileReducer<Modulus>{}" if fixed else
                 "Rns8DefaultLadderTileReducer{modulus_index,modulus,modulus_reciprocal}" if row["reducer"]=="default_ladder" else
                 "Rns8DynamicTileReducer{modulus,modulus_reciprocal}")
        tr="region_row_offset+region_tile_row" if row["region"] else "tile_row"
        tc="region_col_offset+region_tile_col" if row["region"] else "tile_col"
        if row["outputs"]==2:
            load=("rns8::detail::load_residue_tile<kRns8HipTileM,kRns8HipTileN,kRns8HipTileK,kRns8HipTileNPadded>("
                  f"a_tile,b_tile,{{m,n,tile_row,tile_col,k_offset}},tile_k,tile_extent,lane,block_threads,{operand('A')},{operand('B')});")
            pair=("rns8::detail::accumulate_residue_tile_pair(&a_tile[thread_row][0],"
                  "&b_tile[0][thread_col_pair*2],kRns8HipTileNPadded,tile_extent,acc0,acc1);")
            if (load not in compact or pair not in compact or compact.count("__syncthreads();")!=2 or
                    "rns8_tiled_residue_body<" in compact or
                    "__launch_bounds__(kRns8HipTileM*(kRns8HipTileN/2))" not in compact):
                raise ValueError(row["symbol"]+": shared pair loader/dot/barrier tuple differs")
        else:
            window=(f"{{m,n,{tr},{tc},k_offset,k_block,ldc,row,col,output_active,accumulate}}" if row["schedule"] not in ("entry","active_entry") else
                    "{static_cast<int>(entry.row_extent),static_cast<int>(entry.col_extent),tile_local_row,tile_local_col,"
                    "k_offset,k_block,ldc,row,col,output_active&&!output_zero_by_row_col,accumulate}")
            call=(f"rns8_tiled_residue_body<rns8::detail::ResidueTileDot::{row['dot']}>(a_tile,b_tile,"
                  f"{window},{operand('A')},{operand('B')},C,{reducer});")
            if compact.count("rns8_tiled_residue_body<")!=1 or call not in compact or "__syncthreads" in body:
                raise ValueError(row["symbol"]+": shared core/policy tuple differs from source")
        if row["schedule"] in ("entry","active_entry"):
            for expression in ("tile_A=A+static_cast<int>(entry.row_offset)*lda;",
                               "tile_B=B+static_cast<int>(entry.col_offset);",
                               "if(output_zero_by_row_col)C[row*ldc+col]=0;",
                               "residue_tile_zero_product<kRns8HipTileM,kRns8HipTileN>(zero_a_rows,zero_b_cols,"
                               "{static_cast<int>(entry.row_extent),static_cast<int>(entry.col_extent),tile_local_row,tile_local_col,"
                               "static_cast<int>(entry.row_offset),static_cast<int>(entry.col_offset)})"):
                if expression not in compact:
                    raise ValueError(row["symbol"]+": scheduled rebase/zero store differs")
        for array in ("a_tile[kRns8HipTileM][kRns8HipTileK]","b_tile[kRns8HipTileK][kRns8HipTileNPadded]"):
            if "__shared__int8_t"+array+";" not in compact:
                raise ValueError(row["symbol"]+": concrete shared-array envelope differs")
        addressing=("task_prefix_planes" if "A_ptrs" in body and "a_plane_offset" in body else
                    "task_pointers" if "A_ptrs" in body else "prefix_planes" if "a_plane_offset" in body else
                    "uniform_inputs_prefix_outputs" if "c_plane_offset" in body else "direct")
        if addressing!=row["addressing"]:
            raise ValueError(row["symbol"]+": task/plane addressing policy differs")
    core=(root/CORE).read_text(encoding="utf-8")
    compact_core=re.sub(r"\s+","",core)
    if compact_core.count("__syncthreads();")!=2 or "load_residue_tile<" not in core or "__forceinline__voidrns8_tiled_residue_body" not in compact_core:
        raise ValueError("shared tiled core lost its explicit barrier/load/inlining structure")
    inputs=(CORE,LOAD,WRAPPERS,SCHEDULED,"src/backend_common/residue_dot.hpp","src/backend_common/finite_u8_reducer.hpp",
            "src/backend_hip_direct/hip_direct_device_common.cuh","metadata/kernels.yaml")
    return dict(schema="rns8_tiled_residue_source_contract_v2",selected_wrappers=len(rows),
                policies=rows,source_sha256={name:file_digest(root/name) for name in inputs},
                scope="lexical_source_contract_not_compiled_lowering_or_execution_proof",
                hardware_qualification=False,performance_qualified=False)


def main():
    data=json.loads((ROOT/"metadata/kernels.yaml").read_text(encoding="utf-8"))
    print(json.dumps(inspect_source(data["direct_residue_tile_dataflow"]),indent=2))


if __name__=="__main__":
    main()
