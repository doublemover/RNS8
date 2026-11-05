"""Offline, authored session files; no synthetic executable is ever launched."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

from compiled_binary_identity import bind_document, canonical_digest
from benchmark_request import file_digest
from session_environment import RUNTIME_MAPPING_KEYS
from test_device_observation import capture as observation_capture, inspection
from test_target_qualification_support import QualificationFixture, write
from test_review_reference_support import add_synthetic_full_reference
from test_benchmark_sweep_support import finite_capture


class ObservationFixture:
    def __init__(
        self,
        root: Path,
        *,
        changed_checksum=False,
        changed_seed=False,
        duplicate_execution=False,
        swapped_device=False,
        cases=2,
        finite=False,
    ):
        base = QualificationFixture(root)
        self.root = base.root
        self.devices = [4, 5, 6, 7]
        self.commands, self.rows, self.captures = [], [], []
        self.binaries = base.binaries
        inspector = base.build / "rns8-inspect"
        inspector.write_bytes(b"AUTHORED INSPECTOR, NOT EXECUTABLE")
        self.binaries["rns8-inspect"] = inspector
        self.cases = [
            ("finite-modulus-" if finite else "bounded-seed-") + str(index)
            for index in range(cases)
        ]
        template = base.capture
        if finite:
            template = finite_capture("hip-direct", 300)
            template.pop("_path", None)
            for key in (
                "build_manifest",
                "compiler",
                "hip_toolchain",
                "git_commit",
                "artifact_capture",
                "request_identity",
                "configured_amdgpu_targets",
            ):
                template[key] = copy.deepcopy(base.capture[key])

        def stage(
            name, executable, arguments, document, environment, artifacts=(), **extra
        ):
            command = dict(name=name, argv=[str(executable), *arguments], **extra)
            raw = write(self.root / (name + ".stdout.log"), document)
            row = dict(
                name=name,
                returncode=0,
                timed_out=False,
                validation_errors=[],
                stdout=raw.read_text(encoding="utf-8"),
                process_id=document["process_id"],
                execution_environment=environment,
                artifacts={
                    str(path.resolve()): file_digest(path)
                    for path in [executable, raw, *artifacts]
                },
            )
            self.commands.append(command)
            self.rows.append(row)
            return row, raw

        for index, (name, binary) in enumerate(self.binaries.items()):
            document = dict(
                schema="rns8_compiled_binary_identity_v1",
                process_id=10 + index,
                build_manifest=base.capture["build_manifest"],
                library_build_manifest=base.capture["build_manifest"],
                library_abi_declaration_sha256="e" * 64,
            )
            row, raw = stage(
                "query-" + name,
                binary,
                ["--build-identity"],
                document,
                dict.fromkeys(RUNTIME_MAPPING_KEYS),
                compiled_identity=True,
            )
            row["compiled_binary_identity"] = bind_document(
                binary, raw, before_sha256=file_digest(binary)
            )

        first_execution = None
        for rank, selected in enumerate(self.devices):
            parent_name, child_name = (
                "parent-" + str(selected),
                "child-" + str(selected),
            )
            parent_environment = dict.fromkeys(RUNTIME_MAPPING_KEYS)
            environment = dict(
                parent_environment,
                HIP_VISIBLE_DEVICES=str(selected),
                RNS8_RANK=str(rank),
                RNS8_WORLD_SIZE=str(len(self.devices)),
                RNS8_MULTI_GPU_MODE="embarrassingly_parallel_shards",
            )
            parent = inspection(selected, device=selected)
            child = inspection(0, device=selected)
            for name, ordinal, info, env, extra in (
                (
                    parent_name,
                    selected,
                    parent,
                    parent_environment,
                    dict(parent_inspect=True, device=None),
                ),
                (
                    child_name,
                    0,
                    child,
                    environment,
                    dict(device=selected, match_parent=parent_name),
                ),
            ):
                info["process_id"] = 100 + len(self.rows)
                stage(
                    name,
                    inspector,
                    ["--backend", "hip-direct", "--device", str(ordinal), "--json"],
                    info,
                    env,
                    inspect_device=ordinal,
                    selected_device=selected,
                    inspect_target="gfx942",
                    inspect_backend="hip-direct",
                    **extra,
                )
            for index, case in enumerate(self.cases):
                capture = copy.deepcopy(template)
                capture.update(observation_capture(device=selected))
                capture["seed"] = (
                    20 + index + (1 if changed_seed and rank == 3 and index == 0 else 0)
                )
                if finite:
                    modulus = 255 - index
                    capture["finite_modulus"] = modulus
                    metadata = capture["backend_metadata"]
                    metadata["accumulator_safety"]["modulus"] = modulus
                    metadata["autotune_key"] = metadata["autotune_key"].replace(
                        "finite_modulus=255;", f"finite_modulus={modulus};"
                    )
                capture["checksum_u64"] = (
                    1000
                    + index
                    + (1 if changed_checksum and rank == 3 and index == 0 else 0)
                )
                capture["execution_id"] = str(uuid.uuid4())
                if first_execution is None:
                    first_execution = capture["execution_id"]
                elif duplicate_execution and rank == 3 and index == 0:
                    capture["execution_id"] = first_execution
                capture["process_id"] = 100 + len(self.rows)
                capture["runtime_environment"] = dict(environment)
                request = capture["request_identity"]
                request["arguments"] = [
                    "synthetic-offline-only",
                    "--seed",
                    str(capture["seed"]),
                ]
                request["environment"].update(environment)
                if swapped_device and rank == 3 and index == 0:
                    request["environment"]["HIP_VISIBLE_DEVICES"] = "4"
                request["identity_sha256"] = canonical_digest(
                    {
                        key: value
                        for key, value in request.items()
                        if key != "identity_sha256"
                    }
                )
                add_synthetic_full_reference(capture)
                path = write(
                    self.root / ("work-" + str(selected) + "-" + str(index) + ".json"),
                    capture,
                )
                name = "work-" + str(selected) + "-" + str(index)
                stage(
                    name,
                    self.binaries["rns8-bench"],
                    request["arguments"],
                    capture,
                    environment,
                    [path],
                    capture=str(path),
                    device=selected,
                    selected_device=selected,
                    case_id=case,
                    expected_backend="hip-direct",
                    expected_target="gfx942",
                    observation_parent_stage=parent_name,
                    observation_child_stage=child_name,
                )
                self.captures.append((name, path))
        self.manifest = dict(
            schema_version=2,
            command_working_directory=str(self.root),
            requested_architecture="gfx942",
            devices=self.devices,
            commands=self.commands,
        )
        self.manifest["identity_sha256"] = canonical_digest(self.manifest)
        self.manifest_path = write(self.root / "manifest.json", self.manifest)
        self.results = dict(
            schema="rns8_gpu_session_results_v2",
            run_id=str(uuid.uuid4()),
            manifest_identity_sha256=self.manifest["identity_sha256"],
            results=self.rows,
        )
        self.results_path = write(self.root / "results.json", self.results)

    def receipts(self):
        from multigpu_observation import make_binding

        return [
            write(
                self.root / (stage + ".binding.json"),
                make_binding(path, self.manifest_path, self.results_path, stage),
            )
            for stage, path in self.captures
        ]
