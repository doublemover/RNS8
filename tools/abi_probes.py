"""Generate compiler-checked C17/C++17 ABI probes without running consumers."""

from __future__ import annotations

from pathlib import Path

from abi_surface import Surface, declaration_sha256

LAYOUT_SCHEMA = "rns8_public_abi_layout_v1"


def metric_expressions(surface: Surface) -> dict[str, str]:
    values = {
        "sizeof:void_pointer": "sizeof(void*)",
        "alignof:void_pointer": "RNS8_ABI_ALIGNOF(void*)",
        "sizeof:long": "sizeof(long)",
        "sizeof:uint32_t": "sizeof(uint32_t)",
        "sizeof:uint64_t": "sizeof(uint64_t)",
        "CHAR_BIT": "CHAR_BIT",
    }
    for name in sorted(
        set(surface.records) | set(surface.enums) | set(surface.callbacks) | set(surface.aliases)
    ):
        values["sizeof:" + name] = "sizeof(" + name + ")"
        values["alignof:" + name] = "RNS8_ABI_ALIGNOF(" + name + ")"
    for name in surface.handles:
        values["sizeof:" + name + "_pointer"] = "sizeof(" + name + "*)"
        values["alignof:" + name + "_pointer"] = "RNS8_ABI_ALIGNOF(" + name + "*)"
    for name, fields in surface.records.items():
        for field, _ in fields:
            values["offsetof:" + name + "." + field] = "offsetof(" + name + "," + field + ")"
            values["field_size:" + name + "." + field] = "sizeof(((" + name + "*)0)->" + field + ")"
    for name in surface.constants:
        values["constant:" + name] = name
    for constants in surface.enums.values():
        for name in constants:
            values["constant:" + name] = name
    for name in surface.macros:
        values["macro:" + name] = name
    return dict(sorted(values.items()))


def includes(surface: Surface, include_root: Path) -> str:
    root = include_root.resolve(strict=True)
    if any(character in str(root) for character in ('"', "\n", "\r")):
        raise ValueError("ABI include path cannot be represented safely in a C header directive")
    result = []
    for header in surface.headers:
        path = (root / header).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("ABI probe header escaped installed include root")
        result.append('#include "' + path.as_posix() + '"')
    return "\n".join(result) + "\n"


POLICY_PREAMBLE = """
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#ifdef __cplusplus
#include <type_traits>
#define RNS8_ABI_ALIGNOF(type) alignof(type)
#define RNS8_ABI_ASSERT(value) static_assert(value, #value)
#else
#define RNS8_ABI_ALIGNOF(type) _Alignof(type)
#define RNS8_ABI_ASSERT(value) _Static_assert(value, #value)
#endif
"""


def typedef_declaration(signature: str, name: str) -> str:
    signature = signature.replace("()", "(void)")
    if "(*)" in signature:
        return "typedef " + signature.replace("(*)", "(*" + name + ")", 1) + ";"
    if "[" in signature:
        base, suffix = signature.split("[", 1)
        return "typedef " + base + " " + name + "[" + suffix + ";"
    return "typedef " + signature + " " + name + ";"


def declaration_assertions(surface: Surface, *, references: bool = True) -> str:
    result = [
        '\n#ifdef __cplusplus\n#define RNS8_ABI_MEMBER(record,field,type) static_assert(std::is_same<decltype(((record*)nullptr)->field),type>::value, "public member type")\n#define RNS8_ABI_TYPE(type,expected) static_assert(std::is_same<type,expected>::value, "public typedef type")\n#define RNS8_ABI_FUNCTION(name,type) static_assert(std::is_same<decltype(&name),type>::value, "public function signature")\n#else\n#define RNS8_ABI_MEMBER(record,field,type) _Static_assert(_Generic(&(((record*)0)->field),type*:1,default:0), "public member type")\n#define RNS8_ABI_TYPE(type,expected) _Static_assert(_Generic((type*)0,expected*:1,default:0), "public typedef type")\n#define RNS8_ABI_FUNCTION(name,type) _Static_assert(_Generic(&name,type:1,default:0), "public function signature")\n#endif\n'
    ]
    for name in surface.enums:
        result.append(
            "RNS8_ABI_ASSERT(sizeof(" + name + ")==4 && RNS8_ABI_ALIGNOF(" + name + ")==4);"
        )
    result.append("#ifdef __cplusplus")
    for name in surface.records:
        result.append(
            "static_assert(std::is_standard_layout<" + name + '>::value, "public C record layout");'
        )
    result.append("#endif")
    field_index = 0
    for record, fields in surface.records.items():
        for name, signature in fields:
            alias = "rns8_abi_member_" + str(field_index)
            field_index += 1
            result.append(typedef_declaration(signature, alias))
            result.append("RNS8_ABI_MEMBER(" + record + "," + name + "," + alias + ");")
    result.append('#ifdef __cplusplus\nextern "C" {\n#endif')
    names = [*surface.functions, *surface.header_functions]
    for index, name in enumerate(names):
        signature = (surface.functions.get(name) or surface.header_functions[name]).replace(
            "()", "(void)"
        )
        opening = signature.find("(")
        if opening < 1:
            raise ValueError("unsupported frontend function type: " + name)
        alias = "rns8_abi_function_" + str(index)
        result.append(
            "typedef " + signature[:opening] + "(*" + alias + ")" + signature[opening:] + ";"
        )
        if references and name in surface.functions:
            result.append(
                "static " + alias + " volatile rns8_abi_reference_" + str(index) + "=&" + name + ";"
            )
    result.append("#ifdef __cplusplus\n}\n#endif")
    for category, entries in (
        ("callback", surface.callbacks),
        ("alias", surface.aliases),
    ):
        for index, (name, signature) in enumerate(entries.items()):
            expected = "rns8_abi_" + category + "_" + str(index)
            result.append(typedef_declaration(signature, expected))
            result.append("RNS8_ABI_TYPE(" + name + "," + expected + ");")
    for index, name in enumerate(names):
        result.append("RNS8_ABI_FUNCTION(" + name + ",rns8_abi_function_" + str(index) + ");")
    result.extend(("#undef RNS8_ABI_MEMBER", "#undef RNS8_ABI_TYPE", "#undef RNS8_ABI_FUNCTION"))
    return "\n".join(result) + "\n"


def metric_frontend_source(surface: Surface, include_root: Path) -> tuple[str, list[str]]:
    metrics = metric_expressions(surface)
    # Layout extraction needs only constant expressions, not instantiated C++ traits.
    # Keep type/signature assertions in the actual library and native consumers.
    source = includes(surface, include_root) + POLICY_PREAMBLE.replace(
        "#include <type_traits>\n", ""
    )
    source += "#ifdef __cplusplus\nenum rns8_abi_frontend_metrics : uint64_t {\n#else\nenum rns8_abi_frontend_metrics {\n#endif\n"
    source += ",\n".join(
        "rns8_abi_metric_" + str(index) + "=(uint64_t)(" + expression + ")"
        for index, expression in enumerate(metrics.values())
    )
    source += "\n};\n"
    return source, list(metrics)


def metric_assertions(surface: Surface, expected: dict[str, int] | None) -> str:
    if expected is None:
        return ""
    metrics = metric_expressions(surface)
    if set(metrics) != set(expected):
        raise ValueError(
            "layout baseline does not cover every current public object/member/constant"
        )
    if any(
        type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF
        for value in expected.values()
    ):
        raise ValueError("layout baseline values must be unsigned64 integers")
    return (
        "\n".join(
            "RNS8_ABI_ASSERT((uint64_t)("
            + expression
            + ") == UINT64_C("
            + str(expected[name])
            + "));"
            for name, expression in metrics.items()
        )
        + "\n"
    )


def native_device_smoke(surface: Surface) -> str:
    required = {"rns8_query_native_device", "rns8_status_string"}
    if not required <= surface.functions.keys():
        raise ValueError("installed package lacks current CPU ABI smoke entry points")
    types = (
        "rns8_native_device_type",
        "rns8_native_device_access",
        "rns8_native_device_policy",
    )
    if not set(types) <= surface.enums.keys():
        raise ValueError("native descriptor enums are missing from the ABI inventory")
    adjacent = [max(surface.enums[name].values()) + 1 for name in types]
    if any(value > 0xFFFFFFFF for value in adjacent):
        raise ValueError("native enum adjacency needs an explicit domain fixture")
    source = """
static unsigned rns8_abi_owner_callbacks = 0;
static rns8_status rns8_abi_acquire(void* owner) { (void)owner; ++rns8_abi_owner_callbacks; return RNS8_SUCCESS; }
static void rns8_abi_release(void* owner) { (void)owner; ++rns8_abi_owner_callbacks; }
static int rns8_abi_native_smoke(void) {
  uint64_t dummy = 0;
  rns8_native_device_desc valid;
  rns8_native_device_resources output;
  unsigned char before[sizeof(output)];
  unsigned role, sample;
  const uint32_t invalids[3][3] = {
    {UINT32_MAX,UINT32_C(0x80000000),UINT32_C(@TYPE@)},
    {UINT32_MAX,UINT32_C(0x80000000),UINT32_C(@ACCESS@)},
    {UINT32_MAX,UINT32_C(0x80000000),UINT32_C(@POLICY@)}
  };
  memset(&valid,0,sizeof(valid));
  valid.struct_size=(uint32_t)sizeof(valid);valid.abi_version=RNS8_ABI_VERSION;
  valid.element_type=RNS8_NATIVE_DEVICE_I64;valid.access=RNS8_NATIVE_DEVICE_READ;
  valid.policy=RNS8_NATIVE_DEVICE_BORROW;valid.allocation_base=&dummy;valid.allocation_bytes=sizeof(dummy);
  valid.rows=valid.cols=valid.ld=1;valid.device_id=0;valid.producer_complete=1;
  valid.owner_identity=valid.generation=1;valid.owner=&dummy;
  valid.acquire=rns8_abi_acquire;valid.release=rns8_abi_release;
  memset(&output,0,sizeof(output));output.struct_size=(uint32_t)sizeof(output);output.abi_version=RNS8_ABI_VERSION;
  if(rns8_query_native_device(&valid,&output)!=RNS8_SUCCESS || output.view_span_bytes!=8) return 20;
  for(role=0;role<3;++role) for(sample=0;sample<3;++sample) {
    rns8_native_device_desc bad=valid;
    void* destination=role==0?(void*)&bad.element_type:role==1?(void*)&bad.access:(void*)&bad.policy;
    memcpy(destination,&invalids[role][sample],sizeof(uint32_t));
    memset(&output,0xA5,sizeof(output));output.struct_size=(uint32_t)sizeof(output);output.abi_version=RNS8_ABI_VERSION;
    memcpy(before,&output,sizeof(output));
    if(rns8_query_native_device(&bad,&output)!=RNS8_INVALID_ARGUMENT || memcmp(before,&output,sizeof(output))) return 21;
  }
  {
    struct rns8_abi_larger_descriptor {rns8_native_device_desc value;uint64_t canary;} larger;
    rns8_native_device_desc bad=valid;
    bad.struct_size=(uint32_t)sizeof(bad)-1;
    if(rns8_query_native_device(&bad,&output)!=RNS8_INVALID_ARGUMENT || memcmp(before,&output,sizeof(output))) return 22;
    bad=valid;bad.abi_version=RNS8_ABI_VERSION+1;
    if(rns8_query_native_device(&bad,&output)!=RNS8_INVALID_ARGUMENT || memcmp(before,&output,sizeof(output))) return 23;
    larger.value=valid;larger.value.struct_size=(uint32_t)sizeof(larger);larger.canary=UINT64_C(0xA5A5A5A5A5A5A5A5);
    if(rns8_query_native_device(&larger.value,&output)!=RNS8_INVALID_ARGUMENT ||
       larger.canary!=UINT64_C(0xA5A5A5A5A5A5A5A5) || memcmp(before,&output,sizeof(output))) return 24;
  }
  if(rns8_abi_owner_callbacks) return 25;
  if(!rns8_status_string(RNS8_SUCCESS)) return 26;
  {
    rns8_context_options options;
    rns8_matrix_desc matrix_desc;
    rns8_context* context=NULL;
    rns8_matrix* matrix=NULL;
    rns8_status status;
    memset(&options,0,sizeof(options));
    options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;
    options.requested_backend=RNS8_BACKEND_CPU_REFERENCE;
    if(rns8_create_context(-1,&options,&context)!=RNS8_SUCCESS || !context) return 28;
    memset(&matrix_desc,0,sizeof(matrix_desc));
    matrix_desc.struct_size=sizeof(matrix_desc);matrix_desc.abi_version=RNS8_ABI_VERSION;
    matrix_desc.rows=matrix_desc.cols=matrix_desc.logical_ld=1;
    matrix_desc.semantics=RNS8_BOUNDED_I64;matrix_desc.logical_layout=RNS8_LAYOUT_ROW_MAJOR;
    matrix_desc.bound_kind=RNS8_BOUND_GLOBAL_MAX_ABS;matrix_desc.tile_m=matrix_desc.tile_n=1;matrix_desc.max_prefix=9;
    status=rns8_create_matrix(context,&matrix_desc,&matrix);
    if(status!=RNS8_SUCCESS || !matrix) {rns8_destroy_context(context);return 29;}
    if(rns8_destroy_matrix(matrix)!=RNS8_SUCCESS) {rns8_destroy_context(context);return 30;}
    if(rns8_destroy_context(context)!=RNS8_SUCCESS) return 31;
  }
  return 0;
}
"""
    return (
        source.replace("@TYPE@", str(adjacent[0]))
        .replace("@ACCESS@", str(adjacent[1]))
        .replace("@POLICY@", str(adjacent[2]))
    )


def native_probe_source(
    surface: Surface, include_root: Path, expected: dict[str, int] | None
) -> str:
    source = (
        includes(surface, include_root)
        + POLICY_PREAMBLE
        + "#include <stdio.h>\n#include <string.h>\n"
    )
    source += (
        declaration_assertions(surface)
        + metric_assertions(surface, expected)
        + native_device_smoke(surface)
    )
    metrics = metric_expressions(surface)
    source += "static const rns8_public_abi_metric rns8_expected_metrics[] = {\n"
    source += ",\n".join(
        '  {"' + name + '",(uint64_t)(' + expression + ")}" for name, expression in metrics.items()
    )
    source += "\n};\n"

    source += r"""
static void rns8_abi_json_text(const char* text) {
  const unsigned char* p=(const unsigned char*)text;
  putchar('"');
  for(;*p;++p) {
    if(*p=='"' || *p=='\\') {putchar('\\');putchar(*p);}
    else if(*p<32) printf("\\u%04x",(unsigned)*p);
    else putchar(*p);
  }
  putchar('"');
}
"""
    source += "int main(void) {\n  int smoke=rns8_abi_native_smoke();\n  rns8_public_abi_info library;\n  size_t metric;\n  if(smoke) return smoke;\n"
    source += "  memset(&library,0,sizeof(library));library.struct_size=sizeof(library);library.abi_version=RNS8_ABI_VERSION;\n"
    source += "  if(rns8_get_public_abi_info(&library)!=RNS8_SUCCESS || library.metric_schema_version!=RNS8_PUBLIC_ABI_METRICS_VERSION || !library.metrics || library.metric_count!=sizeof(rns8_expected_metrics)/sizeof(rns8_expected_metrics[0])) return 32;\n"
    source += (
        '  if(!library.compiler_id || !library.runtime_model || !library.declaration_sha256 || strcmp(library.declaration_sha256,"'
        + declaration_sha256(surface)
        + '")) return 33;\n'
    )
    source += "  for(metric=0;metric<library.metric_count;++metric) if(!library.metrics[metric].name || strcmp(library.metrics[metric].name,rns8_expected_metrics[metric].name) || library.metrics[metric].value!=rns8_expected_metrics[metric].value) return 34;\n"
    for index, name in enumerate([*surface.functions, *surface.header_functions]):
        if name in surface.functions:
            source += "  if(!rns8_abi_reference_" + str(index) + ") return 27;\n"
    source += '  printf("{\\"schema\\":\\"' + LAYOUT_SCHEMA + '\\",\\"metrics\\":{");\n'
    for index, (name, expression) in enumerate(metric_expressions(surface).items()):
        prefix = "," if index else ""
        source += (
            '  printf("'
            + prefix
            + '\\"'
            + name
            + '\\":%llu",(unsigned long long)('
            + expression
            + "));\n"
        )
    source += '  printf("},\\"library_compiler_id\\":");rns8_abi_json_text(library.compiler_id);\n'
    source += (
        '  printf(",\\"library_runtime_model\\":");rns8_abi_json_text(library.runtime_model);\n'
    )
    source += (
        '  printf(",\\"declaration_sha256\\":");rns8_abi_json_text(library.declaration_sha256);\n'
    )
    source += '  printf(",\\"native_device_raw_enum_cases\\":9,\\"all_external_enum_paths_covered\\":false}\\n");\n  return 0;\n}\n'
    return source


def installed_consumer_cmake(prefix: Path, linkage: str, config_dir: Path) -> str:
    if linkage not in ("static", "shared"):
        raise ValueError("explicit static/shared ABI consumer required")
    target = "rns8::rns8_static" if linkage == "static" else "rns8::rns8"
    prefix = prefix.resolve(strict=True)
    config_dir = config_dir.resolve(strict=True)
    if any(character in str(prefix) + str(config_dir) for character in ('"', ";", "$", "\n", "\r")):
        raise ValueError("ABI package path cannot be represented safely in a CMake literal")
    # find_package has one explicit package location; RNS8 headers are included
    # by absolute installed paths in generated source, never a source-tree fallback.
    return (
        """
cmake_minimum_required(VERSION 3.22)
project(RNS8InstalledABI LANGUAGES C CXX)
find_package(RNS8 CONFIG REQUIRED PATHS "@CONFIG@" NO_DEFAULT_PATH)
if(NOT TARGET @TARGET@)
  message(FATAL_ERROR "Requested installed ABI linkage target is absent")
endif()
foreach(_lane IN ITEMS c cpp)
  add_executable(rns8_abi_@LINKAGE@_DOLLAR{_lane} "probe.DOLLAR{_lane}")
  target_link_libraries(rns8_abi_@LINKAGE@_DOLLAR{_lane} PRIVATE @TARGET@)
  target_include_directories(rns8_abi_@LINKAGE@_DOLLAR{_lane} BEFORE PRIVATE "@PREFIX@/include")
  set_target_properties(rns8_abi_@LINKAGE@_DOLLAR{_lane} PROPERTIES
    C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
endforeach()
file(GENERATE OUTPUT "installed-binary-DOLLAR<CONFIG>.txt" CONTENT "DOLLAR<TARGET_FILE:@TARGET@>")
""".replace("@CONFIG@", config_dir.as_posix())
        .replace("@PREFIX@", prefix.as_posix())
        .replace("@TARGET@", target)
        .replace("@LINKAGE@", linkage)
        .replace("DOLLAR", "$")
    )


def library_inventory_source(surface: Surface) -> str:
    source = "// Generated by tools/check_public_abi.py author; do not edit manually.\n"
    source += "".join("#include <" + header + ">\n" for header in surface.headers)
    source += POLICY_PREAMBLE + declaration_assertions(surface, references=False)
    source += "\nnamespace {\n"
    source += (
        'constexpr char rns8_public_declaration_sha256[] = "' + declaration_sha256(surface) + '";\n'
    )
    source += "constexpr rns8_public_abi_metric rns8_compiled_abi_metrics[] = {\n"
    source += ",\n".join(
        '  {"' + name + '",(uint64_t)(' + expression + ")}"
        for name, expression in metric_expressions(surface).items()
    )
    source += "\n};\n}\n"
    return source
