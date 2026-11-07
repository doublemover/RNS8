// File-only, single-threaded PDB lookup. No process is inspected or invaded.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#ifndef RNS8_PDB_SYMBOLIZE_SOURCE_SHA256
#error "CMake must bind the PDB reader to its source"
#endif

namespace {
using json = nlohmann::json;
constexpr DWORD64 base = 0x100000000;
constexpr size_t max_names = 4096;
[[noreturn]] void fail(const char* reason) {
  throw std::runtime_error(reason);
}

std::string utf8(const wchar_t* text, size_t limit) {
  size_t n = 0;
  while (n < limit && text[n])
    ++n;
  if (n == limit)
    fail("unterminated or oversized native debug text");
  if (!n)
    return {};
  const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, static_cast<int>(n),
                                         nullptr, 0, nullptr, nullptr);
  if (!length)
    fail("invalid native UTF-16 debug text");
  std::string result(static_cast<size_t>(length), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, static_cast<int>(n), result.data(),
                          length, nullptr, nullptr) != length)
    fail("native debug text conversion failed");
  return result;
}

std::filesystem::path local_file(const std::string& text) {
  if (text.empty() || text.find('\0') != std::string::npos)
    fail("local file required");
  const auto path = std::filesystem::canonical(std::filesystem::u8path(text));
  const auto name = path.wstring();
  if (name.size() < 3 || name[1] != L':' || name[2] != L'\\' ||
      !std::filesystem::is_regular_file(path))
    fail("local drive file required");
  return path;
}

struct Library {
  HMODULE value = nullptr;
  ~Library() {
    if (value)
      FreeLibrary(value);
  }
  template <class T> T function(const char* name) {
    const auto raw = GetProcAddress(value, name);
    if (!raw)
      fail("selected DbgHelp lacks a required Unicode API");
    T result;
    static_assert(sizeof(result) == sizeof(raw));
    std::memcpy(&result, &raw, sizeof(result));
    return result;
  }
};
struct Session {
  HANDLE key = this;
  decltype(&SymCleanup) cleanup;
  bool active = false;
  ~Session() {
    if (active)
      cleanup(key);
  }
};

json symbolize(const json& request) {
  if (!request.is_object() || request.size() != 4 || !request.contains("pdb") ||
      !request.contains("dbghelp") || !request.contains("image_size") ||
      !request.contains("rvas") || !request.at("pdb").is_string() ||
      !request.at("dbghelp").is_string() || !request.at("image_size").is_number_unsigned() ||
      !request.at("rvas").is_array())
    fail("bounded typed PDB request required");
  const auto size = request.at("image_size").get<uint64_t>();
  const auto& rvas = request.at("rvas");
  if (!size || size > UINT32_MAX || rvas.empty() || rvas.size() > 512)
    fail("bounded nonempty image and RVA inventory required");
  for (const auto& rva : rvas)
    if (!rva.is_number_unsigned() || rva.get<uint64_t>() >= size)
      fail("RVA outside the explicit image span");
  const auto pdb = local_file(request.at("pdb").get<std::string>());
  const auto dll = local_file(request.at("dbghelp").get<std::string>());
  if (CompareStringOrdinal(pdb.extension().c_str(), -1, L".pdb", -1, TRUE) != CSTR_EQUAL ||
      CompareStringOrdinal(dll.filename().c_str(), -1, L"dbghelp.dll", -1, TRUE) != CSTR_EQUAL)
    fail("explicit local .pdb and dbghelp.dll required");
  const auto pdb_size = std::filesystem::file_size(pdb);
  const auto pdb_time = std::filesystem::last_write_time(pdb);
  if (!pdb_size || pdb_size > (uint64_t(512) << 20))
    fail("PDB byte budget exceeded");
  // Load only the explicitly retained DLL and its local/system dependencies.
  Library library;
  library.value = LoadLibraryExW(dll.c_str(), nullptr,
                                 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!library.value)
    fail("selected DbgHelp could not be loaded");
  std::vector<wchar_t> loaded_library(32768);
  const DWORD library_chars = GetModuleFileNameW(library.value, loaded_library.data(),
                                                 static_cast<DWORD>(loaded_library.size()));
  if (!library_chars || library_chars >= loaded_library.size() ||
      !std::filesystem::equivalent(std::filesystem::canonical(loaded_library.data()), dll))
    fail("selected DbgHelp module path differs from loaded library");
  auto options = library.function<decltype(&SymSetOptions)>("SymSetOptions");
  auto initialize = library.function<decltype(&SymInitializeW)>("SymInitializeW");
  auto load = library.function<decltype(&SymLoadModuleExW)>("SymLoadModuleExW");
  auto module_info = library.function<decltype(&SymGetModuleInfoW64)>("SymGetModuleInfoW64");
  auto from_address = library.function<decltype(&SymFromAddrW)>("SymFromAddrW");
  auto from_line = library.function<decltype(&SymGetLineFromAddrW64)>("SymGetLineFromAddrW64");
  Session session{{}, library.function<decltype(&SymCleanup)>("SymCleanup"), false};
  session.key = &session;
  constexpr DWORD policy = SYMOPT_LOAD_LINES | SYMOPT_EXACT_SYMBOLS | SYMOPT_FAIL_CRITICAL_ERRORS |
                           SYMOPT_IGNORE_NT_SYMPATH | SYMOPT_NO_IMAGE_SEARCH | SYMOPT_NO_PROMPTS |
                           SYMOPT_DISABLE_SYMSRV_AUTODETECT;
  options(policy);
  // Direct PDB loading needs neither an executable search nor a symbol server.
  // A literal local directory replaces environment/default search paths.
  if (!initialize(session.key, pdb.parent_path().c_str(), FALSE))
    fail("file-only symbol session initialization failed");
  session.active = true;
  if (load(session.key, nullptr, pdb.c_str(), nullptr, base, static_cast<DWORD>(size), nullptr,
           0) != base)
    fail("explicit PDB load failed");
  IMAGEHLP_MODULEW64 module{};
  module.SizeOfStruct = sizeof(module);
  const GUID empty_guid{};
  // A direct PDB load has no executable for DbgHelp to match: PdbUnmatched
  // may be set. The consumer binds these native GUID/age fields to the PE's
  // independently parsed RSDS identity before accepting any source lookup.
  if (!module_info(session.key, base, &module) || module.BaseOfImage != base ||
      module.ImageSize != size || (module.SymType != SymPdb && module.SymType != SymDia) ||
      !module.PdbAge || std::memcmp(&module.PdbSig70, &empty_guid, sizeof(GUID)) == 0 ||
      module.DbgUnmatched)
    fail("native PDB module identity unavailable or unmatched");
  // The fixed native array can truncate long paths. Bound termination before
  // handing it to filesystem APIs; equivalent also handles Windows path case.
  const auto loaded_name =
      utf8(module.LoadedPdbName, sizeof(module.LoadedPdbName) / sizeof(wchar_t));
  const auto loaded = std::filesystem::canonical(std::filesystem::u8path(loaded_name));
  if (!std::filesystem::equivalent(loaded, pdb))
    fail("DbgHelp opened another PDB");
  json guid = json::array();
  const auto* guid_bytes = reinterpret_cast<const unsigned char*>(&module.PdbSig70);
  for (size_t i = 0; i < sizeof(GUID); ++i)
    guid.push_back(guid_bytes[i]);
  json rows = json::array();
  for (const auto& value : rvas) {
    const auto rva = value.get<uint64_t>(), address = base + rva;
    json row{{"rva", rva}, {"symbol", nullptr}, {"source", nullptr}};
    // DWORD64 backing supplies SYMBOL_INFOW alignment and trailing name space.
    std::vector<DWORD64> storage((sizeof(SYMBOL_INFOW) + max_names * sizeof(wchar_t) + 7) / 8);
    auto* symbol = reinterpret_cast<SYMBOL_INFOW*>(storage.data());
    symbol->SizeOfStruct = sizeof(SYMBOL_INFOW);
    symbol->MaxNameLen = static_cast<ULONG>(max_names);
    DWORD64 displacement = 0;
    if (from_address(session.key, address, &displacement, symbol)) {
      if (!symbol->NameLen || symbol->NameLen >= max_names || symbol->ModBase != base ||
          symbol->Address < base || symbol->Address > address ||
          displacement != address - symbol->Address)
        fail("invalid native symbol extent");
      row["symbol"] = {{"name", utf8(symbol->Name, max_names)},
                       {"start_rva", symbol->Address - base},
                       {"displacement", displacement}};
    } else
      row["symbol_error"] = GetLastError();
    IMAGEHLP_LINEW64 line{};
    line.SizeOfStruct = sizeof(line);
    DWORD offset = 0;
    if (from_line(session.key, address, &offset, &line)) {
      if (!line.FileName || !line.LineNumber || line.Address < base || line.Address > address ||
          offset != address - line.Address)
        fail("invalid native source line extent");
      row["source"] = {{"path", utf8(line.FileName, max_names)},
                       {"line", line.LineNumber},
                       {"start_rva", line.Address - base},
                       {"displacement", offset},
                       {"text", nullptr}};
    } else
      row["source_error"] = GetLastError();
    rows.push_back(std::move(row));
  }
  if (std::filesystem::file_size(pdb) != pdb_size ||
      std::filesystem::last_write_time(pdb) != pdb_time)
    fail("PDB changed during lookup");
  return {{"schema", "rns8_native_pdb_symbols_v1"},
          {"process_id", GetCurrentProcessId()},
          {"reader_source_sha256", RNS8_PDB_SYMBOLIZE_SOURCE_SHA256},
          {"pdb_path", utf8(loaded.c_str(), 32768)},
          {"dbghelp_path", utf8(loaded_library.data(), loaded_library.size())},
          {"pdb_guid_bytes_le", guid},
          {"pdb_age", module.PdbAge},
          {"image_size", size},
          {"symbols", std::move(rows)},
          {"lookup", "explicit_local_PDB_RVAs_not_a_live_process"},
          {"inlined_callchain_resolved", false},
          {"source_contents_verified", false}};
}
} // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    if (argc == 2 && std::wstring(argv[1]) == L"--build-identity") {
      std::cout << json{{"schema", "rns8_pdb_symbolizer_identity_v1"},
                        {"process_id", GetCurrentProcessId()},
                        {"pointer_bytes", sizeof(void*)},
                        {"source_sha256", RNS8_PDB_SYMBOLIZE_SOURCE_SHA256}}
                       .dump()
                << '\n';
      return 0;
    }
    if (argc != 3 || std::wstring(argv[1]) != L"--input")
      fail("usage: rns8-pdb-symbolize --input local-request.json");
    const auto path = std::filesystem::canonical(argv[2]);
    const auto size = std::filesystem::file_size(path);
    if (!size || size > 65536)
      fail("request exceeds 64 KiB budget");
    std::ifstream stream(path, std::ios::binary);
    std::string text(static_cast<size_t>(size), '\0');
    if (!stream.read(text.data(), static_cast<std::streamsize>(size)) ||
        stream.peek() != std::char_traits<char>::eof())
      fail("request changed during read");
    std::cout << symbolize(json::parse(text)).dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rns8-pdb-symbolize: " << error.what() << '\n';
    return 2;
  }
}
