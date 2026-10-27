#pragma once

#include "backend_common/sha256.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rns8::bench {

struct NativeReferenceVerification {
  std::string method = "not_requested", key, payload_sha256;
  uint64_t checked_cells = 0;
  double elapsed_us = 0;
};

inline void reference_append_word(std::string& bytes, uint64_t word) {
  for (unsigned b = 0; b < 8; ++b) bytes.push_back(static_cast<char>((word >> (8 * b)) & 255));
}

inline uint64_t reference_word(const char* bytes) {
  uint64_t word = 0;
  for (unsigned b = 0; b < 8; ++b) word |= uint64_t(static_cast<unsigned char>(bytes[b])) << (8 * b);
  return word;
}

template<class T> std::string reference_input_digest(const std::vector<T>& values) {
  static_assert((sizeof(T) == 8 || sizeof(T) == 1) && std::is_integral_v<T>);
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  static_assert(!std::is_signed_v<T> || std::numeric_limits<T>::min() == -std::numeric_limits<T>::max() - 1);
  return detail::sha256(std::string_view(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T)));
#else
  std::string bytes;
  bytes.reserve(values.size() * sizeof(T));
  for (T value : values) {
    if constexpr(sizeof(T)==1)bytes.push_back(static_cast<char>(value));
    else reference_append_word(bytes, static_cast<uint64_t>(value));
  }
  return detail::sha256(bytes);
#endif
}

template<class T>
std::string native_reference_key(uint64_t m, uint64_t n, uint64_t k,
                                 const std::vector<T>& a, const std::vector<T>& b) {
  return detail::sha256(std::string("rns8-native-gemm-reference-v1\n") +
      (std::is_signed_v<T> ? "bounded-i64\n" : "bounded-u64\n") +
      std::to_string(m) + "\n" + std::to_string(n) + "\n" + std::to_string(k) + "\n" +
      reference_input_digest(a) + "\n" + reference_input_digest(b) + "\n");
}

inline constexpr uint64_t reference_cache_max_bytes = 128 * 1024 * 1024;
inline constexpr std::size_t reference_cache_header_bytes = 144;

inline std::string read_native_reference(const std::filesystem::path& path,
                                          const std::string& key, uint64_t cells) {
  if (!cells || cells > (reference_cache_max_bytes - reference_cache_header_bytes) / 8)
    throw std::runtime_error("reference cache cell budget");
  const uint64_t size = reference_cache_header_bytes + cells * 8;
  if (size > reference_cache_max_bytes || std::filesystem::file_size(path) != size)
    throw std::runtime_error("reference cache size mismatch");
  std::ifstream file(path, std::ios::binary);
  std::string bytes(static_cast<std::size_t>(size), '\0');
  if (!file.read(bytes.data(), static_cast<std::streamsize>(size)) || file.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("reference cache read failed");
  if (bytes.compare(0, 8, "RNS8REF1") || bytes.compare(8, 64, key) ||
      reference_word(bytes.data() + 136) != cells ||
      detail::sha256(std::string_view(bytes).substr(reference_cache_header_bytes)) != bytes.substr(72, 64))
    throw std::runtime_error("reference cache identity or payload digest mismatch");
  return bytes.substr(reference_cache_header_bytes);
}

inline void write_native_reference(const std::filesystem::path& directory,
                                    const std::string& key, const std::string& payload) {
  if(key.size()!=64 || key.find_first_not_of("0123456789abcdef")!=std::string::npos || payload.empty() ||
      payload.size()%8 || payload.size()>reference_cache_max_bytes-reference_cache_header_bytes)
    throw std::invalid_argument("reference cache key or payload budget");
  std::filesystem::create_directories(directory);
  const auto target = directory / (key + ".bin");
  const auto lock = directory / (key + ".lock");
  if (!std::filesystem::create_directory(lock)) throw std::runtime_error("reference cache population already active");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
  } cleanup{lock};
  // The private lock directory holds an incomplete file until atomic publication.
  const auto temporary = lock / "reference.bin";
  std::string header = "RNS8REF1" + key + detail::sha256(payload);
  reference_append_word(header, payload.size() / 8);
  std::ofstream file(temporary, std::ios::binary);
  file.write(header.data(), static_cast<std::streamsize>(header.size()));
  file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  file.close();
  if (!file) throw std::runtime_error("reference cache write failed");
  if (std::filesystem::exists(target)) {
    if (read_native_reference(target, key, payload.size() / 8) != payload)
      throw std::runtime_error("concurrent reference cache population disagrees");
  } else {
    std::filesystem::rename(temporary, target);
  }
}

enum class ReferenceSemantics {bounded_i64,bounded_u64,wrap64,finite_ring,finite_field,wide_signed,wide_unsigned};
struct ReferenceContract {
  ReferenceSemantics semantics;
  uint32_t modulus=0,limbs=1;
};
inline const char* reference_semantic_name(ReferenceSemantics semantics) {
  switch(semantics) {
    case ReferenceSemantics::bounded_i64:return "bounded-i64";
    case ReferenceSemantics::bounded_u64:return "bounded-u64";
    case ReferenceSemantics::wrap64:return "wrap-u64-mod-2-64";
    case ReferenceSemantics::finite_ring:return "finite-ring-u8";
    case ReferenceSemantics::finite_field:return "finite-field-u8";
    case ReferenceSemantics::wide_signed:return "exact-wide-signed";
    case ReferenceSemantics::wide_unsigned:return "exact-wide-unsigned";
  }
  throw std::invalid_argument("unknown reference semantic");
}
template<class T> void validate_reference_contract(ReferenceContract contract) {
  const bool finite=contract.semantics==ReferenceSemantics::finite_ring || contract.semantics==ReferenceSemantics::finite_field;
  const bool wide=contract.semantics==ReferenceSemantics::wide_signed || contract.semantics==ReferenceSemantics::wide_unsigned;
  const bool sign=contract.semantics==ReferenceSemantics::bounded_i64 || contract.semantics==ReferenceSemantics::wide_signed;
  (void)reference_semantic_name(contract.semantics);
  if(!contract.limbs || contract.limbs>(wide?32u:1u) ||
      (finite?(contract.modulus<2 || contract.modulus>256 || sizeof(T)!=1 || std::is_signed_v<T>):
          (contract.modulus || sizeof(T)!=8 || std::is_signed_v<T>!=sign)))
    throw std::invalid_argument("reference semantic/input/output contract mismatch");
  if(contract.semantics==ReferenceSemantics::finite_field)
    for(uint32_t q=2;q*q<=contract.modulus;++q)if(contract.modulus%q==0)throw std::invalid_argument("field reference needs prime modulus");
}
template<class T> std::string integer_reference_key(uint64_t m,uint64_t n,uint64_t k,
    const std::vector<T>& a,const std::vector<T>& b,ReferenceContract contract) {
  validate_reference_contract<T>(contract);
  // Preserve the existing bounded fixtures byte-for-byte.
  if(contract.semantics==ReferenceSemantics::bounded_i64 || contract.semantics==ReferenceSemantics::bounded_u64)
    return native_reference_key(m,n,k,a,b);
  return detail::sha256(std::string("rns8-integer-gemm-reference-v2\n")+reference_semantic_name(contract.semantics)+"\n"+
      std::to_string(contract.modulus)+"\n"+std::to_string(contract.limbs)+"\n"+std::to_string(sizeof(T))+"\n"+
      std::to_string(m)+"\n"+std::to_string(n)+"\n"+std::to_string(k)+"\n"+
      reference_input_digest(a)+"\n"+reference_input_digest(b)+"\n");
}
template<class T,class Output>
NativeReferenceVerification verify_integer_reference(uint64_t m,uint64_t n,uint64_t k,uint64_t ldc,
    const std::vector<T>& a,const std::vector<T>& b,const std::vector<Output>& c,ReferenceContract contract,
    const std::string& cache_directory,bool populate) {
  static_assert(std::is_integral_v<T> && std::is_integral_v<Output>);
  using boost::multiprecision::cpp_int;
  const auto start=std::chrono::steady_clock::now();
  validate_reference_contract<T>(contract);
  const uint64_t limbs=contract.limbs;
  if(!m || !n || !k || ldc<n || m>UINT64_MAX/k || k>UINT64_MAX/n ||
      m>UINT64_MAX/ldc/limbs || a.size()!=m*k || b.size()!=k*n || c.size()<m*ldc*limbs ||
      m>(reference_cache_max_bytes-reference_cache_header_bytes)/8/limbs/n)
    throw std::invalid_argument("reference cache shape or byte budget");
  const bool finite=contract.semantics==ReferenceSemantics::finite_ring || contract.semantics==ReferenceSemantics::finite_field;
  if(finite)for(const auto* input:{&a,&b})for(T value:*input)if(uint32_t(value)>=contract.modulus)
    throw std::invalid_argument("noncanonical finite reference input");
  NativeReferenceVerification result;result.key=integer_reference_key(m,n,k,a,b,contract);
  std::string payload;const auto path=std::filesystem::path(cache_directory)/(result.key+".bin");
  const bool hit=!cache_directory.empty() && std::filesystem::exists(path);
  if(hit)payload=read_native_reference(path,result.key,m*n*limbs);
  else {
    payload.reserve(size_t(m*n*limbs*8));
    const cpp_int storage=cpp_int(1)<<(64*limbs),mask=(cpp_int(1)<<64)-1;
    const bool sign=contract.semantics==ReferenceSemantics::bounded_i64 || contract.semantics==ReferenceSemantics::wide_signed;
    const cpp_int lower=sign?cpp_int(-(storage>>1)):cpp_int(0),upper=sign?cpp_int((storage>>1)-1):cpp_int(storage-1);
    for(uint64_t row=0;row<m;++row)for(uint64_t col=0;col<n;++col) {
      cpp_int expected=0;
      for(uint64_t q=0;q<k;++q)expected+=cpp_int(a[row*k+q])*b[q*n+col];
      if(finite)expected%=contract.modulus;
      else if(contract.semantics==ReferenceSemantics::wrap64)expected&=mask;
      else if(expected<lower || expected>upper)throw std::overflow_error("exact reference exceeds declared output width");
      if(expected<0)expected+=storage;
      for(uint32_t word=0;word<limbs;++word){reference_append_word(payload,static_cast<uint64_t>(expected&mask));expected>>=64;}
    }
  }
  for(uint64_t row=0;row<m;++row)for(uint64_t col=0;col<n;++col)for(uint32_t word=0;word<limbs;++word)
    if(reference_word(payload.data()+((row*n+col)*limbs+word)*8)!=static_cast<uint64_t>(c[(row*ldc+col)*limbs+word]))
      throw std::runtime_error("public output disagrees with complete integer reference");
  if(!hit && populate) {
    if(cache_directory.empty())throw std::invalid_argument("reference population requires a directory");
    write_native_reference(cache_directory,result.key,payload);
  }
  result.method=hit?"cached_full_integer_reference":"boost_cpp_int_full_integer_reference";
  result.payload_sha256=detail::sha256(payload);result.checked_cells=m*n;
  result.elapsed_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();return result;
}
template<class T>
NativeReferenceVerification verify_native_reference(uint64_t m,uint64_t n,uint64_t k,uint64_t ldc,
    const std::vector<T>& a,const std::vector<T>& b,const std::vector<T>& c,const std::string& directory,bool populate) {
  static_assert(sizeof(T)==8 && std::is_integral_v<T>);
  return verify_integer_reference(m,n,k,ldc,a,b,c,
      {std::is_signed_v<T>?ReferenceSemantics::bounded_i64:ReferenceSemantics::bounded_u64},directory,populate);
}
}  // namespace rns8::bench
