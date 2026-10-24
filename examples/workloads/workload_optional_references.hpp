#ifndef RNS8_WORKLOAD_OPTIONAL_REFERENCES_HPP
#define RNS8_WORKLOAD_OPTIONAL_REFERENCES_HPP
#include "workload_baselines.hpp"
#if defined(RNS8_WORKLOAD_HAVE_GMP) || defined(RNS8_WORKLOAD_HAVE_FLINT)
#include <gmp.h>
#endif
#if defined(RNS8_WORKLOAD_HAVE_FLINT)
#include <flint/flint.h>
#include <flint/fmpz_mat.h>
#endif

namespace rns8_workload {
inline Json optional_reference_identity() {
  Json configured=Json::object();
#ifdef RNS8_WORKLOAD_REFERENCE_IDENTITY_JSON
  configured=Json::parse(RNS8_WORKLOAD_REFERENCE_IDENTITY_JSON);
#endif
  Json result={{"version",1},{"gmp",{{"enabled",false},{"status","unavailable"},
      {"reason","GMP workload adapter not enabled in this executable"}}},
    {"flint",{{"enabled",false},{"status","unavailable"},
      {"reason","FLINT workload adapter not enabled in this executable"}}}};
#if defined(RNS8_WORKLOAD_HAVE_GMP) || defined(RNS8_WORKLOAD_HAVE_FLINT)
  const std::string gmp_header=std::to_string(__GNU_MP_VERSION)+"."+std::to_string(__GNU_MP_VERSION_MINOR)+"."+std::to_string(__GNU_MP_VERSION_PATCHLEVEL);
  require(gmp_version && gmp_header==gmp_version,"selected GMP headers/runtime version differ");
#endif
#if defined(RNS8_WORKLOAD_HAVE_GMP)
  result["gmp"]={{"enabled",true},{"status","available"},{"header_version",gmp_header},
    {"runtime_version",gmp_version},{"artifacts",configured.at("GMP")}};
#endif
#if defined(RNS8_WORKLOAD_HAVE_FLINT)
  require(std::string(FLINT_VERSION)==flint_version,"selected FLINT headers/runtime version differ");
  result["flint"]={{"enabled",true},{"status","available"},{"header_version",FLINT_VERSION},
    {"runtime_version",flint_version},{"artifacts",configured.at("FLINT")},
    {"gmp_header_version",gmp_header},{"gmp_runtime_version",gmp_version},{"gmp_artifacts",configured.at("GMP")}};
#endif
  return result;
}

inline uint64_t optional_reference_memory(const Options& options,bool chain) {
  require(options.m>=1 && options.m<=512 && options.n>=1 && options.n<=512 && options.k>=1 && options.k<=512,
      "optional comparator shape exceeds1..512");
  const uint64_t cells=uint64_t(options.m)*options.k+uint64_t(options.k)*options.n+
      uint64_t(options.m)*options.n+(chain?uint64_t(options.n)*options.n+uint64_t(options.m)*options.n:0);
  // Charge each optional adapter separately, in addition to common inputs and
  // Boost/native references. Library-private multiplication temporaries are
  // allocator/algorithm dependent; this is an explicit model, not a peak claim.
  const uint64_t serialization=uint64_t(options.m)*options.n*3*8;
  require(cells<=(UINT64_MAX-serialization-4096)/1024,"optional comparator memory estimate overflow");
  require(options.memory && options.memory_mib>=1 && options.memory_mib<=4096,"invalid optional comparator memory budget");
  const uint64_t bytes=1024*cells+serialization+4096;
  options.memory->reserve(bytes,options.memory_mib*1024*1024,"optional exact comparator state and workspace allowance");
  return bytes;
}
inline bool optional_reference_shape(const Dataset& data,const Options& options,bool chain,uint32_t limbs) {
  require(options.m>=1 && options.m<=512 && options.n>=1 && options.n<=512 && options.k>=1 && options.k<=512,
      "optional comparator shape exceeds1..512");
  require(limbs==1 || limbs==3,"optional comparator supports explicit one/three-limb signed outputs");
  require(data.a.size()==size_t(options.m)*options.k && data.b.size()==size_t(options.k)*options.n &&
      (!chain || data.t.size()==size_t(options.n)*options.n),"optional comparator input shape mismatch");
  return true;
}
inline Json optional_comparator_contract(const std::string& library,bool chain,uint32_t limbs,uint32_t reuse=1) {
  return {{"version",1},{"library",library},{"arithmetic",library=="gmp"?"blocked_mpz_addmul_tile32":"fmpz_mat_mul_library_dispatch"},
    {"application_boundary","native_signed64_host_inputs_to_complete_signed_host_words"},
    {"input_conversion_per_call",true},{"retained_result_reuse",false},{"retained_allocations",true},
    {"intermediate_serialization",false},{"chain",chain},{"real_gemm_products_per_call",chain?2:reuse},
    {"output_limbs_per_cell",limbs},{"threads",1},{"output_conversion","checked_twos_complement_u64_limbs"},
    {"library_private_peak_bytes",nullptr},{"actual_peak_measured",false}};
}

#if defined(RNS8_WORKLOAD_HAVE_GMP) || defined(RNS8_WORKLOAD_HAVE_FLINT)
struct WorkloadMpz {
  mpz_t value;
  WorkloadMpz() {mpz_init2(value,256);}
  ~WorkloadMpz() {mpz_clear(value);}
  WorkloadMpz(const WorkloadMpz&)=delete; WorkloadMpz& operator=(const WorkloadMpz&)=delete;
};
inline void native_to_mpz(mpz_ptr output,int64_t value) {
  const uint64_t absolute=magnitude(value);
  // unsigned long is only32 bits on Windows; never use set_si for full64 input.
  mpz_import(output,1,1,sizeof(absolute),0,0,&absolute);
  if(value<0) mpz_neg(output,output);
}
struct OptionalWordEncoder {
  WorkloadMpz positive_limit,negative_limit,encoded;
  uint32_t limbs;
  explicit OptionalWordEncoder(uint32_t width):limbs(width) {
    require(width==1 || width==3,"optional output width must be one or three limbs");
    mpz_set_ui(positive_limit.value,1); mpz_mul_2exp(positive_limit.value,positive_limit.value,64*limbs-1);
    mpz_neg(negative_limit.value,positive_limit.value);
  }
  void store(mpz_srcptr value,Words& words,size_t cell) {
    require(words.size()%limbs==0 && cell<words.size()/limbs,"optional output word capacity is insufficient");
    require(mpz_cmp(value,negative_limit.value)>=0 && mpz_cmp(value,positive_limit.value)<0,
        "optional exact comparator output exceeds requested signed width");
    auto* destination=words.data()+cell*limbs; std::fill(destination,destination+limbs,0);
    mpz_fdiv_r_2exp(encoded.value,value,64*limbs);
    require((mpz_sizeinbase(encoded.value,2)+63)/64<=limbs,"optional output exceeds word capacity before export");
    size_t count=0; mpz_export(destination,&count,-1,sizeof(uint64_t),0,0,encoded.value);
    require(count<=limbs,"optional output serialization exceeded capacity");
  }
};
#endif

#if defined(RNS8_WORKLOAD_HAVE_GMP)
struct GmpBaseline {
  const Dataset& data; const Options& options; bool chain;
  bool shape_admitted;
  std::unique_ptr<WorkloadMpz[]> a,b,t,first,final;
  OptionalWordEncoder encoder; Words words;
  GmpBaseline(const Dataset& input,const Options& config,bool chained,uint32_t limbs):data(input),options(config),chain(chained),
    shape_admitted(optional_reference_shape(input,config,chained,limbs)),
    a(std::make_unique<WorkloadMpz[]>(input.a.size())),b(std::make_unique<WorkloadMpz[]>(input.b.size())),
    t(chained?std::make_unique<WorkloadMpz[]>(input.t.size()):nullptr),
    first(std::make_unique<WorkloadMpz[]>(size_t(config.m)*config.n)),
    final(chained?std::make_unique<WorkloadMpz[]>(size_t(config.m)*config.n):nullptr),encoder(limbs),words(size_t(config.m)*config.n*limbs) {}
  static void multiply(const WorkloadMpz* a,const WorkloadMpz* b,WorkloadMpz* c,uint32_t m,uint32_t n,uint32_t k) {
    for(size_t cell=0;cell<size_t(m)*n;++cell) mpz_set_ui(c[cell].value,0);
    for(uint32_t ii=0;ii<m;ii+=32) for(uint32_t qq=0;qq<k;qq+=32) for(uint32_t jj=0;jj<n;jj+=32)
      for(uint32_t row=ii;row<std::min(m,ii+32);++row) for(uint32_t q=qq;q<std::min(k,qq+32);++q)
        for(uint32_t col=jj;col<std::min(n,jj+32);++col)
          mpz_addmul(c[size_t(row)*n+col].value,a[size_t(row)*k+q].value,b[size_t(q)*n+col].value);
  }
  void execute(Phases& phases,uint64_t) {
    phases.measure("optional_input_conversion_ns",[&] {
      for(size_t i=0;i<data.a.size();++i) native_to_mpz(a[i].value,data.a[i]);
      for(size_t i=0;i<data.b.size();++i) native_to_mpz(b[i].value,data.b[i]);
      if(chain) for(size_t i=0;i<data.t.size();++i) native_to_mpz(t[i].value,data.t[i]);
    });
    phases.measure("optional_exact_gemm_ns",[&] {
      multiply(a.get(),b.get(),first.get(),options.m,options.n,options.k);
      if(chain) multiply(first.get(),t.get(),final.get(),options.m,options.n,options.n);
    });
    phases.measure("optional_output_serialization_ns",[&] {
      for(size_t cell=0;cell<size_t(options.m)*options.n;++cell)
        encoder.store((chain?final:first)[cell].value,words,cell);
    });
  }
};
#endif

#if defined(RNS8_WORKLOAD_HAVE_FLINT)
struct WorkloadFlintMatrix {
  fmpz_mat_t value;
  WorkloadFlintMatrix(uint32_t rows,uint32_t cols) {fmpz_mat_init(value,rows,cols);}
  ~WorkloadFlintMatrix() {fmpz_mat_clear(value);}
  WorkloadFlintMatrix(const WorkloadFlintMatrix&)=delete; WorkloadFlintMatrix& operator=(const WorkloadFlintMatrix&)=delete;
};
struct FlintBaseline {
  const Dataset& data; const Options& options; bool chain;
  bool shape_admitted;
  WorkloadFlintMatrix a,b,t,first,final;
  WorkloadMpz conversion; OptionalWordEncoder encoder; Words words;
  FlintBaseline(const Dataset& input,const Options& config,bool chained,uint32_t limbs):data(input),options(config),chain(chained),
    shape_admitted(optional_reference_shape(input,config,chained,limbs)),
    a(config.m,config.k),b(config.k,config.n),t(chained?config.n:0,chained?config.n:0),
    first(config.m,config.n),final(chained?config.m:0,chained?config.n:0),encoder(limbs),words(size_t(config.m)*config.n*limbs) {
    flint_set_num_threads(1); require(flint_get_num_threads()==1,"FLINT thread selection failed");
  }
  void import(WorkloadFlintMatrix& target,const std::vector<int64_t>& values,uint32_t rows,uint32_t cols) {
    for(uint32_t row=0;row<rows;++row) for(uint32_t col=0;col<cols;++col) {
      native_to_mpz(conversion.value,values[size_t(row)*cols+col]);
      fmpz_set_mpz(fmpz_mat_entry(target.value,row,col),conversion.value);
    }
  }
  void execute(Phases& phases,uint64_t) {
    phases.measure("optional_input_conversion_ns",[&] {
      import(a,data.a,options.m,options.k); import(b,data.b,options.k,options.n);
      if(chain) import(t,data.t,options.n,options.n);
    });
    phases.measure("optional_exact_gemm_ns",[&] {
      fmpz_mat_mul(first.value,a.value,b.value);
      if(chain) fmpz_mat_mul(final.value,first.value,t.value);
    });
    phases.measure("optional_output_serialization_ns",[&] {
      for(uint32_t row=0;row<options.m;++row) for(uint32_t col=0;col<options.n;++col) {
        fmpz_get_mpz(conversion.value,fmpz_mat_entry(chain?final.value:first.value,row,col));
        encoder.store(conversion.value,words,size_t(row)*options.n+col);
      }
    });
  }
};
#endif

inline void add_optional_integer_baselines(std::vector<Variant>& variants,Json& skipped,const Dataset& data,
    const Options& options,uint32_t limbs,bool chain=false) {
  const auto libraries=optional_reference_identity();
  for(const std::string library:{"gmp","flint"}) {
    const std::string name=(library=="gmp"?"gmp_mpz_blocked":"flint_fmpz_mat_mul")+std::string(chain?"_chain":"");
    if(!libraries[library]["enabled"].get<bool>()) {
      skipped.push_back({{"name",name},{"status","unavailable"},{"reason",libraries[library]["reason"]}}); continue;
    }
    const uint64_t memory=optional_reference_memory(options,chain);
    Variant variant;
#if defined(RNS8_WORKLOAD_HAVE_GMP)
    if(library=="gmp") variant=state_variant<GmpBaseline>(name,data,options,chain,limbs);
#endif
#if defined(RNS8_WORKLOAD_HAVE_FLINT)
    if(library=="flint") variant=state_variant<FlintBaseline>(name,data,options,chain,limbs);
#endif
    require(bool(variant.execute),"enabled optional adapter is missing");
    variant.metadata["optional_exact_reference"]=optional_comparator_contract(library,chain,limbs,std::max(options.gram_reuse,1u));
    variant.metadata["optional_exact_reference"]["identity"]=libraries[library];
    variant.metadata["optional_exact_reference"]["modeled_memory_allowance_bytes"]=memory;
    variants.push_back(std::move(variant));
  }
}
} // namespace rns8_workload
#endif
