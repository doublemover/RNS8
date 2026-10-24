#include "backend_hip_direct/hip_backend.hpp"
#include "core/hip_resources.hpp"
#include "core/internal.hpp"
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace {
using namespace rns8::detail;
using boost::multiprecision::cpp_int;
using Clock=std::chrono::steady_clock;
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };
void check(rns8_status status) {
  if(status==RNS8_UNSUPPORTED_BACKEND) throw Unsupported(rns8_status_string(status));
  if(status!=RNS8_SUCCESS) throw std::runtime_error(rns8_status_string(status));
}
uint32_t number(const char* text,uint32_t low,uint32_t high) {
  uint32_t value=0;const auto end=text+std::strlen(text);
  const auto parsed=std::from_chars(text,end,value);
  if(parsed.ec!=std::errc{} || parsed.ptr!=end || value<low || value>high)
    throw std::invalid_argument("numeric argument outside supported range");
  return value;
}
uint64_t random_word(uint64_t& state) {
  uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
  z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
  return z^(z>>31);
}
std::string checksum(const void* data,size_t size) {
  uint64_t value=UINT64_C(14695981039346656037);const auto* bytes=static_cast<const uint8_t*>(data);
  for(size_t i=0;i<size;++i) value=(value^bytes[i])*UINT64_C(1099511628211);
  std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<value;return out.str();
}
struct Buffers {
  hip_direct_device_buffer native,plain,tile,output,status;
  void allocate(uint64_t cells,uint64_t prefix,uint64_t tile_bytes,uint64_t output_bytes) {
    check(native.allocate(0,cells*8));check(plain.allocate(0,cells*prefix));
    if(tile_bytes) check(tile.allocate(0,tile_bytes));
    check(output.allocate(0,output_bytes));check(status.allocate(0,4));
  }
  void release() {
    check(status.reset());check(output.reset());check(tile.reset());check(plain.reset());check(native.reset());
  }
};
}
int main(int argc,char** argv) {
  try {
    const char* usage="usage: rns8-crt-calibrate <garner|direct|balanced|fixed_dot4|fixed_wmma|tile_balanced|tile_convert_balanced|tile_convert_garner|import_scalar_signed|import_scalar_unsigned|import_dot4_signed|import_dot4_unsigned|import_wmma_signed|import_wmma_unsigned> cells1..1048576 prefix1..20 repeats1..10000 warmups1..1000 unique-run-id [resident|cold]\nresident is default; cold includes buffer allocation, host upload, conversion, output readback and checked free. Explicit HIP device0; no fallback.";
    if(argc==2 && std::string(argv[1])=="--help") {std::cout<<usage<<'\n';return 0;}
    if(argc!=7 && argc!=8) throw std::invalid_argument(usage);
    const std::string candidate=argv[1],run_id=argv[6],scope=argc==8?argv[7]:"resident";
    const bool importing=candidate.rfind("import_",0)==0, signed_values=candidate=="import_scalar_signed" || candidate=="import_dot4_signed" || candidate=="import_wmma_signed";
    const bool dot4=candidate=="fixed_dot4" || candidate=="import_dot4_signed" || candidate=="import_dot4_unsigned";
    const bool wmma=candidate=="fixed_wmma" || candidate=="import_wmma_signed" || candidate=="import_wmma_unsigned";
    const bool tile_input=candidate.rfind("tile_",0)==0;
    if((candidate!="garner" && candidate!="direct" && candidate!="balanced" && candidate!="fixed_dot4" && candidate!="fixed_wmma" &&
        candidate!="tile_balanced" && candidate!="tile_convert_balanced" && candidate!="tile_convert_garner" &&
        candidate!="import_scalar_signed" && candidate!="import_scalar_unsigned" && candidate!="import_dot4_signed" && candidate!="import_dot4_unsigned" && candidate!="import_wmma_signed" && candidate!="import_wmma_unsigned") ||
        (scope!="resident" && scope!="cold") || run_id.empty() || run_id.size()>128 ||
        run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")!=std::string::npos)
      throw std::invalid_argument("invalid candidate/scope/run identity");
    const auto cells=number(argv[2],1,1048576),prefix=number(argv[3],1,20),
               repeats=number(argv[4],1,10000),warmups=number(argv[5],1,1000);
    const uint64_t cols=cells%256==0?256:cells,rows=cells/cols;
    ResidueLayout plain,tiled;
    if(!make_residue_layout(ResidueLayoutKind::modulus_major,rows,cols,cols,prefix,0,0,plain) ||
       !make_residue_layout(ResidueLayoutKind::tile_major,rows,cols,cols,prefix,8,16,tiled))
      throw std::invalid_argument("conversion layout overflow");
    const uint64_t tile_bytes=tile_input?tiled.bytes:0,output_bytes=uint64_t(cells)*24+64;
    const uint64_t source_bytes=importing?uint64_t(cells)*8:tile_input?tile_bytes:plain.bytes;
    // Native values, residue planes, three full output/canary vectors, tile
    // staging, common device buffers and active-input readback. Fixed metadata,
    // allocator bookkeeping, context/runtime/event objects are excluded.
    const uint64_t envelope=uint64_t(cells)*16+plain.bytes*2+output_bytes*4+tile_bytes*2+source_bytes+16384;
    constexpr uint64_t budget=UINT64_C(768)*1024*1024;
    if(envelope>budget) throw std::invalid_argument("aggregate payload exceeds768MiB");
    const auto oracle_start=Clock::now();
    std::vector<uint64_t> native(cells);
    std::vector<int8_t> residues(plain.bytes),tile_residues;
    std::vector<uint8_t> expected(output_bytes,99),observed(output_bytes,99),poison(output_bytes,99),source_observed(source_bytes);
    const cpp_int product=modulus_product(prefix);uint64_t state=3903;
    const uint64_t edges[]={0,1,UINT64_MAX,UINT64_C(1)<<63,UINT64_C(0x8080808080808080),UINT64_C(0x7f8081ff007fff80)};
    for(uint32_t cell=0;cell<cells;++cell) {
      const uint64_t r0=random_word(state),r1=random_word(state),r2=random_word(state);
      native[cell]=cell%257<6?edges[cell%257]:r0;
      cpp_int value;
      if(importing) {value=native[cell];if(signed_values && (native[cell]>>63)) value-=cpp_int(1)<<64;}
      else {value=(cpp_int(r0)+(cpp_int(r1)<<64)+(cpp_int(r2)<<128))%product;
        if(cell==0)value=0;if(cell==1)value=product-1;if(cell==2)value=product/2;}
      for(uint32_t p=0;p<prefix;++p) residues[size_t(p)*cells+cell]=centered_residue(value,kDefaultModuli[p]);
      if(!importing) for(unsigned limb=0;limb<3;++limb) {
        const uint64_t word=static_cast<uint64_t>((value>>(limb*64))&UINT64_MAX);
        for(unsigned byte=0;byte<8;++byte) expected[size_t(cell)*24+limb*8+byte]=uint8_t(word>>(byte*8));
      }
    }
    if(importing) std::memcpy(expected.data(),residues.data(),residues.size());
    if(tile_input) {tile_residues.assign(tile_bytes,99);
      if(!convert_residue_layout_host(residues.data(),plain,tile_residues.data(),tiled)) throw std::runtime_error("host tile conversion failed");}
    const void* host_source=importing?static_cast<const void*>(native.data()):tile_input?static_cast<const void*>(tile_residues.data()):static_cast<const void*>(residues.data());
    const auto input_checksum=checksum(host_source,source_bytes),output_checksum=checksum(expected.data(),expected.size());
    const double oracle_ns=std::chrono::duration<double,std::nano>(Clock::now()-oracle_start).count();
    rns8_device_info device{};check(hip_direct_probe(0,device));
    const auto initialize=[&](Buffers& buffers) {
      buffers.allocate(cells,prefix,tile_bytes,output_bytes);
      check(hip_direct_copy_host_to_device(0,buffers.native.get(),native.data(),native.size()*8));
      check(hip_direct_copy_host_to_device(0,buffers.plain.get(),residues.data(),residues.size()));
      if(tile_input) check(hip_direct_copy_host_to_device(0,buffers.tile.get(),tile_residues.data(),tile_residues.size()));
      check(hip_direct_copy_host_to_device(0,buffers.output.get(),poison.data(),poison.size()));
    };
    const auto convert=[&](Buffers& buffers) {
      if(dot4 || wmma) {bool started=false;const auto status=hip_direct_fixed_conversion_device(0,
          importing?buffers.native.get():buffers.plain.get(),buffers.output.get(),cells,prefix,(importing?(signed_values?0:1):2)+(wmma?3:0),4,started);
        check(status);if(!started) throw std::runtime_error("successful fixed conversion call did not report issued output");return;}
      if(importing) {const auto call=signed_values?hip_direct_native_i64_to_rns_device:hip_direct_native_u64_to_rns_device;
        check(call(0,buffers.native.get(),buffers.output.get(),1,cells,prefix));return;}
      if(candidate=="tile_balanced") {check(hip_direct_export_canonical_balanced_crt_layout_device(0,buffers.tile.get(),tiled,buffers.output.get()));return;}
      if(tile_input) check(hip_direct_convert_residue_layout(0,buffers.tile.get(),tiled,buffers.plain.get(),plain));
      if(candidate=="direct") check(hip_direct_export_canonical_direct_crt_device(0,buffers.plain.get(),buffers.output.get(),cells,prefix));
      else if(candidate=="balanced" || candidate=="tile_convert_balanced") check(hip_direct_export_canonical_balanced_crt_device(0,buffers.plain.get(),buffers.output.get(),cells,prefix));
      else {void* status=buffers.status.get();size_t size=4;
        check(hip_direct_export_exact_wide_unsigned_limbs_to_device(0,buffers.plain.get(),buffers.output.get(),&status,&size,1,cells,prefix,3));
        if(status!=buffers.status.get() || size!=4) throw std::runtime_error("unexpected Garner status reallocation");}
    };
    Buffers resident;hip_direct_timing_set_enabled(false);
    const auto setup_start=Clock::now();if(scope=="resident") initialize(resident);
    const double setup_ns=std::chrono::duration<double,std::nano>(Clock::now()-setup_start).count();
    const auto audit_source=[&](Buffers& buffers) {
      const void* source=importing?buffers.native.get():tile_input?buffers.tile.get():buffers.plain.get();
      check(hip_direct_copy_device_to_host(0,source_observed.data(),source,source_bytes));
      if(std::memcmp(source_observed.data(),host_source,source_bytes)) throw std::runtime_error("conversion changed active input");
    };
    const auto execute=[&]() {
      if(scope=="resident") convert(resident);
      else {Buffers cold;initialize(cold);convert(cold);
        check(hip_direct_copy_device_to_host(0,observed.data(),cold.output.get(),observed.size()));
        // Input preservation readback remains outside the production interval:
        // cold inputs are freed here; their read-only behavior is covered by the
        // independent resident audit below and authored kernel differentials.
        cold.release();}
    };
    const auto prepare=[&]() {if(scope=="resident") check(hip_direct_copy_host_to_device(0,resident.output.get(),poison.data(),poison.size()));};
    const auto verify=[&]() {
      if(scope=="resident") {check(hip_direct_copy_device_to_host(0,observed.data(),resident.output.get(),observed.size()));audit_source(resident);}
      if(observed!=expected) throw std::runtime_error("complete conversion output or tail canary differs from Boost");
    };
    // A separate resident call audits the active source even for cold captures;
    // it is outside setup/production timings and its buffers are released first.
    if(scope=="cold") {Buffers audit;initialize(audit);convert(audit);audit_source(audit);audit.release();}
    for(uint32_t i=0;i<warmups;++i) {prepare();execute();verify();}
    nlohmann::json samples=nlohmann::json::array();
    for(uint32_t i=0;i<repeats;++i) {
      prepare();hip_direct_timing_reset();hip_direct_timing_set_enabled(true);
      auto start=Clock::now();execute();const double instrumented=std::chrono::duration<double,std::nano>(Clock::now()-start).count();
      hip_direct_timing_flush_pending_events();nlohmann::json events=nlohmann::json::array();
      for(const auto& event:hip_direct_timing_snapshot()) events.push_back({{"label",event.label},{"elapsed_ns",event.microseconds>0?nlohmann::json(event.microseconds*1000):nlohmann::json(nullptr)}});
      hip_direct_timing_set_enabled(false);verify();prepare();start=Clock::now();execute();
      const double production=std::chrono::duration<double,std::nano>(Clock::now()-start).count();verify();
      samples.push_back({{"host_call_ns",instrumented},{"uninstrumented_host_call_ns",production},{"events",events},
                         {"exact_reference_passed",true},{"uninstrumented_exact_reference_passed",true}});
    }
    hip_direct_timing_set_enabled(false);prepare();uint32_t calls=0;double elapsed=0;const auto start=Clock::now();
    do {execute();++calls;elapsed=std::chrono::duration<double,std::nano>(Clock::now()-start).count();}
    while(calls<4096 && (calls<16 || elapsed<250000000));verify();
    nlohmann::json report={{"schema","rns8_crt_microcase_v2"},{"run_id",run_id},{"candidate",candidate},{"scope",scope},
      {"cells",cells},{"prefix",prefix},{"seed",3903},{"dataset_recipe","splitmix64_conversion_seed3903_extrema_v1"},
      {"repeats",repeats},{"warmups",warmups},{"device_requested",0},{"device_actual",0},
      {"backend_requested","hip-direct"},{"backend_actual","hip-direct"},{"device",device.name},{"target",device.gcn_arch},
      {"runtime_version",device.hip_runtime_version},{"build_manifest",nlohmann::json::parse(RNS8_BUILD_MANIFEST_JSON)},
      {"timer","std_chrono_steady_clock_nanoseconds"},{"production_timing_hooks_enabled",false},
      {"public_ownership_measured",false},{"complete_output_and_canaries_checked",true},
      {"input_preservation_scope",scope=="cold"?"one_separate_resident_audit_before_warmups":"outside_each_verified_call_and_final_burst"},
      {"operation",importing?"native_import":"canonical_crt"},{"signed_inputs",importing?nlohmann::json(signed_values):nlohmann::json(nullptr)},
      {"input_contract",importing?"native64_bits":tile_input?"tile_centered_residue_planes":"centered_residue_planes"},
      {"output_contract",importing?"centered_residue_planes":"canonical_three_limbs"},{"output_boundary",scope=="cold"?"host":"device"},
      {"reference_method",importing?"boost_cpp_int_all_operand_residues":"boost_cpp_int_all_canonical_values"},
      {"input_checksum",input_checksum},{"output_checksum",output_checksum},{"canary_bytes",64},
      {"source_bytes_modeled",source_bytes},{"source_layout",importing?"native64_compact_v1":tile_input?"tile_major_8x16_v1":"modulus_major_v1"},
      {"source_physical_bytes",source_bytes},{"logical_rows",rows},{"logical_cols",cols},
      {"conversion_inside_call",tile_input && candidate!="tile_balanced"},
      {"conversion_scratch_bytes",tile_input && candidate!="tile_balanced"?plain.bytes:0},
      {"output_bytes_modeled",uint64_t(cells)*(importing?prefix:24)},{"output_allocation_bytes",output_bytes},
      {"probe_bytes",(dot4 || wmma)?4:0},{"common_status_capacity_bytes",4},{"logical_payload_envelope_bytes",envelope},{"memory_budget_bytes",budget},
      {"oracle_and_dataset_ns",oracle_ns},{"resident_setup_ns",scope=="resident"?nlohmann::json(setup_ns):nlohmann::json(nullptr)},
      {"timing_scope",scope=="cold"?"allocate_upload_convert_readback_checked_free":"resident_checked_convert_with_sync_no_upload_or_readback"},
      {"cache_state",scope=="cold"?"fresh_buffers_each_call":"repeated_same_buffers"},{"actual_hbm_bytes",nullptr},
      {"timing_resolution_ns",nullptr},{"isa_verified",false},{"promotion_eligible",false},
      {"samples",samples},{"sustained",{{"calls",calls},{"minimum_calls",16},{"maximum_calls",4096},{"target_ns",250000000},{"independent_audit_inside_interval",false},{"elapsed_ns",elapsed},{"mean_call_ns",elapsed/calls},{"final_exact_reference_passed",true},
        {"scope",scope=="cold"?"fresh_buffers_transfer_inclusive_no_intermediate_reference":"resident_same_buffers_no_intermediate_poison_or_reference"}}}};
    resident.release();std::cout<<report.dump(2)<<'\n';return 0;
  } catch(const Unsupported& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 77;}
  catch(const std::exception& error) {hip_direct_timing_set_enabled(false);std::cerr<<error.what()<<'\n';return 1;}
}
