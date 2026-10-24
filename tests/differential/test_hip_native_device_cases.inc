#include "../support/native_device_fixture.hpp"
TEST_CASE("native HIP leases import and export full-width padded views without host payload") {
  if(!hip_available()) SKIP("no HIP device for native leases");
  using namespace rns8::detail;
  using namespace native_device_fixture;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool signed_values:{false,true}) for(bool retain:{false,true}) {
    hip_direct_device_buffer input,output;
    std::vector<uint64_t> values(24,99),actual(24,77);
    const uint64_t edges[]={0,1,UINT64_MAX,UINT64_C(1)<<63,UINT64_C(0x7fffffffffffffff)};
    for(int row=0;row<3;++row) for(int col=0;col<5;++col) values[1+row*7+col]=edges[(row+col)%5];
    REQUIRE(input.allocate(0,values.size()*8)==RNS8_SUCCESS);REQUIRE(output.allocate(0,actual.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,output.get(),actual.data(),actual.size()*8)==RNS8_SUCCESS);
    Owner reader,writer;auto in=descriptor(input.get(),values.size()*8,reader),out=descriptor(output.get(),actual.size()*8,writer);
    in.element_type=out.element_type=signed_values?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64;
    out.access=RNS8_NATIVE_DEVICE_WRITE;out.owner_identity=202;
    rns8_native_device_lease *read=nullptr,*write=nullptr;
    REQUIRE(rns8_create_native_device_lease(ctx,&in,&read)==RNS8_SUCCESS);
    REQUIRE(rns8_create_native_device_lease(ctx,&out,&write)==RNS8_SUCCESS);
    REQUIRE(reader.acquired==1);REQUIRE(writer.acquired==1);REQUIRE(reader.released==0);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_BUSY);
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
    md.rows=3;md.cols=5;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
    md.flags=retain?RNS8_MATRIX_RETAIN_NATIVE_INPUT:0;
    md.semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;md.bound_kind=RNS8_BOUND_NONE;
    rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_import_native_device(ctx,matrix,write)==RNS8_INVALID_ARGUMENT);
    REQUIRE(rns8_import_native_device(ctx,matrix,read)==RNS8_SUCCESS);
    REQUIRE(matrix->device_residues_current);REQUIRE_FALSE(matrix->host_residues_current);
    REQUIRE(matrix->device_native_current==retain);
    REQUIRE(matrix->native_i64.empty());REQUIRE(matrix->native_u64.empty());
    void* native=signed_values?matrix->hip_native_i64:matrix->hip_native_u64;
    if(retain) {
      REQUIRE(native!=input.get());REQUIRE(native!=nullptr);
      std::vector<uint64_t> compact(15);
      REQUIRE(hip_direct_copy_device_to_host(0,compact.data(),native,120)==RNS8_SUCCESS);
      for(int row=0;row<3;++row) for(int col=0;col<5;++col)
        REQUIRE(compact[row*5+col]==values[1+row*7+col]);
    }
    REQUIRE_FALSE(matrix->in_flight_writer);REQUIRE(matrix->in_flight_readers==0);REQUIRE(matrix->prepared_references==0);
    REQUIRE(matrix->exact_range_prefix==20);REQUIRE_FALSE(matrix->imported_source);REQUIRE(matrix->source_version==matrix->content_generation);
    const auto before_verified=matrix->content_generation;
    REQUIRE(rns8_import_native_device_verified(ctx,matrix,read,7)==RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(matrix->content_generation==before_verified);
    REQUIRE(rns8_import_native_device_verified(ctx,matrix,read,8)==RNS8_SUCCESS);
    REQUIRE(matrix->exact_max_magnitude==(signed_values?(boost::multiprecision::cpp_int(1)<<63):boost::multiprecision::cpp_int(UINT64_MAX)));
    std::vector<int8_t> packed(20*15);
    REQUIRE(hip_direct_copy_device_to_host(0,packed.data(),matrix->hip_residues,packed.size())==RNS8_SUCCESS);
    for(unsigned plane=0;plane<20;++plane) for(int row=0;row<3;++row) for(int col=0;col<5;++col) {
      const auto bits=values[1+row*7+col];boost::multiprecision::cpp_int value=bits;
      if(signed_values && (bits>>63)) value-=boost::multiprecision::cpp_int(1)<<64;
      REQUIRE(packed[plane*15+row*5+col]==centered_residue(value,kDefaultModuli[plane]));
    }
    rns8_native_device_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_query_native_device(&out,&resources)==RNS8_SUCCESS);
    REQUIRE(rns8_export_native_device(ctx,matrix,write,resources.export_scratch_bytes-1)==RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output.get(),actual.size()*8)==RNS8_SUCCESS);
    for(auto value:actual) REQUIRE(value==77);
    REQUIRE(rns8_export_native_device(ctx,matrix,write,resources.export_scratch_bytes)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output.get(),actual.size()*8)==RNS8_SUCCESS);
    for(size_t i=0;i<actual.size();++i) {
      const bool logical=i>=1 && i<20 && (i-1)%7<5;
      REQUIRE(actual[i]==(logical?values[i]:77));
    }
    // Release before external mutation. Recycled address gets a new owner id;
    // importing again cannot hit host-pack's source-version shortcut.
    REQUIRE(rns8_destroy_native_device_lease(read)==RNS8_SUCCESS);REQUIRE(reader.released==1);
    for(int row=0;row<3;++row) for(int col=0;col<5;++col) values[1+row*7+col]=2;
    values[1]=7;REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    if(retain) {
      // Releasing/recycling the external source cannot alter the owned snapshot.
      std::vector<uint64_t> compact(15);
      REQUIRE(hip_direct_copy_device_to_host(0,compact.data(),native,120)==RNS8_SUCCESS);
      for(int row=0;row<3;++row) for(int col=0;col<5;++col)
        REQUIRE(compact[row*5+col]==edges[(row+col)%5]);
    }
    in.owner_identity=303;in.generation=2;
    REQUIRE(rns8_create_native_device_lease(ctx,&in,&read)==RNS8_SUCCESS);
    REQUIRE(rns8_import_native_device_verified(ctx,matrix,read,8)==RNS8_SUCCESS);
    REQUIRE(matrix->exact_max_magnitude==7);
    if(retain) {
      REQUIRE((signed_values?matrix->hip_native_i64:matrix->hip_native_u64)==native);
      REQUIRE(matrix->device_native_current);
      std::vector<uint64_t> compact(15);
      REQUIRE(hip_direct_copy_device_to_host(0,compact.data(),native,120)==RNS8_SUCCESS);
      for(int row=0;row<3;++row) for(int col=0;col<5;++col)
        REQUIRE(compact[row*5+col]==values[1+row*7+col]);
    }
    REQUIRE(rns8_export_native_device(ctx,matrix,write,resources.export_scratch_bytes)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output.get(),actual.size()*8)==RNS8_SUCCESS);REQUIRE(actual[1]==7);
    REQUIRE(rns8_destroy_native_device_lease(write)==RNS8_SUCCESS);REQUIRE(writer.released==1);
    REQUIRE(rns8_destroy_native_device_lease(read)==RNS8_SUCCESS);REQUIRE(reader.acquired==2);REQUIRE(reader.released==2);
    Owner alias_owner;auto alias=descriptor(matrix->hip_residues,matrix->hip_residue_bytes,alias_owner);
    alias.element_type=in.element_type;rns8_native_device_lease* alias_lease=nullptr;
    REQUIRE(rns8_create_native_device_lease(ctx,&alias,&alias_lease)==RNS8_SUCCESS);
    const auto generation=matrix->content_generation;
    REQUIRE(rns8_import_native_device(ctx,matrix,alias_lease)==RNS8_INVALID_ARGUMENT);
    REQUIRE(matrix->content_generation==generation);
    REQUIRE(rns8_destroy_native_device_lease(alias_lease)==RNS8_SUCCESS);REQUIRE(alias_owner.released==1);
    if(retain) {
      Owner native_owner;auto same=descriptor(native,120,native_owner);
      same.element_type=in.element_type;same.byte_offset=0;same.ld=5;
      rns8_native_device_lease* same_lease=nullptr;
      REQUIRE(rns8_create_native_device_lease(ctx,&same,&same_lease)==RNS8_SUCCESS);
      REQUIRE(rns8_import_native_device(ctx,matrix,same_lease)==RNS8_INVALID_ARGUMENT);
      REQUIRE(matrix->content_generation==generation);
      REQUIRE(rns8_destroy_native_device_lease(same_lease)==RNS8_SUCCESS);
      same.access=RNS8_NATIVE_DEVICE_WRITE;
      REQUIRE(rns8_create_native_device_lease(ctx,&same,&same_lease)==RNS8_SUCCESS);
      REQUIRE(rns8_export_native_device(ctx,matrix,same_lease,UINT64_MAX)==RNS8_INVALID_ARGUMENT);
      REQUIRE(matrix->content_generation==generation);
      REQUIRE(rns8_destroy_native_device_lease(same_lease)==RNS8_SUCCESS);
      REQUIRE(native_owner.acquired==2);REQUIRE(native_owner.released==2);
    }
    REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    // Acquire rejection never transfers owner responsibility; ordinary host
    // memory rejection does transfer then immediately release the acquired ref.
    Owner rejected;rejected.status=RNS8_BUSY;auto bad=descriptor(input.get(),values.size()*8,rejected);read=nullptr;
    REQUIRE(rns8_create_native_device_lease(ctx,&bad,&read)==RNS8_BUSY);REQUIRE(rejected.released==0);
    rejected.status=RNS8_SUCCESS;bad.allocation_base=values.data();
    REQUIRE(rns8_create_native_device_lease(ctx,&bad,&read)!=RNS8_SUCCESS);REQUIRE(read==nullptr);
    REQUIRE(rejected.acquired==1);REQUIRE(rejected.released==1);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("copy-owned native HIP leases survive external mutation and deallocation") {
  if(!hip_available()) SKIP("no HIP device for native snapshots");
  using namespace rns8::detail;using namespace native_device_fixture;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool signed_values:{false,true}) {
    hip_direct_device_buffer input;std::vector<uint64_t> values(24,99);
    const uint64_t edges[]{0,UINT64_MAX,UINT64_C(1)<<63,127,UINT64_C(0x80ff7f0100feffff)};
    for(int row=0;row<3;++row) for(int col=0;col<5;++col) values[1+row*7+col]=edges[(row+col)%5];
    REQUIRE(input.allocate(0,values.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    Owner owner;auto d=descriptor(input.get(),values.size()*8,owner);
    d.element_type=signed_values?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64;
    d.policy=RNS8_NATIVE_DEVICE_COPY_OWNED;d.snapshot_budget_bytes=119;
    rns8_native_device_lease* lease=nullptr;
    REQUIRE(rns8_create_native_device_lease(ctx,&d,&lease)==RNS8_WORKSPACE_TOO_SMALL);
    REQUIRE(lease==nullptr);REQUIRE(owner.acquired==0);
    d.snapshot_budget_bytes=120;
    REQUIRE(rns8_create_native_device_lease(ctx,&d,&lease)==RNS8_SUCCESS);
    REQUIRE(owner.acquired==1);REQUIRE(owner.released==1);
    REQUIRE(rns8_destroy_context(ctx)==RNS8_BUSY);
    std::vector<uint64_t> changed(values.size(),7);
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),changed.data(),changed.size()*8)==RNS8_SUCCESS);
    REQUIRE(input.reset()==RNS8_SUCCESS); // library now owns an independent allocation
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
    md.rows=3;md.cols=5;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;md.bound_kind=RNS8_BOUND_NONE;
    md.semantics=signed_values?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;
    rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    for(bool verified:{false,true}) {
      REQUIRE((verified?rns8_import_native_device_verified(ctx,matrix,lease,8):rns8_import_native_device(ctx,matrix,lease))==RNS8_SUCCESS);
      std::vector<int8_t> residues(300);
      REQUIRE(hip_direct_copy_device_to_host(0,residues.data(),matrix->hip_residues,residues.size())==RNS8_SUCCESS);
      for(int row=0;row<3;++row) for(int col=0;col<5;++col) {
        const auto raw=values[1+row*7+col];boost::multiprecision::cpp_int value=raw;
        if(signed_values && (raw>>63)) value-=boost::multiprecision::cpp_int(1)<<64;
        for(unsigned p=0;p<20;++p) REQUIRE(residues[p*15+row*5+col]==centered_residue(value,kDefaultModuli[p]));
      }
    }
    REQUIRE(rns8_destroy_native_device_lease(lease)==RNS8_SUCCESS);REQUIRE(owner.released==1);
    REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}
