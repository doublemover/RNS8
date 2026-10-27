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

TEST_CASE("native HIP limb exports preserve exact wide products and padded destinations") {
  if(!hip_available()) SKIP("no HIP device for limb leases");
  using namespace rns8::detail; using namespace native_device_fixture;
  using boost::multiprecision::cpp_int;
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool sign:{false,true}) {
    auto gd=sign?exact_signed_desc(2,2,1,RNS8_BACKEND_HIP_DIRECT):
                 exact_unsigned_desc(2,2,1,RNS8_BACKEND_HIP_DIRECT);
    gd.max_prefix=20; gd.flags|=RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan=nullptr; rns8_workspace* workspace=nullptr;
    REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);
    REQUIRE(rns8_create_workspace(ctx,plan,&workspace)==RNS8_SUCCESS);
    rns8_matrix *a=nullptr,*b=nullptr,*c=nullptr;
    auto md=matrix_desc(2,1,gd.semantics,RNS8_BOUND_NONE); md.max_prefix=20;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);
    md.rows=1;md.cols=2;md.logical_ld=2; REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
    md.rows=2; REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    const int64_t sa[]{INT64_MIN,INT64_MAX}, sb[]{INT64_MIN,-1};
    const uint64_t ua[]{UINT64_MAX,UINT64_C(1)<<63}, ub[]{UINT64_MAX,1};
    REQUIRE((sign?rns8_pack_i64(ctx,a,sa,1,0):rns8_pack_u64(ctx,a,ua,1,0))==RNS8_SUCCESS);
    REQUIRE((sign?rns8_pack_i64(ctx,b,sb,2,0):rns8_pack_u64(ctx,b,ub,2,0))==RNS8_SUCCESS);
    REQUIRE(rns8_gemm_rns(ctx,plan,a,b,c,workspace)==RNS8_SUCCESS);
    for(uint32_t width:{1u,2u,3u,32u}) {
      std::vector<uint64_t> actual(1+6*width,77),expected=actual;
      hip_direct_device_buffer output;
      REQUIRE(output.allocate(0,actual.size()*8)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,output.get(),actual.data(),actual.size()*8)==RNS8_SUCCESS);
      Owner owner; auto d=descriptor(output.get(),actual.size()*8,owner);
      d.rows=2;d.cols=2;d.ld=3;d.access=RNS8_NATIVE_DEVICE_WRITE;d.limb_count=width;
      d.element_type=sign?RNS8_NATIVE_DEVICE_SIGNED_LIMBS:RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS;
      rns8_native_device_resources r{};r.struct_size=sizeof(r);r.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_query_native_device(&d,&r)==RNS8_SUCCESS);
      rns8_native_device_lease* lease=nullptr;
      REQUIRE(rns8_create_native_device_lease(ctx,&d,&lease)==RNS8_SUCCESS);
      CHECK(rns8_import_native_device(ctx,c,lease)==RNS8_INVALID_ARGUMENT);
      if(width==1) CHECK(rns8_export_native_device(ctx,c,lease,r.export_scratch_bytes)==RNS8_RANGE_ERROR);
      else {
        CHECK(rns8_export_native_device(ctx,c,lease,r.export_scratch_bytes-1)==RNS8_WORKSPACE_TOO_SMALL);
        const auto before=hip_direct_allocation_counters_snapshot();
        REQUIRE(rns8_export_native_device(ctx,c,lease,r.export_scratch_bytes)==RNS8_SUCCESS);
        const auto after=hip_direct_allocation_counters_snapshot();
        CHECK(after.h2d_bytes==before.h2d_bytes);
        CHECK(after.d2h_bytes-before.d2h_bytes==4);
        CHECK(after.d2d_bytes-before.d2d_bytes==4*width*8);
        for(int row=0;row<2;++row) for(int col=0;col<2;++col) {
          cpp_int value=(sign?cpp_int(sa[row]):cpp_int(ua[row]))*
                        (sign?cpp_int(sb[col]):cpp_int(ub[col]));
          if(value<0) value+=cpp_int(1)<<(64*width);
          for(uint32_t word=0;word<width;++word)
            expected[1+(row*3+col)*width+word]=((value>>(64*word))&UINT64_MAX).convert_to<uint64_t>();
        }
      }
      REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),output.get(),actual.size()*8)==RNS8_SUCCESS);
      CHECK(actual==expected);
      REQUIRE(rns8_destroy_native_device_lease(lease)==RNS8_SUCCESS);
      CHECK(owner.acquired==1);CHECK(owner.released==1);
    }
    for(auto* matrix:{a,b,c}) REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(workspace)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
}

TEST_CASE("native transfer prepared owners retain leases, refresh generations and replay checked graphs") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;using namespace native_device_fixture;
  if(!hip_available())SKIP("no HIP device");auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool u:{false,true}) {
    hip_direct_device_buffer input,output;std::vector<uint64_t> values(24,99),observed(24,77);
    for(int r=0;r<3;++r)for(int c=0;c<5;++c)values[1+r*7+c]=u?uint64_t(r*5+c):uint64_t(int64_t(r*5+c)-7);
    REQUIRE(input.allocate(0,192)==RNS8_SUCCESS);REQUIRE(output.allocate(0,192)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),192)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,output.get(),observed.data(),192)==RNS8_SUCCESS);
    Owner source_owner,destination_owner;auto source=descriptor(input.get(),192,source_owner),destination=descriptor(output.get(),192,destination_owner);
    source.element_type=destination.element_type=u?RNS8_NATIVE_DEVICE_U64:RNS8_NATIVE_DEVICE_I64;destination.access=RNS8_NATIVE_DEVICE_WRITE;destination.owner_identity=202;
    rns8_native_device_lease *read=nullptr,*write=nullptr;REQUIRE(rns8_create_native_device_lease(ctx,&source,&read)==RNS8_SUCCESS);REQUIRE(rns8_create_native_device_lease(ctx,&destination,&write)==RNS8_SUCCESS);
    auto md=matrix_desc(3,5,u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED,RNS8_BOUND_NONE);md.max_prefix=3;md.flags=RNS8_MATRIX_RETAIN_NATIVE_INPUT;
    rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    rns8_native_transfer_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.import_values=1;options.verify_input_bound=1;options.max_input_magnitude=14;options.max_total_scratch_bytes=184;
    rns8_prepared_execution *importer=nullptr,*exporter=nullptr;
    REQUIRE(rns8_prepare_native_transfer_hip(ctx,matrix,read,&options,&importer)==RNS8_WORKSPACE_TOO_SMALL);REQUIRE(!importer);
    options.max_total_scratch_bytes=185; // 45 RNS + 16 status/scan + 4 pinned + 120 retained native
    REQUIRE(rns8_prepare_native_transfer_hip(ctx,matrix,read,&options,&importer)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(read)==RNS8_BUSY);REQUIRE(source_owner.released==0);
    REQUIRE(rns8_capture_prepared_hip_graph(importer)==RNS8_SUCCESS);
    rns8_completion_token initial{};REQUIRE(rns8_execute_prepared(importer,&initial)==RNS8_SUCCESS);
    REQUIRE(matrix->device_native_current);REQUIRE(matrix->exact_max_magnitude==14);
    options.import_values=0;options.verify_input_bound=0;options.max_input_magnitude=0;options.max_total_scratch_bytes=128;
    REQUIRE(rns8_prepare_native_transfer_hip(ctx,matrix,write,&options,&exporter)==RNS8_SUCCESS);REQUIRE(rns8_capture_prepared_hip_graph(exporter)==RNS8_SUCCESS);
    hip_unique_stream producer_stream,import_stream,export_stream;
    REQUIRE(producer_stream.create_non_blocking()==hipSuccess);REQUIRE(import_stream.create_non_blocking()==hipSuccess);REQUIRE(export_stream.create_non_blocking()==hipSuccess);
    for(bool invalid:{false,true}) {
      REQUIRE(rns8_begin_native_device_write(read)==RNS8_SUCCESS);
      REQUIRE(rns8_destroy_native_device_lease(read)==RNS8_BUSY);
      REQUIRE(rns8_end_native_device_write(read,1)==RNS8_INVALID_ARGUMENT);
      // Device producer writes one scalar on its own stream, without a host
      // payload transfer. Invalid input is an authored value, not fault injection.
      REQUIRE(hipMemsetAsync(static_cast<uint8_t*>(input.get())+8,invalid?0xff:0,8,producer_stream.get())==hipSuccess);
      // Signed -1 fits; force unsigned 255 / signed +255 with a separate byte.
      if(invalid){REQUIRE(hipMemsetAsync(static_cast<uint8_t*>(input.get())+8,0,8,producer_stream.get())==hipSuccess);
        REQUIRE(hipMemsetAsync(static_cast<uint8_t*>(input.get())+8,0xff,1,producer_stream.get())==hipSuccess);}
      REQUIRE(rns8_end_native_device_write_hip(read,invalid?3:2,producer_stream.get())==RNS8_SUCCESS);
      rns8_completion_token untouched{99,99};REQUIRE(rns8_submit_prepared_hip_graph(importer,import_stream.get(),&untouched)==RNS8_INVALID_ARGUMENT);REQUIRE(untouched.epoch==99);
      REQUIRE(rns8_refresh_prepared_inputs(importer)==RNS8_SUCCESS);
      const auto before=hip_direct_allocation_counters_snapshot();rns8_completion_token imported{},exported{};
      REQUIRE(rns8_submit_prepared_hip_graph(importer,import_stream.get(),&imported)==RNS8_SUCCESS);
      REQUIRE(rns8_begin_native_device_write(read)==RNS8_BUSY);REQUIRE(rns8_destroy_prepared_execution(importer)==RNS8_BUSY);
      REQUIRE(rns8_bind_prepared_dependency(exporter,importer,imported)==RNS8_SUCCESS);
      REQUIRE(rns8_submit_prepared_hip_graph_pipeline(exporter,export_stream.get(),&exported)==RNS8_SUCCESS);
      rns8_completion_info completion{};completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(exporter,exported,&completion)==RNS8_SUCCESS);
      REQUIRE(completion.operation_status==(invalid?RNS8_RANGE_ERROR:RNS8_SUCCESS));REQUIRE(bool(completion.output_is_current)==!invalid);
      const auto after=hip_direct_allocation_counters_snapshot();REQUIRE(after.allocate_calls==before.allocate_calls);REQUIRE(after.free_calls==before.free_calls);
      rns8_prepared_transfer_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_get_prepared_transfer_info(importer,&info)==RNS8_SUCCESS);REQUIRE(info.external_generation==(invalid?3u:2u));REQUIRE(info.retained_native_matrix_bytes==120);REQUIRE(info.verified_input_bound);
      REQUIRE(rns8_get_prepared_transfer_info(exporter,&info)==RNS8_SUCCESS);REQUIRE(bool(info.external_result_current)==!invalid);
      if(!invalid){REQUIRE(hip_direct_copy_device_to_host(0,observed.data(),output.get(),192)==RNS8_SUCCESS);
        values[1]=0;for(int r=0;r<3;++r)for(int c=0;c<5;++c)REQUIRE(observed[1+r*7+c]==values[1+r*7+c]);
        REQUIRE(observed[0]==77);REQUIRE(observed[6]==77);REQUIRE(observed[23]==77);}
    }
    REQUIRE(rns8_destroy_prepared_execution(exporter)==RNS8_SUCCESS);REQUIRE(rns8_destroy_prepared_execution(importer)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(write)==RNS8_SUCCESS);REQUIRE(rns8_destroy_native_device_lease(read)==RNS8_SUCCESS);
    REQUIRE(source_owner.released==1);REQUIRE(destination_owner.released==1);REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP not compiled");
#endif
}


TEST_CASE("prepared native export shares every signed/unsigned prefix and limb-width kernel") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;using namespace native_device_fixture;
  if(!hip_available())SKIP("no HIP device");auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool u:{false,true})for(uint32_t prefix:{1u,3u,9u,18u,20u}) {
    auto md=matrix_desc(1,1,u?RNS8_EXACT_WIDE_UNSIGNED:RNS8_EXACT_WIDE_SIGNED,RNS8_BOUND_NONE);md.max_prefix=prefix;
    rns8_matrix* matrix=nullptr;REQUIRE(rns8_create_matrix(ctx,&md,&matrix)==RNS8_SUCCESS);
    const uint64_t one=u?1:UINT64_MAX;REQUIRE((u?rns8_pack_u64(ctx,matrix,&one,1,0):rns8_pack_i64(ctx,matrix,reinterpret_cast<const int64_t*>(&one),1,0))==RNS8_SUCCESS);
    for(uint32_t limbs:{1u,2u,3u,4u,8u,16u,32u}) {
      const uint64_t width=8*limbs;std::vector<uint64_t> values(3*limbs,99);hip_direct_device_buffer output;REQUIRE(output.allocate(0,width*3)==RNS8_SUCCESS);
      REQUIRE(hip_direct_copy_host_to_device(0,output.get(),values.data(),width*3)==RNS8_SUCCESS);
      Owner owner;auto desc=descriptor(output.get(),width*3,owner);desc.rows=desc.cols=1;desc.ld=3;desc.byte_offset=width;desc.access=RNS8_NATIVE_DEVICE_WRITE;
      desc.element_type=u?RNS8_NATIVE_DEVICE_UNSIGNED_LIMBS:RNS8_NATIVE_DEVICE_SIGNED_LIMBS;desc.limb_count=limbs;
      rns8_native_device_lease* lease=nullptr;REQUIRE(rns8_create_native_device_lease(ctx,&desc,&lease)==RNS8_SUCCESS);
      rns8_native_transfer_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.max_total_scratch_bytes=width+8;
      rns8_prepared_execution* prepared=nullptr;REQUIRE(rns8_prepare_native_transfer_hip(ctx,matrix,lease,&options,&prepared)==RNS8_SUCCESS);
      REQUIRE(rns8_capture_prepared_hip_graph(prepared)==RNS8_SUCCESS);hip_unique_stream stream;REQUIRE(stream.create_non_blocking()==hipSuccess);
      rns8_completion_token token{};REQUIRE(rns8_submit_prepared_hip_graph(prepared,stream.get(),&token)==RNS8_SUCCESS);
      rns8_completion_info result{};result.struct_size=sizeof(result);result.abi_version=RNS8_ABI_VERSION;
      REQUIRE(rns8_wait_prepared_hip(prepared,token,&result)==RNS8_SUCCESS);REQUIRE(result.output_is_current);
      REQUIRE(hip_direct_copy_device_to_host(0,values.data(),output.get(),width*3)==RNS8_SUCCESS);
      for(uint32_t i=0;i<3*limbs;++i)REQUIRE(values[i]==(i<limbs || i>=2*limbs?99:u?(i==limbs?1:0):UINT64_MAX));
      REQUIRE(rns8_destroy_prepared_execution(prepared)==RNS8_SUCCESS);REQUIRE(rns8_destroy_native_device_lease(lease)==RNS8_SUCCESS);REQUIRE(owner.released==1);
    }
    REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP not compiled");
#endif
}

TEST_CASE("distinct immutable native leases share reads but never an external writer or mutation epoch") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;using namespace native_device_fixture;
  if(!hip_available())SKIP("no HIP device");auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  hip_direct_device_buffer input;REQUIRE(input.allocate(0,8)==RNS8_SUCCESS);int64_t value=2;REQUIRE(hip_direct_copy_host_to_device(0,input.get(),&value,8)==RNS8_SUCCESS);
  Owner owner;auto desc=descriptor(input.get(),8,owner);desc.rows=desc.cols=desc.ld=1;desc.byte_offset=0;
  rns8_native_device_lease *leases[2]{};REQUIRE(rns8_create_native_device_lease(ctx,&desc,&leases[0])==RNS8_SUCCESS);REQUIRE(rns8_create_native_device_lease(ctx,&desc,&leases[1])==RNS8_SUCCESS);
  auto writer=desc;writer.access=RNS8_NATIVE_DEVICE_WRITE;rns8_native_device_lease* rejected=nullptr;
  REQUIRE(rns8_create_native_device_lease(ctx,&writer,&rejected)==RNS8_INVALID_ARGUMENT);REQUIRE(!rejected);
  REQUIRE(rns8_begin_native_device_write(leases[0])==RNS8_BUSY);
  rns8_matrix* matrices[2]{};rns8_prepared_execution* operations[2]{};
  auto md=matrix_desc(1,1,RNS8_EXACT_WIDE_SIGNED,RNS8_BOUND_NONE);md.max_prefix=3;
  rns8_native_transfer_options options{};options.struct_size=sizeof(options);options.abi_version=RNS8_ABI_VERSION;options.import_values=1;options.verify_input_bound=1;options.max_input_magnitude=2;options.max_total_scratch_bytes=23;
  for(unsigned i=0;i<2;++i){REQUIRE(rns8_create_matrix(ctx,&md,&matrices[i])==RNS8_SUCCESS);REQUIRE(rns8_prepare_native_transfer_hip(ctx,matrices[i],leases[i],&options,&operations[i])==RNS8_SUCCESS);}
  rns8_prepared_execution* same=nullptr;REQUIRE(rns8_prepare_native_transfer_hip(ctx,matrices[1],leases[0],&options,&same)==RNS8_SUCCESS);
  rns8_prepared_execution* duplicate[]{operations[0],same};rns8_prepared_group* group=nullptr;
  REQUIRE(rns8_create_prepared_group(duplicate,2,&group)==RNS8_INVALID_ARGUMENT);REQUIRE(!group);REQUIRE(rns8_destroy_prepared_execution(same)==RNS8_SUCCESS);
  REQUIRE(rns8_create_prepared_group(operations,2,&group)==RNS8_SUCCESS);hip_unique_stream streams[2];for(auto& stream:streams)REQUIRE(stream.create_non_blocking()==hipSuccess);
  void* raw[]{streams[0].get(),streams[1].get()};rns8_completion_token tokens[2]{};uint32_t issued=0;
  REQUIRE(rns8_submit_prepared_group_hip(group,raw,2,tokens,2,&issued)==RNS8_SUCCESS);REQUIRE(issued==2);
  for(int i:{1,0}){rns8_completion_info completion{};completion.struct_size=sizeof(completion);completion.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_wait_prepared_hip(operations[i],tokens[i],&completion)==RNS8_SUCCESS);REQUIRE(completion.output_is_current);REQUIRE(owner.released==0);}
  REQUIRE(rns8_destroy_prepared_group(group)==RNS8_SUCCESS);
  for(auto* operation:operations)REQUIRE(rns8_destroy_prepared_execution(operation)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_native_device_lease(leases[1])==RNS8_SUCCESS);REQUIRE(owner.released==1);
  REQUIRE(rns8_begin_native_device_write(leases[0])==RNS8_SUCCESS);REQUIRE(rns8_end_native_device_write(leases[0],2)==RNS8_SUCCESS);
  REQUIRE(rns8_destroy_native_device_lease(leases[0])==RNS8_SUCCESS);REQUIRE(owner.acquired==2);REQUIRE(owner.released==2);
  for(auto* matrix:matrices)REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP not compiled");
#endif
}

TEST_CASE("native import lift GEMM and host export share one observed graph transaction") {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;using namespace native_device_fixture;
  using boost::multiprecision::cpp_int;
  if(!hip_available())SKIP("no HIP device for typed transfer DAG");
  auto* ctx=create_context(RNS8_BACKEND_HIP_DIRECT);
  for(bool sign:{false,true})for(bool retain:{false,true}) {
    hip_direct_device_buffer input;std::vector<uint64_t> values(24,99);
    for(int row=0;row<3;++row)for(int col=0;col<5;++col)
      values[1+row*7+col]=sign?uint64_t(int64_t(row*5+col)-7):uint64_t(row*5+col);
    REQUIRE(input.allocate(0,values.size()*8)==RNS8_SUCCESS);
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    Owner owner;auto source=descriptor(input.get(),values.size()*8,owner);
    source.element_type=sign?RNS8_NATIVE_DEVICE_I64:RNS8_NATIVE_DEVICE_U64;
    rns8_native_device_lease* lease=nullptr;REQUIRE(rns8_create_native_device_lease(ctx,&source,&lease)==RNS8_SUCCESS);
    rns8_matrix_desc md{};md.struct_size=sizeof(md);md.abi_version=RNS8_ABI_VERSION;
    md.rows=3;md.cols=5;md.logical_layout=RNS8_LAYOUT_ROW_MAJOR;md.max_prefix=20;
    md.semantics=sign?RNS8_EXACT_WIDE_SIGNED:RNS8_EXACT_WIDE_UNSIGNED;md.bound_kind=RNS8_BOUND_NONE;
    md.flags=retain?RNS8_MATRIX_RETAIN_NATIVE_INPUT:0;
    rns8_matrix *a=nullptr,*copy=nullptr,*b=nullptr,*c=nullptr;
    REQUIRE(rns8_create_matrix(ctx,&md,&a)==RNS8_SUCCESS);REQUIRE(rns8_create_matrix(ctx,&md,&copy)==RNS8_SUCCESS);
    md.flags=0;md.rows=5;md.cols=2;REQUIRE(rns8_create_matrix(ctx,&md,&b)==RNS8_SUCCESS);
    md.rows=3;REQUIRE(rns8_create_matrix(ctx,&md,&c)==RNS8_SUCCESS);
    std::vector<uint64_t> bv(10);for(unsigned i=0;i<10;++i)bv[i]=sign?uint64_t(int64_t(i%4)-1):i%3;
    REQUIRE((sign?rns8_pack_i64(ctx,b,reinterpret_cast<const int64_t*>(bv.data()),2,1):rns8_pack_u64(ctx,b,bv.data(),2,1))==RNS8_SUCCESS);
    rns8_gemm_desc gd{};gd.struct_size=sizeof(gd);gd.abi_version=RNS8_ABI_VERSION;
    gd.semantics=md.semantics;gd.bound_kind=RNS8_BOUND_NONE;gd.requested_backend=RNS8_BACKEND_HIP_DIRECT;
    gd.m=3;gd.n=2;gd.k=5;gd.max_prefix=20;gd.flags=RNS8_PLAN_FORCE_FIXED_PREFIX;
    rns8_plan* plan=nullptr;rns8_workspace* work=nullptr;
    REQUIRE(rns8_create_plan(ctx,&gd,&plan)==RNS8_SUCCESS);REQUIRE(rns8_create_workspace(ctx,plan,&work)==RNS8_SUCCESS);
    std::vector<uint64_t> output(27,99);
    rns8_resident_node nodes[6]{};
    for(auto& node:nodes){node.struct_size=sizeof(node);node.abi_version=RNS8_ABI_VERSION;}
    for(unsigned i=0;i<2;++i) {
      auto& node=nodes[i];node.kind=RNS8_RESIDENT_NATIVE_TO_RNS;node.c=i?copy:a;node.native_source=lease;
      node.target_prefix=3;node.verify_native_bound=1;node.native_max_magnitude=14;
      node.max_transfer_storage_bytes=65+(retain?120:0);
      nodes[i+2].kind=RNS8_RESIDENT_BASE_EXTEND;nodes[i+2].c=node.c;nodes[i+2].target_prefix=20;
      nodes[i+2].max_staged_residue_bytes=255;nodes[i+2].max_transfer_storage_bytes=263;
    }
    nodes[4].kind=RNS8_RESIDENT_MATMUL;nodes[4].plan=plan;nodes[4].a=a;nodes[4].b=b;nodes[4].c=c;nodes[4].workspace=work;
    nodes[5].kind=RNS8_RESIDENT_EXPORT_LIMBS;nodes[5].plan=plan;nodes[5].c=c;
    nodes[5].export_limbs=output.data();nodes[5].export_ld=3;nodes[5].export_limb_count=3;nodes[5].max_transfer_storage_bytes=300;
    rns8_resident_graph* graph=nullptr;
    REQUIRE(rns8_create_resident_graph(ctx,nodes,6,&graph)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(lease)==RNS8_BUSY);CHECK(owner.released==0);
    REQUIRE(rns8_capture_resident_graph_hip(graph)==RNS8_SUCCESS);
    rns8_resident_graph_resources resources{};resources.struct_size=sizeof(resources);resources.abi_version=RNS8_ABI_VERSION;
    REQUIRE(rns8_get_resident_graph_resources(graph,&resources)==RNS8_SUCCESS);
    CHECK(resources.transfer_nodes==5);CHECK(resources.retained_device_scratch_bytes==259);
    CHECK(resources.retained_pinned_host_bytes==168);CHECK(resources.retained_native_matrix_bytes==(retain?240:0));
    hip_unique_stream stream,producer;REQUIRE(stream.create_non_blocking()==hipSuccess);REQUIRE(producer.create_non_blocking()==hipSuccess);
    rns8_graph_execution_info info{};info.struct_size=sizeof(info);info.abi_version=RNS8_ABI_VERSION;
    const auto audit=[&]() {
      for(int row=0;row<3;++row)for(int col=0;col<2;++col) {
        cpp_int expected=0;
        for(int q=0;q<5;++q) {
          const auto av=values[1+row*7+q],v=bv[q*2+col];
          expected+=(sign?cpp_int(static_cast<int64_t>(av)):cpp_int(av))*(sign?cpp_int(static_cast<int64_t>(v)):cpp_int(v));
        }
        const auto pos=(row*3+col)*3;cpp_int actual=cpp_int(output[pos])+(cpp_int(output[pos+1])<<64)+(cpp_int(output[pos+2])<<128);
        if(sign && output[pos+2]>>63)actual-=cpp_int(1)<<192;
        REQUIRE(actual==expected);
      }
      for(int row=0;row<3;++row)for(int limb=0;limb<3;++limb)CHECK(output[(row*3+2)*3+limb]==99);
      CHECK(a->exact_range_prefix==20);CHECK(copy->exact_range_prefix==20);
      CHECK(a->device_native_current==retain);CHECK(copy->device_native_current==retain);
    };
    for(unsigned pass=0;pass<2;++pass) {
      if(pass) {
        REQUIRE(rns8_begin_native_device_write(lease)==RNS8_SUCCESS);values[1]=sign?uint64_t(-4):4;
        REQUIRE(hipMemcpyAsync(input.get(),values.data(),values.size()*8,hipMemcpyHostToDevice,producer.get())==hipSuccess);
        REQUIRE(rns8_end_native_device_write_hip(lease,2,producer.get())==RNS8_SUCCESS);
        rns8_graph_completion_token rejected{77,88};
        REQUIRE(rns8_replay_resident_graph_hip(graph,stream.get(),&rejected)==RNS8_INVALID_ARGUMENT);
        REQUIRE(rejected.graph_id==77);REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_SUCCESS);
      }
      const auto before=output;const auto generation=a->content_generation;
      rns8_graph_completion_token token{};
      REQUIRE((pass?rns8_replay_resident_graph_hip(graph,stream.get(),&token):rns8_submit_resident_graph_hip(graph,stream.get(),&token))==RNS8_SUCCESS);
      CHECK(output==before);CHECK(a->content_generation==generation);
      CHECK(rns8_begin_native_device_write(lease)==RNS8_BUSY);
      REQUIRE(rns8_wait_resident_graph_hip(graph,token,&info)==RNS8_SUCCESS);REQUIRE(info.operation_status==RNS8_SUCCESS);
      CHECK(a->content_generation==generation+1);CHECK(info.completed_nodes==6);audit();
    }
    REQUIRE(rns8_begin_native_device_write(lease)==RNS8_SUCCESS);values[1]=100;
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    REQUIRE(rns8_end_native_device_write(lease,3)==RNS8_SUCCESS);REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_SUCCESS);
    const auto before=output;rns8_graph_completion_token failed{};
    REQUIRE(rns8_replay_resident_graph_hip(graph,stream.get(),&failed)==RNS8_SUCCESS);
    REQUIRE(rns8_wait_resident_graph_hip(graph,failed,&info)==RNS8_SUCCESS);
    CHECK(info.operation_status==RNS8_RANGE_ERROR);CHECK(info.failed_node==0);CHECK(info.completed_nodes==0);
    CHECK(output==before);CHECK_FALSE(a->device_residues_current);CHECK_FALSE(c->device_residues_current);CHECK(b->device_residues_current);
    REQUIRE(rns8_begin_native_device_write(lease)==RNS8_SUCCESS);values[1]=2;
    REQUIRE(hip_direct_copy_host_to_device(0,input.get(),values.data(),values.size()*8)==RNS8_SUCCESS);
    REQUIRE(rns8_end_native_device_write(lease,4)==RNS8_SUCCESS);REQUIRE(rns8_refresh_resident_graph_inputs(graph)==RNS8_SUCCESS);
    REQUIRE(rns8_execute_resident_graph(graph,&info)==RNS8_SUCCESS);REQUIRE(info.operation_status==RNS8_SUCCESS);audit();
    REQUIRE(rns8_destroy_resident_graph(graph)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_native_device_lease(lease)==RNS8_SUCCESS);CHECK(owner.released==1);
    for(auto* matrix:{a,copy,b,c})REQUIRE(rns8_destroy_matrix(matrix)==RNS8_SUCCESS);
    REQUIRE(rns8_destroy_workspace(work)==RNS8_SUCCESS);REQUIRE(rns8_destroy_plan(plan)==RNS8_SUCCESS);
  }
  REQUIRE(rns8_destroy_context(ctx)==RNS8_SUCCESS);
#else
  SKIP("HIP not compiled");
#endif
}
