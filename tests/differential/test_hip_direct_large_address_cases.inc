TEST_CASE("budgeted finite controls read real device addresses beyond signed 32-bit range", "[large-address]") {
  const char* requested=std::getenv("RNS8_LARGE_ADDRESS_BUDGET_BYTES");
  if(!requested || !*requested)SKIP("large allocation gate unrun: set RNS8_LARGE_ADDRESS_BUDGET_BYTES explicitly (3 GiB suffices)");
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  using Clock=std::chrono::steady_clock;
  using json=nlohmann::json;
  uint64_t budget=0;const auto end=requested+std::char_traits<char>::length(requested);
  const auto parsed=std::from_chars(requested,end,budget);
  REQUIRE(parsed.ec==std::errc{});REQUIRE(parsed.ptr==end);
  constexpr int64_t m=32769,n=1,k=65536,ldc=3,window=128;
  constexpr uint64_t a_bytes=uint64_t(m)*k,b_bytes=k,c_bytes=uint64_t(m)*ldc;
  constexpr uint64_t required=a_bytes+b_bytes+c_bytes,reserve=512ull<<20;
  if(budget<required)SKIP("explicit device budget cannot fit the real large-address fixture");
  if(!hip_available())SKIP("no applicable HIP device for physical large-address qualification");
  REQUIRE(hipSetDevice(0)==hipSuccess);size_t free_bytes=0,total_bytes=0;
  REQUIRE(hipMemGetInfo(&free_bytes,&total_bytes)==hipSuccess);
  INFO("required="<<required<<" budget="<<budget<<" free="<<free_bytes<<" total="<<total_bytes);
  if(uint64_t(free_bytes)<required+reserve)SKIP("available GPU memory cannot fit fixture plus 512 MiB reserve");
  rns8_device_info device{};device.struct_size=sizeof(device);device.abi_version=RNS8_ABI_VERSION;
  REQUIRE(hip_direct_probe(0,device)==RNS8_SUCCESS);
  struct TimingMode {
    bool previous=hip_direct_timing_enabled();
    TimingMode(){hip_direct_timing_set_enabled(false);}
    ~TimingMode(){hip_direct_timing_set_enabled(previous);}
  } timing_mode;
  auto* ledger=hip_direct_allocation_ledger(0);REQUIRE(ledger!=nullptr);
  const auto baseline=ledger->snapshot();REQUIRE(baseline.complete);uint64_t epoch=0;
  REQUIRE(ledger->reset_peaks(epoch));
  hip_direct_device_buffer da,db,dc;
  REQUIRE(da.allocate(0,size_t(a_bytes))==RNS8_SUCCESS);
  REQUIRE(db.allocate(0,size_t(b_bytes))==RNS8_SUCCESS);
  REQUIRE(dc.allocate(0,size_t(c_bytes))==RNS8_SUCCESS);
  const auto allocated=ledger->snapshot();REQUIRE(allocated.complete);
  REQUIRE(allocated.domains[0].live_bytes-baseline.domains[0].live_bytes==required);

  // A uploads use a fixed 16 MiB host chunk plus small period tables.
  // Generate actual stored bytes;
  // the kernels still load every K term from the complete >2 GiB allocation.
  constexpr int64_t chunk_rows=256,period=253;
  std::vector<int8_t> chunk(size_t(chunk_rows*k)),b(size_t(k),0),actual(size_t(c_bytes),91),expected(actual);
  std::array<std::array<int8_t,period>,period> patterns{};
  for(int64_t row=0;row<period;++row)for(int64_t q=0;q<period;++q)
    patterns[size_t(row)][size_t(q)]=int8_t((17*row+13*q)%period-126);
  for(int64_t q=0;q<k;++q)b[size_t(q)]=int8_t((7*q+5)%period-126);
  const auto fill_start=Clock::now();
  for(int64_t first=0;first<m;first+=chunk_rows) {
    const auto rows=std::min(chunk_rows,m-first);
    for(int64_t row=0;row<rows;++row) {
      const auto& pattern=patterns[size_t((first+row)%period)];
      for(int64_t q=0;q<k;q+=period)
        std::copy_n(pattern.data(),size_t(std::min(period,k-q)),chunk.data()+size_t(row*k+q));
    }
    REQUIRE(hip_direct_copy_host_to_device(0,static_cast<int8_t*>(da.get())+size_t(first*k),chunk.data(),size_t(rows*k))==RNS8_SUCCESS);
  }
  REQUIRE(hip_direct_copy_host_to_device(0,db.get(),b.data(),b.size())==RNS8_SUCCESS);
  const double fill_ns=std::chrono::duration<double,std::nano>(Clock::now()-fill_start).count();
  // Independent exact period-sum oracle: no generated input table is reused.
  std::array<int8_t,period> oracle{};
  for(int64_t row=0;row<period;++row) {
    cpp_int whole=0,tail=0;
    for(int64_t q=0;q<period;++q) {
      const cpp_int product=cpp_int((17*row+13*q)%period-126)*((7*q+5)%period-126);
      whole+=product;if(q<k%period)tail+=product;
    }
    const cpp_int sum=whole*(k/period)+tail;
    int value=(sum%period).convert_to<int>();if(value<0)value+=period;if(value>=127)value-=period;
    oracle[size_t(row)]=int8_t(value);
  }
  REQUIRE(oracle[0]!=oracle[size_t((m-1)%period)]);
  const auto reset_output=[&] {
    std::fill(actual.begin(),actual.end(),91);
    REQUIRE(hip_direct_copy_host_to_device(0,dc.get(),actual.data(),actual.size())==RNS8_SUCCESS);
  };
  const auto execute=[&](int64_t logical_m,int64_t first,int64_t rows,FiniteGemmAddressWidth width) {
    da.mark_pending();db.mark_pending();dc.mark_pending();
    const auto status=hip_direct_gemm_finite_indexed_window_device(0,da.get(),db.get(),dc.get(),logical_m,n,k,k,n,ldc,253,first,rows,width);
    if(status==RNS8_SUCCESS){da.mark_completed();db.mark_completed();dc.mark_completed();}
    REQUIRE(status==RNS8_SUCCESS);
  };
  json phases=json::array();
  for(auto width:{FiniteGemmAddressWidth::signed32,FiniteGemmAddressWidth::unsigned64}) {
    const int64_t rows=width==FiniteGemmAddressWidth::signed32?32767:m;
    reset_output();expected.assign(size_t(c_bytes),91);
    for(int64_t row=0;row<rows;++row)expected[size_t(row*ldc)]=oracle[size_t(row%period)];
    const auto start=Clock::now();
    for(int64_t first=0;first<rows;first+=window)execute(rows,first,std::min(window,rows-first),width);
    const double ns=std::chrono::duration<double,std::nano>(Clock::now()-start).count();
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
    REQUIRE(actual==expected);
    phases.push_back({{"address_width",width==FiniteGemmAddressWidth::signed32?"signed32":"unsigned64"},
        {"rows",rows},{"windows",(rows+window-1)/window},{"host_completed_ns",ns}});
  }
  cpp_int checksum=0,oracle_checksum=0;
  for(int64_t row=0;row<m;++row) {
    checksum+=cpp_int(int(actual[size_t(row*ldc)]))*(row+1);
    oracle_checksum+=cpp_int(int(oracle[size_t(row%period)]))*(row+1);
  }
  REQUIRE(checksum==oracle_checksum);
  reset_output();
  for(int64_t rows:{32768,32769}) {
    REQUIRE(hip_direct_gemm_finite_u8_resident_device(0,da.get(),db.get(),dc.get(),rows,n,k,k,n,ldc,253)==RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(hip_direct_gemm_finite_indexed_window_device(0,da.get(),db.get(),dc.get(),rows,n,k,k,n,ldc,253,rows-1,1,FiniteGemmAddressWidth::signed32)==RNS8_UNSUPPORTED_BACKEND);
    REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
    REQUIRE(actual==std::vector<int8_t>(size_t(c_bytes),91));
  }

  // Matched same-body specializations below the narrow boundary. All original
  // bases/pitches and the near-limit row coordinate are identical.
  constexpr int64_t matched_m=32767,matched_first=matched_m-window;
  for(unsigned warmup=0;warmup<2;++warmup)for(auto width:{FiniteGemmAddressWidth::signed32,FiniteGemmAddressWidth::unsigned64})
    execute(matched_m,matched_first,window,width);
  std::array<std::vector<double>,2> samples;
  for(unsigned repeat=0;repeat<9;++repeat)for(unsigned position=0;position<2;++position) {
    const unsigned mode=(repeat+position)&1;
    const auto start=Clock::now();execute(matched_m,matched_first,window,static_cast<FiniteGemmAddressWidth>(mode));
    samples[mode].push_back(std::chrono::duration<double,std::nano>(Clock::now()-start).count());
  }
  REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
  expected.assign(size_t(c_bytes),91);
  for(int64_t row=matched_first;row<matched_m;++row)expected[size_t(row*ldc)]=oracle[size_t(row%period)];
  REQUIRE(actual==expected);
  json diagnostic=json::array();
  hip_direct_timing_set_enabled(true);
  for(unsigned mode=0;mode<2;++mode) {
    hip_direct_timing_reset();execute(matched_m,matched_first,window,static_cast<FiniteGemmAddressWidth>(mode));
    json events=json::array(),failures=json::array();
    for(const auto& event:hip_direct_timing_snapshot())events.push_back({{"label",event.label},{"microseconds",event.microseconds}});
    for(const auto& failure:hip_direct_timing_failure_snapshot())failures.push_back({{"label",failure.label},{"phase",failure.phase},{"backend_status",failure.backend_status}});
    diagnostic.push_back({{"address_width",mode?"unsigned64":"signed32"},{"events",events},{"failures",failures},{"failure_observer_complete",hip_direct_timing_failures_complete()}});
  }
  hip_direct_timing_set_enabled(false);
  // Separately exercise the existing default-basis u64 primitive with a real
  // sparse high address. Prefix one means modulus 256, explicitly not 253.
  constexpr int64_t high_pitch=(int64_t{1}<<31)+1;
  const int8_t seven=7,minus_five=-5,three=3;
  REQUIRE(hip_direct_copy_host_to_device(0,da.get(),&seven,1)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,static_cast<int8_t*>(da.get())+high_pitch,&minus_five,1)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,db.get(),&three,1)==RNS8_SUCCESS);
  reset_output();da.mark_pending();db.mark_pending();dc.mark_pending();
  const auto sparse=hip_direct_gemm_rns_wide_device(0,da.get(),db.get(),dc.get(),2,1,1,high_pitch,1,ldc,1);
  if(sparse==RNS8_SUCCESS){da.mark_completed();db.mark_completed();dc.mark_completed();}
  REQUIRE(sparse==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),dc.get(),actual.size())==RNS8_SUCCESS);
  expected.assign(size_t(c_bytes),91);expected[0]=21;expected[size_t(ldc)]=-15;REQUIRE(actual==expected);
  REQUIRE(rns8_default_modulus(0)==256);
  const auto peak=ledger->snapshot();REQUIRE(peak.complete);
  REQUIRE(peak.domains[0].peak_bytes-baseline.domains[0].live_bytes==required);
  const auto copies=hip_direct_allocation_counters_snapshot();
  REQUIRE(dc.reset()==RNS8_SUCCESS);REQUIRE(db.reset()==RNS8_SUCCESS);REQUIRE(da.reset()==RNS8_SUCCESS);
  const auto retired=ledger->snapshot();REQUIRE(retired.complete);
  REQUIRE(retired.domains[0].live_bytes==baseline.domains[0].live_bytes);
  REQUIRE(retired.domains[0].live_allocations==baseline.domains[0].live_allocations);
  REQUIRE(retired.domains[0].allocation_calls-baseline.domains[0].allocation_calls==3);
  REQUIRE(retired.domains[0].free_calls-baseline.domains[0].free_calls==3);
  json report={{"schema","rns8_large_address_qualification_v1"},{"target",device.gcn_arch},{"device",device.name},
      {"hip_runtime_version",device.hip_runtime_version},{"hip_driver_version",device.hip_driver_version},
      {"semantic_contract","finite_ring_mod253_centered_i8"},{"dataset","affine_17_13_7_5_mod253_centered126_v1"},
      {"m",m},{"n",n},{"k",k},{"ldc",ldc},{"window_rows",window},{"device_budget_bytes",budget},
      {"available_device_bytes_before",free_bytes},{"total_device_bytes",total_bytes},{"reserve_bytes",reserve},
      {"a_allocation_bytes",a_bytes},{"b_allocation_bytes",b_bytes},{"c_allocation_bytes",c_bytes},
      {"observed_requested_device_peak_delta",peak.domains[0].peak_bytes-baseline.domains[0].live_bytes},
      {"host_input_chunk_bytes",chunk.size()},{"input_generation_upload_ns",fill_ns},{"qualification_phases",phases},
      {"all_rows_and_padding_exact",true},{"centered_output_weighted_checksum",checksum.str()},
      {"first_a_offset_above_int32",uint64_t(32768)*k},{"compute_host_pointer_rebasing",false},
      {"narrow_rejected_before_write",{32768,32769}},{"matched_m",matched_m},{"matched_first_row",matched_first},
      {"matched_warmups",2},{"matched_repeats",9},{"matched_i32_host_ns",samples[0]},{"matched_u64_host_ns",samples[1]},
      {"timing_source","host_monotonic_completed_backend_window"},{"matched_timing_hooks_enabled",false},
      {"nominal_clock_tick_ns",1e9*double(Clock::period::num)/double(Clock::period::den)},
      {"timing_includes",{"backend_wrapper","required_sync","owner_pending_tracking","status_assertion"}},
      {"backend","hip-direct"},{"prefix_count",1},{"promotion_eligible",false},
      {"separate_event_diagnostics",diagnostic},{"sparse_existing_u64_control_modulus",256},
      {"sparse_existing_u64_high_offset",high_pitch},{"h2d_payload_bytes_process_counter",copies.h2d_bytes},
      {"device_allocation_calls",retired.domains[0].allocation_calls-baseline.domains[0].allocation_calls},
      {"device_free_calls",retired.domains[0].free_calls-baseline.domains[0].free_calls},
      {"all_owned_device_bytes_retired",true},{"synchronization","hipDeviceSynchronize_in_each_backend_wrapper"}};
#if defined(RNS8_COMPILED_COMMIT)
  report["compiled_source_commit"]=RNS8_COMPILED_COMMIT;
#endif
#if defined(RNS8_BUILD_MANIFEST_JSON)
  report["build_manifest"]=json::parse(RNS8_BUILD_MANIFEST_JSON);
#endif
  std::cout<<"RNS8_LARGE_ADDRESS_RECEIPT "<<report.dump()<<'\n';
#else
  SKIP("large-address GPU gate requires a compiled HIP backend");
#endif
}
