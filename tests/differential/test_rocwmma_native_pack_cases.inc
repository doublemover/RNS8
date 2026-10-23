template <typename T>
void check_rocwmma_native_pack_ladder() {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  constexpr int k=17, n=19, ld=23, padded=32, planes=28;
  constexpr int moduli[] = {256,255,253,251,247,239,233,229,227,223,217,211,199,197,
                            193,191,181,179,173,167,163,157,151,149,139,137,131,127};
  const T values[] = {T{0}, T{1}, std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(), static_cast<T>(std::numeric_limits<T>::max()-1),
      static_cast<T>(-1), static_cast<T>(-255), static_cast<T>(256)};
  std::vector<T> input(k*ld, std::numeric_limits<T>::max());
  for (int row=0; row<k; ++row)
    for (int col=0; col<n; ++col) input[row*ld+col]=values[(row*n+col)%8];
  constexpr size_t bytes=planes*padded*padded;
  std::vector<int8_t> actual(bytes+32,99), expected(bytes+32,99);
  for (int p=0; p<planes; ++p)
    for (int col=0; col<padded; ++col)
      for (int row=0; row<padded; ++row) {
        int residue=0;
        if (row<k && col<n) {
          residue=(cpp_int(input[row*ld+col])%moduli[p]).convert_to<int>();
          if (residue<0) residue+=moduli[p];
          if (residue>=(moduli[p]+1)/2) residue-=moduli[p];
        }
        expected[(p*padded+col)*padded+row]=static_cast<int8_t>(residue);
      }
  hip_direct_device_buffer source, destination;
  REQUIRE(source.allocate(0,input.size()*sizeof(T))==RNS8_SUCCESS);
  REQUIRE(destination.allocate(0,actual.size())==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,source.get(),input.data(),input.size()*sizeof(T))==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_host_to_device(0,destination.get(),actual.data(),actual.size())==RNS8_SUCCESS);
  const auto pack=std::is_signed_v<T> ? rocwmma_prepack_b_native_i64_device : rocwmma_prepack_b_native_u64_device;
  REQUIRE(pack(0,source.get(),destination.get(),bytes,k,n,ld,planes)==RNS8_SUCCESS);
  REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),destination.get(),actual.size())==RNS8_SUCCESS);
  REQUIRE(actual==expected);
  REQUIRE(pack(0,source.get(),destination.get(),bytes,k,n,ld,planes+1)==RNS8_INVALID_ARGUMENT);
  REQUIRE(hip_direct_copy_device_to_host(0,actual.data(),destination.get(),actual.size())==RNS8_SUCCESS);
  REQUIRE(actual==expected);
}

TEST_CASE("rocWMMA native B packing covers full-width values and all 28 ladder planes", "[hip][rocwmma][pack]") {
  check_rocwmma_native_pack_ladder<int64_t>();
  check_rocwmma_native_pack_ladder<uint64_t>();
}
