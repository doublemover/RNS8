namespace native_moment_reference {
template <typename T>
rns8_exact_matrix_ranges construct(const T* input,int64_t rows,int64_t cols,int64_t ld,
                                   boost::multiprecision::cpp_int& maximum) {
  using boost::multiprecision::cpp_int;
  rns8_exact_matrix_ranges result;result.rows.resize(size_t(rows));result.cols.resize(size_t(cols));maximum=0;
  const auto axis=[&](int64_t count,int64_t offset,int64_t stride) {
    rns8_exact_axis_range out;
    for(int64_t i=0;i<count;++i) {
      const cpp_int value=input[offset+i*stride];
      const cpp_int magnitude=boost::multiprecision::abs(value);
      out.max_magnitude=std::max(out.max_magnitude,magnitude);out.absolute_sum+=magnitude;
      out.sum+=value;out.squared_sum+=value*value;
      if(i==0)out.minimum=out.maximum=value;
      else{out.minimum=std::min(out.minimum,value);out.maximum=std::max(out.maximum,value);}
    }
    if(count) {
      const cpp_int center=(out.minimum+out.maximum)/2;
      for(int64_t i=0;i<count;++i)out.deviation_sum+=boost::multiprecision::abs(cpp_int(input[offset+i*stride])-center);
      out.sum_known=out.squared_sum_known=true;
    }
    return out;
  };
  for(int64_t r=0;r<rows;++r){result.rows[size_t(r)]=axis(cols,r*ld,1);maximum=std::max(maximum,result.rows[size_t(r)].max_magnitude);}
  for(int64_t c=0;c<cols;++c)result.cols[size_t(c)]=axis(rows,c,ld);
  return result;
}
inline void equal(const rns8_exact_axis_range& actual,const rns8_exact_axis_range& expected) {
  CHECK(actual.max_magnitude==expected.max_magnitude);CHECK(actual.absolute_sum==expected.absolute_sum);
  CHECK(actual.sum==expected.sum);CHECK(actual.minimum==expected.minimum);CHECK(actual.maximum==expected.maximum);
  CHECK(actual.deviation_sum==expected.deviation_sum);CHECK(actual.squared_sum==expected.squared_sum);
  CHECK(actual.sum_known==expected.sum_known);CHECK(actual.squared_sum_known==expected.squared_sum_known);
}
template <typename T>
void check(const std::vector<T>& input,int64_t rows,int64_t cols,int64_t ld) {
  boost::multiprecision::cpp_int maximum,reference_maximum;
  const auto expected=construct(input.data(),rows,cols,ld,reference_maximum);
  const auto actual=rns8::detail::native_exact_ranges(input.data(),rows,cols,ld,maximum);
  REQUIRE(actual.rows.size()==expected.rows.size());REQUIRE(actual.cols.size()==expected.cols.size());
  CHECK(maximum==reference_maximum);
  for(size_t i=0;i<actual.rows.size();++i)equal(actual.rows[i],expected.rows[i]);
  for(size_t i=0;i<actual.cols.size();++i)equal(actual.cols[i],expected.cols[i]);
}
template <typename T>
void cases() {
  uint64_t state=UINT64_C(0x93fba5821ae4d617);
  for(const auto shape:{std::pair<int64_t,int64_t>{1,1},{1,33},{3,15},{17,16},{19,17},{5,31},{7,32},{9,33},{257,3},{33,129},{0,4},{4,0}}) {
    const auto rows=shape.first,cols=shape.second,ld=cols+5;
    CAPTURE(rows,cols,ld);
    std::vector<T> values(size_t(rows*ld),std::numeric_limits<T>::max());
    for(int64_t r=0;r<rows;++r)for(int64_t c=0;c<cols;++c) {
      state=state*UINT64_C(6364136223846793005)+UINT64_C(1442695040888963407);
      if constexpr(std::is_signed_v<T>)values[size_t(r*ld+c)]=(state>>63)?INT64_MIN+int64_t(state&INT64_MAX):int64_t(state);
      else values[size_t(r*ld+c)]=state;
    }
    if(rows && cols)values[0]=std::numeric_limits<T>::lowest();
    if(rows && cols>1)values[1]=std::numeric_limits<T>::max();
    check(values,rows,cols,ld);
  }
}
}

TEST_CASE("native moment fixed accumulators preserve every arbitrary precision axis fact","[exact-moment]") {
  native_moment_reference::cases<int64_t>();native_moment_reference::cases<uint64_t>();
  for(const auto& values:std::vector<std::vector<int64_t>>{{-5,0},{INT64_MIN,INT64_MAX},{INT64_MIN,INT64_MIN},{INT64_MAX,INT64_MAX},{-7,-2}}) {
    native_moment_reference::check(values,1,2,2);
    native_moment_reference::check(values,2,1,1);
  }
}
