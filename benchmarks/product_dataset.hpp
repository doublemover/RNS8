#pragma once
#include <boost/multiprecision/cpp_int.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <limits>
#include <string>

namespace rns8::bench {
inline std::array<boost::multiprecision::cpp_int,2> product_dataset_inputs(const nlohmann::json& c) {
  using boost::multiprecision::cpp_int;
  std::array<cpp_int,2> values{1,1};
  if(c.contains("input_values")) {
    const auto& a=c.at("input_values");
    if(!a.is_array() || a.size()!=2)throw std::invalid_argument("two constant native inputs required");
    for(unsigned i=0;i<2;++i) {
      if(!a[i].is_number_integer())throw std::invalid_argument("integer dataset input");
      values[i]=a[i].is_number_unsigned()?cpp_int(a[i].get<uint64_t>()):cpp_int(a[i].get<int64_t>());
      const cpp_int lo=c.at("unsigned_inputs").get<bool>()?cpp_int(0):cpp_int(INT64_MIN);
      const cpp_int hi=c.at("unsigned_inputs").get<bool>()?cpp_int(UINT64_MAX):cpp_int(INT64_MAX);
      if(values[i]<lo || values[i]>hi)throw std::invalid_argument("dataset native range");
    }
  }
  return values;
}
inline std::string product_dataset_identity(const nlohmann::json& c) {
  const auto values=product_dataset_inputs(c);
  return values[0]==1 && values[1]==1?"all_one_inputs_exact_integer_dag_v1":
      "constant_native64_dag_v1;A="+values[0].str()+";B="+values[1].str();
}
inline uint64_t product_dataset_word(const boost::multiprecision::cpp_int& value) {
  return value<0?static_cast<uint64_t>(value.convert_to<int64_t>()):value.convert_to<uint64_t>();
}
} // namespace rns8::bench
