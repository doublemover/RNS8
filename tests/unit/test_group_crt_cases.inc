TEST_CASE("group CRT preserves canonical subgroup identity and checked complete roots") {
  using namespace rns8::detail;
  using boost::multiprecision::cpp_int;
  const auto wide=[](balanced_crt::Wide x) -> cpp_int {
    return cpp_int(x.limb[0])+(cpp_int(x.limb[1])<<64)+(cpp_int(x.limb[2])<<128);
  };
  std::mt19937_64 random(3939);
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    cpp_int product=1;
    for(uint32_t p=0;p<prefix;++p)product*=garner::kConstants.moduli[p];
    CHECK(group_crt::plan(prefix).group_count==(prefix+7)/8);
    for(uint32_t count=(prefix+7)/8;count<=std::min(prefix,4u);++count)for(uint32_t variant=0;variant<3;++variant) {
      CAPTURE(prefix,count,variant);
      std::array<uint8_t,20> partition{};
      for(uint32_t p=0;p<prefix;++p)partition[p]=static_cast<uint8_t>(variant==1?p%count:p*count/prefix);
      if(variant==2)std::shuffle(partition.begin(),partition.begin()+prefix,random);
      const auto plan=group_crt::make_plan(prefix,partition.data(),count);
      REQUIRE(plan.merge_count+1==count);
      uint32_t covered=0;
      std::array<cpp_int,4> subtree_products;
      for(uint32_t g=0;g<count;++g) {
        const auto& group=plan.groups[g];REQUIRE(group.count);REQUIRE(group.count<=8);
        CHECK_FALSE(covered&group.channel_mask);covered|=group.channel_mask;
        cpp_int partial=1;
        for(uint32_t i=0;i<group.count;++i) {
          const auto modulus=garner::kConstants.moduli[group.channels[i]];
          CHECK(group.moduli[i]==modulus);
          CHECK((partial*group.inverses[i])%modulus==1);
          partial*=modulus;
        }
        CHECK(partial<=UINT64_MAX);CHECK(partial==group.product);subtree_products[g]=partial;
      }
      CHECK(covered==(uint32_t(1)<<prefix)-1);
      for(uint32_t i=0;i<plan.merge_count;++i) {
        const auto& merge=plan.merges[i];REQUIRE(merge.left<count);REQUIRE(merge.right<count);
        CHECK(wide(merge.a)==subtree_products[merge.left]);CHECK(wide(merge.b)==subtree_products[merge.right]);
        CHECK((wide(merge.a)*wide(merge.inverse))%wide(merge.b)==1);
        subtree_products[merge.left]*=subtree_products[merge.right];
        CHECK(subtree_products[merge.left]<(cpp_int(1)<<160));
      }
      CHECK(subtree_products[0]==product);
      for(uint32_t sample=0;sample<24;++sample) {
        cpp_int value=(cpp_int(random())+(cpp_int(random())<<64)+(cpp_int(random())<<128))%product;
        if(sample==0)value=0;else if(sample==1)value=1;
        else if(sample==2)value=product-1;else if(sample==3)value=product/2;
        else if(sample==4)value=product/2+1;
        std::array<int8_t,20> residues{};
        for(uint32_t p=0;p<prefix;++p) {
          const auto m=garner::kConstants.moduli[p];int r=static_cast<int>(value%m);
          if(r>=int((m+1)/2))r-=int(m);residues[p]=static_cast<int8_t>(r);
        }
        std::array<uint64_t,4> partials{};
        const auto orders=sample<5?2*count:1u;
        for(uint32_t order=0;order<orders;++order) {
          uint32_t arrived=0;
          for(uint32_t physical=0;physical<count;++physical) {
            const auto group=group_crt::producer_group(physical,order,count);
            CHECK_FALSE(arrived&(uint32_t(1)<<group));arrived|=uint32_t(1)<<group;
            partials[group]=group_crt::reconstruct_group(residues.data(),1,0,plan.groups[group]);
            CHECK(cpp_int(partials[group])==value%plan.groups[group].product);
          }
          CHECK(wide(group_crt::merge_groups(partials.data(),plan))==value);
        }
      }
    }
  }
}

TEST_CASE("group CRT rejects malformed partitions and reports exact scratch budgets") {
  using namespace rns8::detail;
  std::array<uint8_t,20> partition{};
  CHECK_THROWS_AS(group_crt::make_plan(0,partition.data(),1),std::invalid_argument);
  CHECK_THROWS_AS(group_crt::make_plan(21,partition.data(),1),std::invalid_argument);
  CHECK_THROWS_AS(group_crt::make_plan(1,nullptr,1),std::invalid_argument);
  CHECK_THROWS_AS(group_crt::make_plan(1,partition.data(),0),std::invalid_argument);
  CHECK_THROWS_AS(group_crt::make_plan(9,partition.data(),1),std::invalid_argument);
  CHECK_THROWS_AS(group_crt::make_plan(8,partition.data(),2),std::invalid_argument);
  partition[0]=4;CHECK_THROWS_AS(group_crt::make_plan(8,partition.data(),4),std::invalid_argument);
  for(uint32_t prefix=1;prefix<=20;++prefix) {
    const auto bytes=uint64_t(257)*((prefix+7)/8)*8;
    const auto local=query_group_crt_support(257,prefix,false,0);
    CHECK(local.status==RNS8_SUCCESS);CHECK(local.scratch_bytes==0);CHECK_FALSE(local.requires_value_proof);
    const auto split=query_group_crt_support(257,prefix,true,bytes);
    CHECK(split.status==RNS8_SUCCESS);CHECK(split.scratch_bytes==bytes);CHECK(split.payload_bytes==257*24);
    const auto short_budget=query_group_crt_support(257,prefix,true,bytes-1);
    CHECK(short_budget.status==RNS8_WORKSPACE_TOO_SMALL);CHECK(short_budget.scratch_bytes==bytes);
  }
  CHECK(query_group_crt_support(INT64_MAX,20,true,UINT64_MAX).status==RNS8_UNSUPPORTED_BACKEND);
}
