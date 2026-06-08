TEST_CASE("sparse A 4:2 contract packs and expands canonical K groups") {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_SIGNED_I8;
  desc.rows = 2;
  desc.expanded_k = 8;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;

  uint64_t group_count = 0;
  uint64_t packed_value_count = 0;
  REQUIRE(rns8_sparse_a_4_to_2_layout_counts(&desc, &group_count, &packed_value_count) == RNS8_SUCCESS);
  CHECK(group_count == 4);
  CHECK(packed_value_count == 8);

  const std::vector<uint8_t> dense = {
      5, 0, 7, 0, 0, 9, 0, 11,
      0, 13, 17, 0, 19, 0, 0, 23,
  };
  std::vector<uint8_t> packed_values(static_cast<std::size_t>(packed_value_count), 0);
  std::vector<uint8_t> packed_indices(static_cast<std::size_t>(group_count), 0);
  REQUIRE(rns8_pack_sparse_a_4_to_2_u8(
              &desc, dense.data(), desc.expanded_k, packed_values.data(), packed_indices.data()) == RNS8_SUCCESS);
  CHECK(packed_values == std::vector<uint8_t>{5, 7, 9, 11, 13, 17, 19, 23});
  CHECK(packed_indices == std::vector<uint8_t>{8, 13, 9, 12});

  std::vector<uint8_t> expanded(dense.size(), 0xff);
  REQUIRE(rns8_expand_sparse_a_4_to_2_u8(
              &desc, packed_values.data(), packed_indices.data(), expanded.data(), desc.expanded_k) == RNS8_SUCCESS);
  CHECK(expanded == dense);
}

TEST_CASE("sparse A 4:2 contract rejects malformed descriptors and groups") {
  rns8_sparse_matrix_desc desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = RNS8_ABI_VERSION;
  desc.contract = RNS8_SPARSE_A_4_TO_2_STRUCTURED_K;
  desc.sparse_operand = RNS8_SPARSE_OPERAND_A;
  desc.index_layout = RNS8_SPARSE_INDEX_LAYOUT_CANONICAL_2BIT_K_GROUPS_V1;
  desc.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSIGNED_U8;
  desc.rows = 1;
  desc.expanded_k = 4;
  desc.group_size = 4;
  desc.nonzeros_per_group = 2;

  uint64_t group_count = 0;
  uint64_t packed_value_count = 0;
  auto bad_k = desc;
  bad_k.expanded_k = 6;
  CHECK(rns8_sparse_a_4_to_2_layout_counts(&bad_k, &group_count, &packed_value_count) == RNS8_INVALID_ARGUMENT);
  auto bad_signedness = desc;
  bad_signedness.value_signedness = RNS8_SPARSE_VALUE_SIGNEDNESS_UNSPECIFIED;
  CHECK(
      rns8_sparse_a_4_to_2_layout_counts(&bad_signedness, &group_count, &packed_value_count) ==
      RNS8_INVALID_ARGUMENT);

  const std::vector<uint8_t> too_many = {1, 2, 3, 0};
  std::vector<uint8_t> packed_values(2, 0);
  std::vector<uint8_t> packed_indices(1, 0);
  CHECK(rns8_pack_sparse_a_4_to_2_u8(
            &desc, too_many.data(), desc.expanded_k, packed_values.data(), packed_indices.data()) ==
        RNS8_INVALID_ARGUMENT);

  const std::vector<uint8_t> too_few = {1, 0, 0, 0};
  CHECK(rns8_pack_sparse_a_4_to_2_u8(
            &desc, too_few.data(), desc.expanded_k, packed_values.data(), packed_indices.data()) ==
        RNS8_INVALID_ARGUMENT);

  packed_values = {7, 9};
  packed_indices = {0xf1};
  std::vector<uint8_t> dense(4, 0);
  CHECK(rns8_expand_sparse_a_4_to_2_u8(
            &desc, packed_values.data(), packed_indices.data(), dense.data(), desc.expanded_k) ==
        RNS8_INVALID_ARGUMENT);
}
