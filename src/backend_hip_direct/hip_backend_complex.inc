#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
extern "C" int rns8_hip_complex_sum(const int8_t *, const int8_t *, int8_t *, int, int, int, int,
                                    int, uint32_t, void*);
extern "C" int rns8_hip_complex_combine(const int8_t *, int8_t *, int8_t *, int, int, int, uint32_t,
                                        bool, void*);
extern "C" int rns8_hip_complex_commit(const int8_t *, const int8_t *, int8_t *, int8_t *, int, int,
                                       int, int, int, void*);
#endif
rns8_status hip_direct_gemm_complex_residues(int device_id,
                                             const ComplexResiduePair<const int8_t> &a,
                                             const ComplexResiduePair<const int8_t> &b,
                                             const ComplexResiduePair<int8_t> &c,
                                             const ComplexMagnitudeBounds &bounds,
                                             ComplexProductMethod method,
                                             uint64_t max_scratch_bytes, bool &commit_started) {
  commit_started = false;
  const auto plan = plan_complex_residue_product(a, b, c, bounds, method, max_scratch_bytes);
  if (plan.status != RNS8_SUCCESS)
    return plan.status;
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  auto status = set_hip_device(device_id);
  if (status != RNS8_SUCCESS)
    return status;
  hip_direct_device_buffer storage;
  status = storage.allocate(device_id, static_cast<std::size_t>(plan.scratch_bytes));
  if (status != RNS8_SUCCESS)
    return status;
  auto *sum_a = static_cast<int8_t *>(storage.get());
  auto *sum_b = sum_a + plan.sum_a_bytes;
  auto *products = sum_b + plan.sum_b_bytes;
  auto *stage_real = static_cast<int8_t*>(storage.get()) + plan.staged_real_offset();
  auto *stage_imag = static_cast<int8_t*>(storage.get()) + plan.staged_imag_offset();
  const auto &basis = a.real.basis;
  const bool three_real = method == ComplexProductMethod::three_real;
  if (three_real) {
    const auto transform = [&](const auto &pair, int8_t *output) {
      return timed_hip_operation("complex_residue_sum_recenter", [&]() {
        const auto launch = static_cast<hipError_t>(rns8_hip_complex_sum(
            pair.real.data, pair.imag.data, output, static_cast<int>(pair.real.rows),
            static_cast<int>(pair.real.cols), static_cast<int>(pair.real.ld),
            static_cast<int>(pair.imag.ld), basis.prefix, basis.modulus, nullptr));
        return launch == hipSuccess ? hipDeviceSynchronize() : launch;
      });
    };
    if (transform(a, sum_a) != hipSuccess || transform(b, sum_b) != hipSuccess)
      return RNS8_BACKEND_FAILURE;
  }
  using View = ComplexResidueView<const int8_t>;
  const View left[]{a.real, a.imag,
                    three_real ? View{sum_a, a.real.rows, a.real.cols, a.real.cols, basis} : a.real,
                    a.imag};
  const View right[]{
      b.real, b.imag,
      three_real ? View{sum_b, b.real.rows, b.real.cols, b.real.cols, basis} : b.imag, b.real};
  for (uint32_t p = 0; p < plan.products; ++p) {
    if (basis.domain == ComplexResidueDomain::finite_ring)
      status = hip_direct_gemm_finite_u8_resident_device(
          device_id, left[p].data, right[p].data, products + p * plan.product_bytes, a.real.rows,
          b.real.cols, a.real.cols, left[p].ld, right[p].ld, b.real.cols,
          static_cast<uint16_t>(basis.modulus));
    else
      status = hip_direct_gemm_rns_device(
          device_id, left[p].data, right[p].data, products + p * plan.product_bytes, a.real.rows,
          b.real.cols, a.real.cols, left[p].ld, right[p].ld, b.real.cols, basis.prefix);
    if (status != RNS8_SUCCESS)
      return status;
  }
  auto error = timed_hip_operation("complex_residue_recombine", [&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_complex_combine(
        products, stage_real, stage_imag, static_cast<int>(c.real.rows),
        static_cast<int>(c.real.cols), basis.prefix, basis.modulus, three_real, nullptr));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  if (error != hipSuccess)
    return RNS8_BACKEND_FAILURE;
  // One logical publication boundary: hardware stores are not atomic across
  // components. A subsequent failure invalidates the pair, never just one half.
  commit_started = true;
  error = timed_hip_operation("complex_residue_commit_pair", [&]() {
    const auto launch = static_cast<hipError_t>(rns8_hip_complex_commit(
        stage_real, stage_imag, c.real.data, c.imag.data, static_cast<int>(c.real.rows),
        static_cast<int>(c.real.cols), static_cast<int>(c.real.ld), static_cast<int>(c.imag.ld),
        basis.prefix, nullptr));
    return launch == hipSuccess ? hipDeviceSynchronize() : launch;
  });
  return error == hipSuccess ? RNS8_SUCCESS : RNS8_BACKEND_FAILURE;
#else
  (void)device_id;
  return RNS8_UNSUPPORTED_BACKEND;
#endif
}

rns8_status hip_direct_enqueue_complex_no_publish(const rns8_resident_node& node,
    const ComplexResidueScratch& resources,void* scratch,void* stream) {
#if defined(RNS8_ENABLE_HIP) && RNS8_ENABLE_HIP
  if(!scratch || !stream) return RNS8_INVALID_ARGUMENT;
  const auto& plan=*node.plan;
  const int m=static_cast<int>(plan.desc.m),n=static_cast<int>(plan.desc.n),k=static_cast<int>(plan.desc.k);
  const int prefix=static_cast<int>(plan.prefix);
  const bool three=node.complex_method==RNS8_COMPLEX_THREE_REAL_PRODUCTS;
  auto* sum_a=static_cast<int8_t*>(scratch);
  auto* sum_b=sum_a+resources.sum_a_bytes;
  auto* products=sum_b+resources.sum_b_bytes;
  auto* real=static_cast<int8_t*>(scratch)+resources.staged_real_offset();
  auto* imag=static_cast<int8_t*>(scratch)+resources.staged_imag_offset();
  const auto bytes=[](const rns8_matrix* matrix){return static_cast<const int8_t*>(matrix->hip_residues);};
  if(three) {
    auto error=rns8_hip_complex_sum(bytes(node.a),bytes(node.a_imag),sum_a,m,k,
        static_cast<int>(node.a->desc.cols),static_cast<int>(node.a_imag->desc.cols),prefix,0,stream);
    if(error!=hipSuccess) return RNS8_BACKEND_FAILURE;
    error=rns8_hip_complex_sum(bytes(node.b),bytes(node.b_imag),sum_b,k,n,
        static_cast<int>(node.b->desc.cols),static_cast<int>(node.b_imag->desc.cols),prefix,0,stream);
    if(error!=hipSuccess) return RNS8_BACKEND_FAILURE;
  }
  const int8_t* left[]{bytes(node.a),bytes(node.a_imag),three?sum_a:bytes(node.a),bytes(node.a_imag)};
  const int8_t* right[]{bytes(node.b),bytes(node.b_imag),three?sum_b:bytes(node.b_imag),bytes(node.b)};
  const int ld_left[]{static_cast<int>(node.a->desc.cols),static_cast<int>(node.a_imag->desc.cols),
      three?k:static_cast<int>(node.a->desc.cols),static_cast<int>(node.a_imag->desc.cols)};
  const int ld_right[]{static_cast<int>(node.b->desc.cols),static_cast<int>(node.b_imag->desc.cols),
      three?n:static_cast<int>(node.b_imag->desc.cols),static_cast<int>(node.b->desc.cols)};
  for(uint32_t product=0;product<resources.products;++product) {
    const auto status=launch_direct_rns_gemm_no_sync(left[product],right[product],
        products+product*resources.product_bytes,m,n,k,ld_left[product],ld_right[product],n,prefix,stream);
    if(status!=hipSuccess) return RNS8_BACKEND_FAILURE;
  }
  auto error=rns8_hip_complex_combine(products,real,imag,m,n,prefix,0,three,stream);
  if(error!=hipSuccess) return RNS8_BACKEND_FAILURE;
  error=rns8_hip_complex_commit(real,imag,static_cast<int8_t*>(node.c->hip_residues),
      static_cast<int8_t*>(node.c_imag->hip_residues),m,n,static_cast<int>(node.c->desc.cols),
      static_cast<int>(node.c_imag->desc.cols),prefix,stream);
  return error==hipSuccess?RNS8_SUCCESS:RNS8_BACKEND_FAILURE;
#else
  (void)node;(void)resources;(void)scratch;(void)stream;return RNS8_UNSUPPORTED_BACKEND;
#endif
}
