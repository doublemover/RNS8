#include "core/api_internal.hpp"
#include "backend_common/gram_plan.hpp"
using namespace rns8::detail;
using namespace rns8::detail::api;
rns8_status rns8_gram_residues(rns8_context* ctx,const rns8_matrix* x,rns8_matrix* c,
    rns8_gram_orientation orientation,rns8_gram_method method,uint64_t budget) {
  return guard_api([&]() -> rns8_status {
    if(!ctx || !x || !c || x==c || uint32_t(orientation)>1 || uint32_t(method)>1)
      return RNS8_INVALID_ARGUMENT;
    if(ctx->hip_failure_poisoned) return RNS8_BACKEND_FAILURE;
    if(x->in_flight_writer || c->in_flight_writer || c->in_flight_readers) return RNS8_BUSY;
    if(x->backend!=RNS8_BACKEND_CPU_REFERENCE && x->backend!=RNS8_BACKEND_HIP_DIRECT)
      return RNS8_UNSUPPORTED_BACKEND;
    const bool hip=x->backend==RNS8_BACKEND_HIP_DIRECT;
    const bool finite=uses_finite_storage(x->desc.semantics);
    if(!finite && x->desc.semantics!=RNS8_EXACT_WIDE_SIGNED) return RNS8_UNSUPPORTED_BACKEND;
    if(!context_accepts_backend(*ctx,x->backend) || x->generation_exhausted || c->generation_exhausted || c->backend!=x->backend ||
       !matrix_backend_compatible_with_plan(*ctx,*x,x->backend) ||
       !matrix_backend_compatible_with_plan(*ctx,*c,c->backend) ||
       (hip && (x->hip_device_id!=ctx->device_id || c->hip_device_id!=ctx->device_id)) ||
       c->desc.semantics!=x->desc.semantics || c->prefix!=x->prefix ||
       !rns_residue_state_current_for_backend(*x,x->backend)) return RNS8_INVALID_ARGUMENT;
    const int64_t n=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?x->desc.cols:x->desc.rows;
    const int64_t k=orientation==RNS8_GRAM_TRANSPOSE_TIMES_SELF?x->desc.rows:x->desc.cols;
    if(c->desc.rows!=n || c->desc.cols!=n) return RNS8_INVALID_ARGUMENT;
    for(const auto* matrix:{x,static_cast<const rns8_matrix*>(c)}) {
      if(!matrix_descriptor_matches(*matrix,x->desc.semantics,RNS8_BOUND_NONE,
          matrix->desc.rows,matrix->desc.cols,x->prefix,0,0) ||
         !(finite?finite_matrix_storage_matches(*matrix,x->backend,matrix->desc.rows,matrix->desc.cols)
                 :rns_matrix_storage_matches(*matrix,x->backend,matrix->desc.rows,matrix->desc.cols,x->prefix)))
        return RNS8_INVALID_ARGUMENT;
    }
    if(finite && !valid_finite_modulus_for_semantics(x->desc.semantics,x->finite_modulus))
      return RNS8_INVALID_ARGUMENT;
    boost::multiprecision::cpp_int bound=0;
    if(!finite) {
      if(!x->exact_range_prefix || x->exact_range_prefix>x->prefix || x->exact_max_magnitude<0 ||
         !exact_range_fits(x->desc.semantics,x->exact_max_magnitude,modulus_product(x->exact_range_prefix)))
        return RNS8_RANGE_ERROR;
      bound=boost::multiprecision::cpp_int(k)*x->exact_max_magnitude*x->exact_max_magnitude;
      if(!exact_range_fits(c->desc.semantics,bound,modulus_product(c->prefix))) return RNS8_RANGE_ERROR;
    }
    GramProblem p;
    p.orientation=static_cast<GramOrientation>(orientation);p.semantics=x->desc.semantics;
    p.source_instance=x->matrix_instance_id;p.source_generation=x->content_generation;
    p.rows=x->desc.rows;p.cols=x->desc.cols;p.source_ld=x->desc.cols;p.output_ld=c->desc.cols;
    p.prefix=finite?1:x->prefix;p.modulus=finite?x->finite_modulus:0;
    const auto support=query_gram_support(p,budget,hip,method==RNS8_GRAM_WMMA);
    if(support.status!=RNS8_SUCCESS) return support.status;
    bool started=false;
    struct Guard {
      rns8_matrix& c;bool& started;bool published=false;
      ~Guard(){if(started && !published) invalidate_output_currentness(c);}
    } write{*c,started};
    if(hip) {
      const auto status=hip_direct_gram_device(ctx->device_id,x->hip_residues,c->hip_residues,p,
          x->matrix_instance_id,x->content_generation,method==RNS8_GRAM_WMMA,budget,started);
      if(status!=RNS8_SUCCESS) return status;
      mark_output_device_residues_current(*c);
    } else {
      const auto jobs=gram_tiles(support);
      started=true;
      gram_residues_cpu(x->residues.data(),c->residues.data(),p,jobs,kDefaultModuli);
      mark_output_host_residues_current(*c);
    }
    c->finite_modulus=finite?x->finite_modulus:0;
    c->exact_axis_ranges.rows.clear();c->exact_axis_ranges.cols.clear();
    if(!finite) {c->exact_max_magnitude.swap(bound);c->exact_range_prefix=c->prefix;}
    write.published=true;
    return RNS8_SUCCESS;
  });
}
