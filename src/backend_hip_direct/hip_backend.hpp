#ifndef RNS8_BACKEND_HIP_DIRECT_HPP
#define RNS8_BACKEND_HIP_DIRECT_HPP

#include <cstddef>
#include <cstdint>
#include "backend_common/gram_plan.hpp"
#include "backend_common/signed_pack_policy.hpp"
#include <string>
#include <vector>

#include "rns8/rns8.h"
#include "backend_common/residue_layout.hpp"
#include "backend_common/native_tile_bound.hpp"
#include "backend_common/native_tiled_resources.hpp"
#include "backend_common/operation_support.hpp"
#include "backend_common/wide_gemm_support.hpp"
#include "backend_common/adaptive_native_support.hpp"
#include "backend_common/native_wmma_queue.hpp"
#include "backend_common/ragged_residues.hpp"
#include "backend_common/block_sparse.hpp"
#include "backend_common/integrity_identity.hpp"
#include "backend_common/native_narrow.hpp"

namespace rns8::detail {
template<class Byte> struct ComplexResiduePair;
struct ComplexMagnitudeBounds;
enum class ComplexProductMethod;
// Private split-complex composition. Before commit_started, both outputs are
// preserved; after a commit failure, caller must invalidate BOTH components.
// Caller owns input certificates, lifetimes and joint publication/ownership.
rns8_status hip_direct_gemm_complex_residues(
    int device_id, const ComplexResiduePair<const int8_t>& a,
    const ComplexResiduePair<const int8_t>& b, const ComplexResiduePair<int8_t>& c,
    const ComplexMagnitudeBounds& bounds, ComplexProductMethod method,
    uint64_t max_scratch_bytes, bool& commit_started);
struct ComplexResidueScratch;
// Private prepared complex enqueue: caller has admitted complete operands,
// exact component bounds, disjoint scratch capacity and paired output writers.
// No allocation, synchronization, proof evaluation or publication here.
rns8_status hip_direct_enqueue_complex_no_publish(const rns8_resident_node& node,
    const ComplexResidueScratch& resources,void* scratch,void* stream);
// Private synchronous heterogeneous native-input queue. Caller owns integer
// range proofs and output invalidation once outputs_may_be_written is set.
rns8_status hip_direct_gemm_native_wmma_queue(
    int device_id, const std::vector<NativeWmmaTask>& tasks, uint32_t workers,
    uint64_t max_scratch_bytes, bool& outputs_may_be_written);
// Read-only private sentinel. Failure bits: 1 malformed residue, 2 bound,
// 4 check-257, 8 check-263. Successful checks do not prove arbitrary correctness.
rns8_status hip_direct_check_native_product_integrity(
    int device_id,const void* a,const void* b,const void* candidate,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,bool unsigned_inputs,const uint64_t bound[3],
    const IntegrityIdentity& expected,const IntegrityIdentity& candidate_identity,uint32_t& failures);
// Stream-ordered sentinel. Caller retains device and pinned-host flags and
// all operands until its covering event or a successful drain completes.
rns8_status hip_direct_enqueue_native_product_integrity(
    int device_id,const void* a,const void* b,const void* candidate,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,bool unsigned_inputs,const uint64_t bound[3],
    const IntegrityIdentity& expected,const IntegrityIdentity& candidate_identity,
    void* device_flag,void* pinned_host_flag,void* stream);
// Private one-level modular Strassen. Caller owns integer range admission and
// invalidation after a failed final commit; earlier failures preserve C.
rns8_status hip_direct_gemm_strassen_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t max_scratch_bytes,bool& commit_started,void* retained_scratch=nullptr,uint64_t retained_bytes=0,bool matrix_subproducts=false);
rns8_status hip_direct_gemm_rank23_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t max_scratch_bytes,bool& commit_started,void* retained_scratch=nullptr,uint64_t retained_bytes=0,bool matrix_subproducts=false);
// Exact pinned alternative-basis SLP, separate from the coefficient control.
// Full 23-operand/product materialization is included in the scratch budget.
rns8_status hip_direct_gemm_rank23_slp_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t max_scratch_bytes,bool& commit_started,void* retained_scratch=nullptr,uint64_t retained_bytes=0,bool matrix_subproducts=false);
// October 23-product/55-addition DAG, retained inputs or explicitly recomputed inputs.
rns8_status hip_direct_gemm_rank23_oct55_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch=nullptr,uint64_t retained_bytes=0,bool matrix_subproducts=false);
rns8_status hip_direct_gemm_rank23_oct55_recompute_device(
    int device_id,const void* a,const void* b,void* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint64_t budget,bool& commit_started,
    void* retained_scratch=nullptr,uint64_t retained_bytes=0,bool matrix_subproducts=false);
// Private modular block-sparse producer; its public materialization owner or
// another caller supplies bounded/exact admission and output transaction ownership.
rns8_status hip_direct_gemm_block_sparse_device(
    int device_id,const void* a,const BlockSparseOperand& lhs,uint64_t a_instance,uint64_t a_generation,
    const void* b,const BlockSparseOperand& rhs,uint64_t b_instance,uint64_t b_generation,
    void* c,int64_t ldc,uint64_t max_scratch_bytes,bool* commit_started = nullptr,
    uint32_t method = 0,uint32_t scalar_intersection_limit = 0);
// Private exact ragged-to-uniform materialization. Caller owns generation/proof
// binding and must invalidate destination on any failure after commit_started.
// Borrowed scratch must remain alive until a successful drain, including failures.
rns8_status hip_direct_discover_ragged_bounds(int device_id, const void* source,
    uint32_t prefix, const uint64_t bound[3], RaggedResidueLayout& layout,
    void* device_directory, void* device_status);
rns8_status hip_direct_compact_ragged_residues(int device_id, const void* source,
    uint32_t prefix, const RaggedResidueLayout& layout, void* destination, void* device_directory);
rns8_status hip_direct_materialize_ragged_residues(
    int device_id, const void* source, const RaggedResidueLayout& layout,
    void* destination, const ResidueLayout& target, uint64_t instance, uint64_t generation,
    uint64_t max_scratch_bytes, bool& commit_started,
    void* staged, void* directory, void* device_status, bool prepared_directory_current=false);
// Synchronous private discovery for fixed 16x16 output tiles. Caller allocates
// ceil(m/16)*ceil(n/16) bounds/indices and 22 offsets. Bucket 0 is proven zero;
// buckets 1..20 hold strict minimal prefixes. Order within a bucket is unspecified.
// Outputs are usable only on success and while these exact input values remain
// unchanged; these raw records are not public generation-bound matrix proofs.
rns8_status hip_direct_discover_native_tile_bounds(
    int device_id, const void* a, const void* b, int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, bool unsigned_inputs, uint32_t capacity,
    NativeTileBound* device_bounds, uint32_t* device_indices, uint32_t* device_offsets,
    uint64_t max_scratch_bytes, NativeBoundMethod method = NativeBoundMethod::tile_max,
    uint32_t panel_k = 256);
// Private one-shot native conversion + grouped matrix-engine candidate. The
// explicit flag selects i64/u64 interpretation; output is default-basis residues,
// not a proof of bounded/exact-wide integer range. Group is 1, 2 or 4.
// Private synchronous exact adaptive producer. Fresh bounds select each tile's
// minimal prefix; exact suffix completion makes every output plane valid.
// compact_workers=0 retains the grid control; 1..65535 bounds compact workers.
// Inputs must remain immutable through return. Caller owns publication and must
// invalidate output on any failure after commit_started becomes true.
rns8_status hip_direct_gemm_adaptive_native_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    bool unsigned_inputs, uint32_t capacity, uint32_t group,
    uint64_t max_scratch_bytes, bool& commit_started,
    NativeBoundMethod method = NativeBoundMethod::tile_max, uint32_t panel_k = 256,
    uint32_t compact_workers = 0);
// Private four-wave 32x32 raw-native staging experiment. buffers=0/1/2;
// depth=16/32 (register control requires16). LDS is per block, not global scratch.
// On failure after commit_started, caller must invalidate potentially written C.
rns8_status hip_direct_gemm_native_tiled_device(
    int device_id,const void* a,const void* b,int8_t* c,int64_t m,int64_t n,int64_t k,
    int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix,uint32_t group,bool unsigned_inputs,
    uint32_t depth,uint32_t buffers,uint64_t max_lds_bytes,bool& commit_started);
rns8_status hip_direct_gemm_native_wmma_grouped_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t group, bool unsigned_inputs);
// Explicit finite-ring candidate: input signed bytes are reduced modulo 2..16;
// output uses centered residues. No default-ladder RNS interpretation.
rns8_status hip_direct_gemm_finite_int4_device(
    int device_id, const int8_t* a, const int8_t* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, uint32_t modulus);
// Explicit preparation caches a device/code-object capability result. Admitted
// execution never probes or allocates. Caller owns currentness and failure drain.
rns8_status hip_direct_prepare_int4_capability(int device_id);
rns8_status hip_direct_pack_int4_b(int device_id,const int8_t* b,void* packed,int64_t k,int64_t n,int64_t ldb,uint32_t modulus);
rns8_status hip_direct_gemm_int4_admitted(int device_id,const int8_t* a,const int8_t* b,const void* packed_b,int8_t* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t modulus);
// Research direct CRT to canonical [0,P) values, always three little-endian
// uint64 limbs per cell. This does not infer signedness or bounded output semantics.
rns8_status hip_direct_export_canonical_direct_crt_device(
    int device_id, const void* residues, void* limbs, int64_t cells, uint32_t prefix);
// Admitted quarter-range caller only. The mode is signed0/unsigned1/canonical2.
// Separate host certificate owns source/range/width identity; this raw primitive
// does not infer that identity or claim target instruction qualification.
rns8_status hip_direct_export_quarter_crt_device(int device_id,const void* residues,void* limbs,
    int64_t cells,uint32_t prefix,uint32_t digits,uint32_t limb_count,uint32_t mode);
rns8_status hip_direct_export_canonical_balanced_crt_device(
    int device_id, const void* residues, void* limbs, int64_t cells, uint32_t prefix);

// Direct descriptor-aware consumer: no intermediate modulus-major allocation.
// Logical row-major cells become three canonical limbs; physical padding is ignored.
rns8_status hip_direct_export_canonical_balanced_crt_layout_device(
    int device_id, const void* residues, const ResidueLayout& source, void* limbs);

rns8_status hip_direct_convert_residue_layout(
    int device_id, const void* src, const ResidueLayout& source, void* dst, const ResidueLayout& destination);

// Native signed-i64 scalar candidate admitted by an actual device input scan.
// The returned bound covers every partial absolute product sum. No automatic
// selection or matrix-engine claim; reject rather than infer safety from C.
rns8_status hip_direct_gemm_i64_narrow_checked_device(
    int device_id, const int64_t* a, const int64_t* b, int64_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint64_t* out_abs_sum_bound,uint32_t rns_prefix=0,bool* write_started=nullptr);

// Staged native product followed by complete RNS encoding. Only the encoding
// launch can touch C; commit_started stays false for scan/domain failures.
rns8_status hip_direct_gemm_narrow_rns_device(
    int device_id, const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
    uint32_t prefix, bool unsigned_inputs, bool byte_wmma,
    uint64_t budget, bool& commit_started,bool fused=false);

// Explicit raw-byte WMMA candidate with native i64/u64 storage and output.
// Scan actual operands for signed [-128,127] or unsigned [0,255] membership,
// then prove all INT32 partials. No residue interpretation, cached proof or AUTO
// dispatch. Caller must keep inputs stable and invalidate C after launch failure.
rns8_status hip_direct_gemm_native_byte_wmma_checked_device(
    int device_id, const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    bool unsigned_inputs, uint64_t* out_abs_sum_bound,uint32_t rns_prefix=0,bool* write_started=nullptr);

// Explicit modular expression baseline. Callers must separately prove bounded
// integer semantics and own publication; D/C must not alias. No default routing.
// Private standalone modular combine baseline; C must be distinct from X/D.
rns8_status hip_direct_axpby_residues_device(
    int device_id, const void* x, const void* d, void* c, int64_t m, int64_t n,
    int64_t ldx, int64_t ldd, int64_t ldc, uint32_t prefix, int64_t alpha, int64_t beta);
rns8_status hip_direct_gemm_rns_axpby_device(
    int device_id, const void* a, const void* b, const void* d, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldd, int64_t ldc,
    uint32_t prefix, int64_t alpha, int64_t beta);

// Research baseline: synchronous explicit split-K, no automatic dispatch.
// Scratch holds canonical byte residues; output follows ordinary backend write
// semantics and must be protected by the caller's publication transaction.
rns8_status hip_direct_gemm_rns_split_k_device(
    int device_id, const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t splits, uint64_t max_scratch_bytes);

// Borrowed scratch is retained by the caller across errors. No allocation/free.
// commit_started changes only before the final store, never during partial work.
rns8_status hip_direct_gemm_rns_split_k_workspace(
    int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,
    uint32_t prefix,uint32_t splits,void* scratch,uint64_t scratch_bytes,bool& commit_started,uint32_t fragment_window=0);

// Native matrix-engine split-K: persistent grouped producers, canonical byte
// merge tree, then final store. Output is untouched until commit_started=true.
rns8_status hip_direct_gemm_native_wmma_split_k_device(
    int device_id, const void* a, const void* b, int8_t* c,
    int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
    uint32_t prefix, uint32_t group, bool unsigned_inputs, uint32_t splits,
    uint32_t workers, uint64_t max_scratch_bytes, bool& commit_started,
    bool separate_canonicalization_control = false);

rns8_status hip_direct_enqueue_exact_lift(int device_id,void* residues,int64_t cells,
    uint32_t source_prefix,uint32_t target_prefix,bool signed_value,const uint64_t* bound,
    void* scratch,uint64_t scratch_bytes,void* pinned_status,bool mixed_radix,void* stream);

// Borrowed status+suffix region; no allocations/frees. Caller retains and drains
// it on errors. The first int is aligned, suffix follows; publication is unchanged.
rns8_status hip_direct_lift_exact_suffix_workspace(
    int device_id,void* residues,int64_t cells,uint32_t source_prefix,uint32_t target_prefix,
    bool signed_value,const uint64_t* bound,void* scratch,uint64_t scratch_bytes,
    bool& commit_started,bool mixed_radix=false);

rns8_status hip_direct_lift_exact_suffix(
    int device_id, void* residues, int64_t cells, uint32_t source_prefix, uint32_t target_prefix,
    bool signed_value, const uint64_t* bound, bool& commit_started, bool mixed_radix = false);


struct hip_direct_timing_sample {
  std::string label;
  double microseconds = 0.0;
};

struct hip_direct_allocation_counters {
  uint64_t allocate_calls = 0;
  uint64_t free_calls = 0;
  uint64_t allocated_bytes = 0;
  uint64_t h2d_calls = 0;
  uint64_t h2d_bytes = 0;
  uint64_t d2h_calls = 0;
  uint64_t d2h_bytes = 0;
  uint64_t d2d_calls = 0;
  uint64_t d2d_bytes = 0;
};

struct hip_direct_grouped_gemm_task {
  rns8_matrix* a = nullptr;
  rns8_matrix* b = nullptr;
  rns8_matrix* c = nullptr;
  rns8_workspace* workspace = nullptr;
};

struct hip_direct_grouped_gemm_descriptor {
  const rns8_plan* plan = nullptr;
  const hip_direct_grouped_gemm_task* tasks = nullptr;
  uint32_t task_count = 0;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
};

struct hip_direct_grouped_gemm_bucket {
  hip_direct_grouped_gemm_descriptor descriptor{};
  uint32_t task_offset = 0;
  uint32_t task_count = 0;
};

struct hip_direct_grouped_gemm_bucket_plan {
  const rns8_plan* plan = nullptr;
  const hip_direct_grouped_gemm_task* tasks = nullptr;
  uint32_t task_count = 0;
  uint32_t bucket_count = 0;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
  bool same_shape_required = true;
  std::vector<hip_direct_grouped_gemm_bucket> buckets;
};

struct hip_direct_grouped_gemm_bucket_request {
  const rns8_plan* plan = nullptr;
  const hip_direct_grouped_gemm_task* tasks = nullptr;
  uint32_t task_offset = 0;
  uint32_t task_count = 0;
  rns8_semantics semantics = RNS8_BOUNDED_I64;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  uint32_t prefix = 0;
};

struct hip_direct_grouped_device_buffer {
  int device_id = -1;
  void* ptr = nullptr;
  std::size_t bytes = 0;

  hip_direct_grouped_device_buffer() = default;
  hip_direct_grouped_device_buffer(const hip_direct_grouped_device_buffer&) = delete;
  hip_direct_grouped_device_buffer& operator=(const hip_direct_grouped_device_buffer&) = delete;
  hip_direct_grouped_device_buffer(hip_direct_grouped_device_buffer&& other) noexcept;
  hip_direct_grouped_device_buffer& operator=(hip_direct_grouped_device_buffer&& other) noexcept;
  ~hip_direct_grouped_device_buffer();

  rns8_status allocate(int requested_device_id, std::size_t requested_bytes);
  rns8_status reset() noexcept;

 private:
  void move_from(hip_direct_grouped_device_buffer& other) noexcept;
};

struct hip_direct_grouped_device_resources {
  int device_id = -1;
  hip_direct_grouped_device_buffer a_slab;
  hip_direct_grouped_device_buffer b_slab;
  hip_direct_grouped_device_buffer c_slab;
  hip_direct_grouped_device_buffer status;
  hip_direct_grouped_device_buffer a_residue_ptrs;
  hip_direct_grouped_device_buffer b_residue_ptrs;
  hip_direct_grouped_device_buffer c_residue_ptrs;
  std::vector<rns8_matrix*> a_matrices;
  std::vector<rns8_matrix*> b_matrices;
  std::vector<rns8_matrix*> c_matrices;

  hip_direct_grouped_device_resources() = default;
  hip_direct_grouped_device_resources(const hip_direct_grouped_device_resources&) = delete;
  hip_direct_grouped_device_resources& operator=(const hip_direct_grouped_device_resources&) = delete;
  hip_direct_grouped_device_resources(hip_direct_grouped_device_resources&&) noexcept = default;
  hip_direct_grouped_device_resources& operator=(hip_direct_grouped_device_resources&&) noexcept = default;

  rns8_status reset() noexcept;
};

// Preallocated cleanup record for private scratch whose device completion is
// uncertain. Queue insertion cannot allocate; successful device drains reclaim it.
enum class hip_deferred_allocation_kind { device, pinned_host };
struct hip_deferred_buffer_cleanup {
  hip_deferred_allocation_kind kind=hip_deferred_allocation_kind::device;
  int device_id=-1;
  void* pointer=nullptr;
  hip_deferred_buffer_cleanup* next=nullptr;
};
void hip_direct_defer_buffer_cleanup(hip_deferred_buffer_cleanup*) noexcept;
rns8_status hip_direct_collect_deferred_buffers(int device_id);

bool hip_direct_compiled();
void hip_direct_timing_set_enabled(bool enabled);
bool hip_direct_timing_enabled();
void hip_direct_timing_reset();
void hip_direct_timing_flush_pending_events();
void hip_direct_timing_record_sample(const char* label, double microseconds);
void hip_direct_timing_record_pending_event(const char* label, void* start_event, void* stop_event);
void hip_direct_timing_record_pending_event_with_alias(
    const char* label,
    const char* alias,
    void* start_event,
    void* stop_event);
std::vector<hip_direct_timing_sample> hip_direct_timing_snapshot();
void hip_direct_allocation_counters_reset();
hip_direct_allocation_counters hip_direct_allocation_counters_snapshot();
rns8_status hip_direct_probe(int device_id, rns8_device_info& out);
rns8_status hip_direct_allocate(int device_id, std::size_t bytes, void** out);
rns8_status hip_direct_free(int device_id, void* ptr);
// Completes the zero-fill before returning, including for cross-stream reuse.
rns8_status hip_direct_zero(int device_id, void* ptr, std::size_t bytes);
rns8_status hip_direct_copy_device_to_host(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_copy_host_to_device(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_copy_host_to_device_labeled(
    int device_id,
    const char* timing_label,
    void* dst,
    const void* src,
    std::size_t bytes);
rns8_status hip_direct_copy_device_to_device(int device_id, void* dst, const void* src, std::size_t bytes);
rns8_status hip_direct_ensure_upload_buffer(int device_id, std::size_t bytes, void** buffer, std::size_t* capacity);
rns8_status hip_direct_copy_compact_matrix_device_to_host(
    int device_id,
    const char* timing_label,
    void* dst,
    int64_t dst_ld,
    const void* src,
    int64_t rows,
    int64_t cols,
    std::size_t cell_bytes,
    bool default_padded_staging = true);
rns8_status hip_direct_build_same_shape_grouped_bucket_plan(
    const rns8_plan* plan,
    const hip_direct_grouped_gemm_task* tasks,
    uint32_t task_count,
    rns8_semantics semantics,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix,
    hip_direct_grouped_gemm_bucket_plan* out);
rns8_status hip_direct_build_bucketed_grouped_gemm_plan(
    const hip_direct_grouped_gemm_bucket_request* requests,
    uint32_t request_count,
    hip_direct_grouped_gemm_bucket_plan* out);
const hip_direct_grouped_gemm_descriptor* hip_direct_single_bucket_descriptor(
    const hip_direct_grouped_gemm_bucket_plan& bucket_plan);
rns8_status hip_direct_validate_grouped_gemm_descriptor_setup(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    int* out_device_id = nullptr);
rns8_status hip_direct_validate_grouped_gemm_descriptor_after_pack(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    uint64_t first_source_version);
rns8_status hip_direct_validate_grouped_gemm_descriptor_after_gemm(
    const hip_direct_grouped_gemm_descriptor& descriptor);
rns8_status hip_direct_allocate_grouped_task_device_resources(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    std::size_t a_slab_bytes,
    std::size_t b_slab_bytes,
    std::size_t c_slab_bytes,
    std::size_t status_bytes,
    hip_direct_grouped_device_resources* out);
rns8_status hip_direct_prepare_grouped_task_residue_pointers(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    int* out_device_id = nullptr);
rns8_status hip_direct_pack_grouped_i64_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const int64_t* a_slab,
    const int64_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint64_t first_source_version);
rns8_status hip_direct_pack_grouped_u64_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const uint64_t* a_slab,
    const uint64_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint64_t first_source_version);
rns8_status hip_direct_pack_grouped_finite_u8_task_inputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    const uint8_t* a_slab,
    const uint8_t* b_slab,
    int64_t lda,
    int64_t ldb,
    uint16_t modulus,
    uint64_t first_source_version);
rns8_status hip_direct_gemm_grouped_rns_task_outputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources);
rns8_status hip_direct_gemm_grouped_finite_u8_task_outputs(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint16_t modulus);
rns8_status hip_direct_export_grouped_i64_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint64_t bound,
    int64_t* dst,
    std::size_t dst_elements);
rns8_status hip_direct_export_grouped_u64_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint64_t bound,
    uint64_t* dst,
    std::size_t dst_elements);
rns8_status hip_direct_export_grouped_finite_u8_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint16_t modulus,
    uint8_t* dst,
    std::size_t dst_elements);
rns8_status hip_direct_export_grouped_exact_wide_task_outputs_to_host(
    const hip_direct_grouped_gemm_descriptor& descriptor,
    hip_direct_grouped_device_resources& resources,
    uint32_t limb_count,
    bool signed_output,
    uint64_t* dst,
    std::size_t dst_limb_elements);
// Research controls share the exact public signed-pack selector. Every entry
// is synchronous; inner wrappers only enqueue. No plan/default promotion.
rns8_status hip_direct_pack_i64_variant_device(int device_id, const void* device_source,
    void* device_residues, int64_t rows, int64_t cols, int64_t ld, uint32_t prefix,
    SignedPackVariant variant);
rns8_status hip_direct_pack_i64_device(
    int device_id,
    const int64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix);
rns8_status hip_direct_pack_u64_device(
    int device_id,
    const uint64_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix);
rns8_status hip_direct_pack_i64_grouped_matrices_device(
    int device_id,
    const int64_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    uint64_t first_source_version);
rns8_status hip_direct_pack_u64_grouped_matrices_device(
    int device_id,
    const uint64_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint32_t prefix,
    uint64_t first_source_version);
rns8_status hip_direct_native_i64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix);
rns8_status hip_direct_native_u64_to_rns_device(
    int device_id,
    const void* device_native,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    uint32_t prefix);
rns8_status hip_direct_pack_finite_u8_device(
    int device_id,
    const uint8_t* src,
    void** upload_buffer,
    std::size_t* upload_bytes,
    void* device_residues,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus);
rns8_status hip_direct_pack_finite_u8_grouped_matrices_device(
    int device_id,
    const uint8_t* src_slab,
    void* device_src_slab,
    std::size_t device_src_slab_bytes,
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_residue_ptrs,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    uint16_t modulus,
    uint64_t first_source_version);
rns8_status hip_direct_ring_gemm_i8_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus,
    uint32_t modulus_index,
    uint32_t selected_prefix);
// Private scalar u64-index residue control. Caller owns range proof and write
// invalidation after execution failure; no public/default dispatch promotion.
rns8_status hip_direct_gemm_rns_wide_device(int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,int64_t lda,int64_t ldb,int64_t ldc,uint32_t prefix);
rns8_status hip_direct_gemm_rns_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint32_t prefix,bool parallel_skinny=false);
rns8_status hip_direct_gemm_rns_region_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t region_row_offset,
    int64_t region_col_offset,
    int64_t region_rows,
    int64_t region_cols,
    uint32_t prefix);
rns8_status hip_direct_gemm_rns_grouped_exact_wide_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix);
rns8_status hip_direct_gemm_rns_grouped_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint32_t prefix);
rns8_status hip_direct_gemm_rns_matrix_launch_current_device_no_sync(
    const rns8_plan* plan,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    void* stream = nullptr);
// Enqueue only: no matrix currentness or generation publication.
rns8_status hip_direct_enqueue_axpby_no_publish(const rns8_plan*, const rns8_matrix*,
    const rns8_matrix*, const rns8_matrix*, rns8_matrix*, int64_t alpha, int64_t beta, void* stream);
rns8_status hip_direct_enqueue_finite_no_publish(const rns8_plan*,const rns8_matrix*,
    const rns8_matrix*,rns8_matrix*,void* stream);
rns8_status hip_direct_enqueue_rns_no_publish(const rns8_plan*,const rns8_matrix*,const rns8_matrix*,rns8_matrix*,void* stream);
rns8_status hip_direct_gemv_n1_rns_prefix9_matrix(
    int device_id,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t k,
    uint64_t source_version);
rns8_status hip_direct_gemv_small_n_rns_prefix9_matrix(
    int device_id,
    const rns8_matrix* A,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    uint64_t source_version);
rns8_status hip_direct_gemm_i64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_i64_native_a_resident_b_prefix9_colpair_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_i64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_native_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_native_a_resident_b_prefix9_colpair_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_i64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_resident_a_native_b_prefix9_colpair_matrix(
    int device_id,
    const rns8_matrix* A,
    const void* device_b_native,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_u64_uniform_small_native_a_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_b_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_resident_a_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_device(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc);
rns8_status hip_direct_gemm_uniform_small_i8_ab_colpair_transient_prefix9_matrix(
    int device_id,
    const void* device_a_i8,
    const void* device_b_i8,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    uint64_t source_version);
rns8_status hip_direct_gemm_rns_tiled_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* entries,
    uint64_t entry_count);
rns8_status hip_direct_gemm_rns_tiled_device_schedule(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const rns8_plan_tile_schedule_entry* host_entries,
    const void* device_entries,
    const void* active_device_entries,
    const void* zero_a_rows,
    const void* zero_b_cols,
    const uint64_t* active_offsets,
    const uint64_t* active_counts,
    uint32_t active_prefix_count,
    uint64_t entry_count,
    const rns8_plan* immutable_plan = nullptr);
rns8_status hip_direct_gemm_finite_u8_resident_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_region_device(
    int device_id,
    const void* device_a_residues,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t region_row_offset,
    int64_t region_col_offset,
    int64_t region_rows,
    int64_t region_cols,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_grouped_matrices_device(
    int device_id,
    rns8_matrix* const* a_matrices,
    rns8_matrix* const* b_matrices,
    rns8_matrix* const* c_matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    const void* device_a_residue_ptrs,
    const void* device_b_residue_ptrs,
    const void* device_c_residue_ptrs,
    int64_t m,
    int64_t n,
    int64_t k,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_native_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_native,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_device(
    int device_id,
    const void* device_a_native,
    const void* device_b_residues,
    void* device_c_residues,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint16_t modulus);
rns8_status hip_direct_gemm_finite_u8_native_a_resident_b_matrix(
    int device_id,
    const void* device_a_native,
    const rns8_matrix* B,
    rns8_matrix* C,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t lda,
    uint16_t modulus,
    uint64_t source_version);
rns8_status hip_direct_export_i64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    int64_t* dst,
    int64_t ld);
rns8_status hip_direct_export_i64_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    void* device_status,
    int64_t rows,
    int64_t cols,
    uint64_t bound);
rns8_status hip_direct_export_i64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    const void* zero_a_rows,
    const void* zero_b_cols,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    int64_t* dst,
    int64_t ld);
rns8_status hip_direct_export_u64_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t bound,
    uint64_t* dst,
    int64_t ld);
rns8_status hip_direct_export_u64_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    void* device_status,
    int64_t rows,
    int64_t cols,
    uint64_t bound);
rns8_status hip_direct_export_finite_u8_grouped_matrices_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint16_t modulus);
rns8_status hip_direct_export_u64_tiled_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    const void* device_entries,
    const void* device_bounds,
    const void* zero_a_rows,
    const void* zero_b_cols,
    uint64_t entry_count,
    uint64_t max_tile_elements,
    bool all_zero_output_tiles,
    uint64_t* dst,
    int64_t ld);
rns8_status hip_direct_export_finite_u8_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    int64_t rows,
    int64_t cols,
    uint16_t modulus,
    uint8_t* dst,
    int64_t ld);
rns8_status hip_direct_export_exact_wide_signed_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_signed_tree_crt_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_signed_limbs_to_device(
    int device_id,
    const void* device_residues,
    void* device_dst,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_signed_matrix_limbs_to_device(
    rns8_matrix* matrix,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_prepare_exact_wide_grouped_matrix_residue_pointers(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    void* device_residue_ptrs,
    std::size_t device_residue_ptr_bytes,
    int* out_device_id,
    uint32_t* out_prefix);
rns8_status hip_direct_prepare_grouped_matrix_residue_pointers(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    rns8_semantics expected_semantics,
    void* device_residue_ptrs,
    std::size_t device_residue_ptr_bytes,
    int* out_device_id,
    uint32_t* out_prefix);
rns8_status hip_direct_export_exact_wide_signed_grouped_matrix_limbs_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_tree_crt_limbs_device(
    int device_id,
    const void* device_residues,
    void** export_buffer,
    std::size_t* export_bytes,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint64_t* dst,
    int64_t ld,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_limbs_to_device(
    int device_id,
    const void* device_residues,
    void* device_dst,
    void** status_buffer,
    std::size_t* status_bytes,
    int64_t rows,
    int64_t cols,
    uint32_t prefix,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_matrix_limbs_to_device(
    rns8_matrix* matrix,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_export_exact_wide_unsigned_grouped_matrix_limbs_to_device(
    rns8_matrix* const* matrices,
    uint32_t task_count,
    const void* device_residue_ptrs,
    void* device_dst,
    int64_t rows,
    int64_t cols,
    uint32_t limb_count);
rns8_status hip_direct_synchronize(int device_id);

// Private strict wrap64 research candidates: signed carry-recode or native U8.
// Compact canonical byte inputs/output; no public dispatch or AUTO promotion.
rns8_status hip_direct_wrap_radix_wmma_device(int device_id,const void* a,const void* b,void* c,
    int64_t m,int64_t n,int64_t k,bool signed_digits,uint64_t max_scratch_bytes,bool& commit_started);

rns8_status hip_direct_gram_device(int device_id,const void* x,void* c,const GramProblem& problem,
    uint64_t source_instance,uint64_t source_generation,bool wmma,uint64_t budget,bool& commit_started);

rns8_status hip_direct_gram_prepared_device(int device_id,const void* x,void* c,const GramProblem& problem,
    const GramTile* host_jobs,void* device_jobs,uint64_t device_job_bytes,bool wmma,
    bool& jobs_ready,bool& commit_started);
// Private fixed-transform modes: 0 signed native import, 1 unsigned import,
// 2 canonical three-limb CRT; modes 3/4/5 are corresponding unsigned-byte
// WMMA transforms. Explicit RDNA3 research candidates only.
rns8_status hip_direct_fixed_conversion_device(int device_id,const void* source,void* destination,
    int64_t cells,uint32_t prefix,uint32_t mode,uint64_t budget,bool& commit_started);

namespace word_mod {struct Problem;struct FloatingRequest;}
rns8_status hip_direct_word_mod_device(int device_id,const void* a,const void* b,void* c,
    const word_mod::Problem& problem,uint64_t budget,bool& commit_started,bool packed_dot4=false,
    const word_mod::FloatingRequest* floating=nullptr,bool byte_wmma=false,bool canonical_dot4=false);

rns8_status hip_direct_native_view_bound(int device_id,const void* source,void* scratch,
    int64_t rows,int64_t cols,int64_t ld,bool signed_values,uint64_t& bound);
rns8_status hip_direct_copy_native_view(int device_id,const void* source,void* destination,
    int64_t rows,int64_t cols,int64_t ld);
rns8_status hip_direct_validate_native_allocation(int device_id,const void* base,uint64_t bytes);
rns8_status hip_direct_enqueue_native_import(int device_id,const void* source,void* residues,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,bool signed_values,bool verify,uint64_t bound,
    void* retained,void* scratch,uint64_t scratch_bytes,void* pinned_status,void* stream);
rns8_status hip_direct_enqueue_native_export(int device_id,const void* residues,void* destination,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,bool signed_values,uint32_t limbs,uint64_t bound,
    void* scratch,uint64_t scratch_bytes,void* pinned_status,void* stream);
rns8_status hip_direct_import_native_view(int device_id,const void* source,void* residues,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,bool signed_values,void* retained_native=nullptr);
rns8_status hip_direct_export_native_view(int device_id,const void* residues,void* destination,void* scratch,
    int64_t rows,int64_t cols,int64_t ld,uint32_t prefix,uint64_t bound,bool signed_values,bool& external_write_started,uint32_t limb_count=0);

struct InstructionRoofProblem;
// Borrowed calibration buffers: a/b contain chains signed bytes, output contains
// waves*chains*256 int32 cells, capability is four bytes. On any failure the
// caller must drain before releasing borrowed storage; no routing is affected.
rns8_status hip_direct_instruction_roof(int device_id,const void* a,const void* b,void* output,
    void* capability,const InstructionRoofProblem& problem);

}  // namespace rns8::detail

#endif
