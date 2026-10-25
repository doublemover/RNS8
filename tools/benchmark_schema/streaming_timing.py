"""Batch-throughput observations are not independent per-call latency samples."""
import math
from measurement_contract import streaming_executed

UNIT="completed_pipeline_batch_mean_per_call_v2"
POLICY="fresh_pinned_stream_event_setup_separate;full_fill_drain_status_and_final_host_copy_inside_each_batch"


def validate_streaming_observations(capture):
    if not streaming_executed(capture):return []
    overlap=capture["streaming_overlap"]
    if "observation_contract_version" not in overlap:return []  # readable historical capture, unqualified downstream
    errors=[]
    if overlap.get("observation_contract_version")!=2:return ["unknown streaming observation ABI"]
    batches=overlap.get("batches");repeats=capture.get("repeats");warmups=capture.get("warmups")
    if not isinstance(batches,list) or not 3<=len(batches)<=64:return ["streaming needs3..64 actual batch observations"]
    if type(repeats) is not int or not 1<=repeats<=65536:return ["streaming calls per batch outside domain"]
    if type(warmups) is not int or not 0<=warmups<=1024:return ["streaming warmup count outside domain"]
    if overlap.get("warmup_batch_count")!=warmups or overlap.get("warmup_calls_per_batch")!=max(2,repeats):errors.append("both slots must participate in each complete warmup batch")
    timing=capture.get("timing_metadata") or {};raw=capture.get("raw_timings_us") or {}
    if timing.get("host_observation_unit")!=UNIT or timing.get("host_result_kind")!="pipeline_batch_throughput":errors.append("streaming host observation identity mismatch")
    if overlap.get("fill_drain_policy")!=POLICY or overlap.get("per_call_latency_available") is not False:errors.append("streaming completion/latency boundary mismatch")
    if overlap.get("schedule") not in ("serial_dependency_chain","overlapped_three_stream"):errors.append("unknown streaming schedule")
    values=raw.get("end_to_end",[])
    if not isinstance(values,list) or len(values)!=len(batches):errors.append("one end-to-end quotient required per actual batch");values=[]
    def finite(v):return type(v) in (int,float) and math.isfinite(v)
    total=0;next_version=warmups*max(2,repeats)+1
    for i,batch in enumerate(batches):
        if not isinstance(batch,dict):errors.append("invalid batch record");continue
        calls=batch.get("calls");duration=batch.get("duration_us")
        if type(calls) is not int or calls!=repeats or not finite(duration) or duration<=0:errors.append("invalid batch calls/duration");continue
        total+=calls
        if batch.get("first_source_version")!=next_version:errors.append("streaming source versions overlap or skip the declared sequence")
        next_version+=calls
        if not finite(batch.get("setup_us")) or batch["setup_us"]<0:errors.append("missing separate pipeline setup duration")
        if batch.get("device_allocate_calls_inside_batch")!=0 or batch.get("device_allocated_bytes_inside_batch")!=0:errors.append("cold staging allocation inside streaming measurement")
        if i<len(values) and (not finite(values[i]) or not math.isclose(values[i],duration/calls,rel_tol=1e-5,abs_tol=1e-5)):errors.append("batch quotient does not match actual duration/calls")
    if overlap.get("final_exact_reference_passed") is not True:errors.append("missing full final-output oracle receipt")
    if total>1048576 or overlap.get("measured_repeat_count")!=total:errors.append("streaming total work count mismatch")
    for phase in ("pack","rns_gemm","crt_export"):
        if raw.get(phase)!=[] or (timing.get("phase_states") or {}).get(phase)!="unsupported":errors.append("device stage events cannot be copied into host phase samples")
    if batches and isinstance(batches[-1],dict) and finite(batches[-1].get("duration_us")):
        if not finite(overlap.get("batch_wall_us")) or not math.isclose(overlap["batch_wall_us"],batches[-1]["duration_us"],rel_tol=1e-5,abs_tol=1e-5):errors.append("legacy last-batch diagnostic does not match last observation")
    return errors


def raw_host_sample_count(capture,phase,default):
    if not streaming_executed(capture) or capture["streaming_overlap"].get("observation_contract_version")!=2:return default
    if phase=="end_to_end":return len(capture["streaming_overlap"].get("batches",[]))
    if phase in ("pack","pack_a","pack_b","rns_gemm","crt_export"):return 0
    return default


def gpu_stage_sample_count(capture,default):
    if streaming_executed(capture) and capture["streaming_overlap"].get("observation_contract_version")==2:
        value=capture["streaming_overlap"].get("measured_repeat_count")
        return value if type(value) is int else 0
    return default
