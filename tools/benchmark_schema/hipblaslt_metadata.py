"""Host library API diagnostics; no inference about allocator bytes or elapsed time."""
FIELDS=('descriptor_create_calls','descriptor_destroy_calls','attribute_set_calls',
        'preference_create_calls','preference_destroy_calls','preference_attribute_calls',
        'heuristic_calls','algorithm_cache_hits','descriptor_set_hits','descriptor_sets_created',
        'matmul_block_calls','maximum_live_sets')
PREPACK_FIELDS=('a_retained_builds','b_retained_builds','a_retained_hits','b_retained_hits',
    'a_retained_planes_packed','b_retained_planes_packed','a_transient_planes_packed','b_transient_planes_packed')


def validate_hipblaslt_prepack(capture):
    meta=capture.get('hipblaslt_operand_prepack')
    if meta is None:return []
    if not isinstance(meta,dict):return ['hipBLASLt operand prepack must be an object']
    errors=[];policy=meta.get('policy')
    if type(meta.get('abi_version')) is not int or meta['abi_version']!=1:errors.append('unknown hipBLASLt operand prepack ABI')
    if policy not in ('none','a','b','both'):errors.append('unknown hipBLASLt operand prepack policy')
    if meta.get('scope')!='calling_thread_successful_pack_submissions_and_cache_matches_not_bytes_or_time':errors.append('incorrect hipBLASLt operand prepack scope')
    previous=None
    for label in ('after_warmups','after_measured_repeats'):
        snapshot=meta.get(label)
        if not isinstance(snapshot,dict) or set(snapshot)!=set(PREPACK_FIELDS) or any(type(snapshot[k]) is not int or snapshot[k]<0 for k in PREPACK_FIELDS):
            errors.append('invalid hipBLASLt operand prepack snapshot');continue
        if previous and any(snapshot[k]<previous[k] for k in PREPACK_FIELDS):errors.append('hipBLASLt operand prepack counters decreased')
        previous=snapshot
        for operand in ('a','b'):
            retained=[snapshot[operand+suffix] for suffix in ('_retained_builds','_retained_hits','_retained_planes_packed')]
            if policy not in (operand,'both') and any(retained):errors.append('disabled hipBLASLt operand was retained')
            if snapshot[operand+'_retained_planes_packed']<snapshot[operand+'_retained_builds']:errors.append('hipBLASLt retained build lacks packed planes')
        if capture.get('backend_selected')!='hipblaslt' and any(snapshot.values()):errors.append('non-hipBLASLt capture recorded hipBLASLt pack work')
    return errors


def validate_hipblaslt_metadata(capture):
    meta=capture.get('hipblaslt_metadata')
    if meta is None:return []  # historical captures remain readable
    if not isinstance(meta,dict):return ['hipblaslt_metadata must be an object']
    errors=[]
    if meta.get('abi_version')!=1:errors.append('unknown hipBLASLt metadata ABI')
    policy=meta.get('policy')
    if policy not in ('per_block_control','per_call_distinct_geometry'):errors.append('unknown hipBLASLt metadata policy')
    if meta.get('scope')!='calling_thread_library_API_attempt_counts_not_allocator_bytes_or_device_work':errors.append('incorrect hipBLASLt counter scope')
    if meta.get('cross_call_raw_descriptor_retention') is not False:errors.append('per-call metadata does not retain descriptors across calls')
    previous=None
    for label in ('after_warmups','after_measured_repeats'):
        snapshot=meta.get(label)
        if not isinstance(snapshot,dict) or any(type(snapshot.get(k)) is not int or snapshot[k]<0 for k in FIELDS):
            errors.append('invalid hipBLASLt metadata counter snapshot');continue
        if previous and any(snapshot[k]<previous[k] for k in FIELDS):errors.append('hipBLASLt counters decreased inside one process')
        previous=snapshot
        # Completed successful capture: no partial construction or live descriptors.
        if snapshot['descriptor_create_calls']!=5*snapshot['descriptor_sets_created'] or snapshot['descriptor_destroy_calls']!=snapshot['descriptor_create_calls']:errors.append('hipBLASLt descriptor lifecycle accounting mismatch')
        if snapshot['attribute_set_calls']!=2*snapshot['descriptor_sets_created']:errors.append('hipBLASLt layout attribute accounting mismatch')
        if snapshot['preference_create_calls']!=snapshot['preference_destroy_calls'] or snapshot['preference_attribute_calls']!=snapshot['preference_create_calls']:errors.append('hipBLASLt preference lifecycle accounting mismatch')
        if snapshot['algorithm_cache_hits']+snapshot['heuristic_calls']!=snapshot['matmul_block_calls']:errors.append('hipBLASLt algorithm lookup accounting mismatch')
        if policy=='per_call_distinct_geometry':
            if snapshot['maximum_live_sets']>2:errors.append('hipBLASLt call metadata exceeded full/tail bound')
            if snapshot['preference_create_calls']!=snapshot['heuristic_calls']:errors.append('preference constructed without heuristic miss')
            if snapshot['descriptor_set_hits']+snapshot['descriptor_sets_created']!=snapshot['matmul_block_calls']:errors.append('hipBLASLt geometry reuse accounting mismatch')
        elif policy=='per_block_control':
            if snapshot['descriptor_sets_created']!=snapshot['matmul_block_calls'] or snapshot['preference_create_calls']!=snapshot['matmul_block_calls']:errors.append('per-block control no longer matches old API work')
    return errors
