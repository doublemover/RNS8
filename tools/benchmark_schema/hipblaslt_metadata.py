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



def validate_hipblaslt_algorithms(capture):
    inventory = capture.get("hipblaslt_algorithms")
    if inventory is None:
        return []  # Earlier captures have no selected-algorithm evidence.
    error = "invalid hipBLASLt selected-algorithm observations"
    if (not isinstance(inventory, dict) or type(inventory.get("abi_version")) is not int or inventory.get("abi_version") != 1 or
            inventory.get("scope") != "library_build_device_padded_geometry_algorithm_index" or
            inventory.get("query_policy") != "opt_in_once_per_cached_algorithm"):
        return [error]
    counters = capture.get("hipblaslt_metadata")
    if not isinstance(counters, dict):
        return [error]
    geometry_fields = ("device_id", "m", "n", "k", "scratch_ld", "workspace_bytes", "attribute_abi")
    previous = None
    previous_queries = 0
    previous_available = None
    previous_complete = True
    for phase in ("after_warmups", "after_measured_repeats"):
        snapshot = inventory.get(phase)
        count_snapshot = counters.get(phase)
        if not isinstance(snapshot, dict) or not isinstance(count_snapshot, dict):
            return [error]
        available, complete = snapshot.get("available"), snapshot.get("complete")
        entries, queries = snapshot.get("entries"), snapshot.get("index_queries")
        calls = count_snapshot.get("matmul_block_calls")
        if (type(available) is not bool or type(complete) is not bool or
                type(queries) is not int or queries < 0 or type(calls) is not int or calls < 0 or
                queries > calls or queries < previous_queries or
                (previous_available is not None and available != previous_available) or
                (not previous_complete and complete) or not isinstance(entries, list) or len(entries) > 16 or
                (complete and not available) or (not available and queries)):
            return [error]
        observed = {}
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != set(geometry_fields) | {"algorithm_index", "matmul_calls"} or
                    any(type(entry.get(key)) is not int for key in geometry_fields) or
                    not 0 <= entry["device_id"] <= 2147483647 or not 0 <= entry["workspace_bytes"] <= 18446744073709551615 or entry["attribute_abi"] != 1 or
                    any(not 0 < entry[key] <= 2147483647 or entry[key] % 16 for key in ("m", "n", "k")) or
                    entry["scratch_ld"] != entry["n"] or
                    type(entry.get("matmul_calls")) is not int or entry["matmul_calls"] <= 0):
                return [error]
            index = entry.get("algorithm_index")
            if index is not None and (type(index) is not int or not 0 <= index <= 2147483647):
                return [error]
            if (complete and index is None) or (not available and index is not None):
                return [error]
            identity = tuple(entry[key] for key in geometry_fields) + (index,)
            if identity in observed:
                return [error]
            observed[identity] = entry["matmul_calls"]
        total = sum(observed.values())
        if total > calls or (complete and total != calls):
            return [error]
        if capture.get("backend_selected") != "hipblaslt" and (total or queries):
            return [error]
        if previous is not None and any(observed.get(key, 0) < value for key, value in previous.items()):
            return [error]
        previous = observed
        previous_queries, previous_available, previous_complete = queries, available, complete
    return []



def hipblaslt_algorithm_signature(capture):
    """Physical algorithm/work signature; input/timing/artifact admission is separate."""
    if capture.get("backend_selected") != "hipblaslt" or validate_hipblaslt_algorithms(capture):
        return None
    owner = capture.get("hipblaslt_algorithms")
    if not isinstance(owner, dict):
        return None
    inventory = owner.get("after_measured_repeats", {})
    if inventory.get("available") is not True or inventory.get("complete") is not True or not inventory.get("entries"):
        return None
    manifest = capture.get("build_manifest", {})
    metadata = capture.get("backend_metadata", {})
    device = capture.get("device", {})
    if not all(isinstance(value, dict) for value in (manifest, metadata, device)):
        return None
    scope = (manifest.get("source_digest"), manifest.get("dependency_digest"), manifest.get("configuration"),
             metadata.get("accelerator_version"), device.get("target_cache_key"), device.get("target_instance_id"))
    if not all(isinstance(value, str) and value and value not in ("unknown", "unavailable") for value in scope):
        return None
    numeric = (device.get("device_id"), device.get("hip_runtime_version"), device.get("hip_driver_version"))
    if any(type(value) is not int or value < (0 if i == 0 else 1) for i, value in enumerate(numeric)):
        return None
    fields = ("device_id", "m", "n", "k", "scratch_ld", "workspace_bytes", "attribute_abi", "algorithm_index", "matmul_calls")
    return scope + numeric + (tuple(sorted(tuple(entry[key] for key in fields) for entry in inventory["entries"])),)
