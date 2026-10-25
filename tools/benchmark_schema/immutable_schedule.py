"""Validate source-work snapshots without treating them as timing evidence."""
FIELDS=('core_exhaustive_validations','fingerprint_traversals','canonical_equivalent_comparisons',
        'backend_schedule_seals','backend_sealed_tile_visits','backend_fast_submissions','backend_raw_submissions')
def validate_immutable_schedule(capture):
    meta=capture.get('immutable_schedule_work')
    if meta is None:return []
    if not isinstance(meta,dict):return ['immutable_schedule_work must be an object']
    errors=[]
    if type(meta.get('abi_version')) is not int or meta['abi_version']!=1:errors.append('unknown immutable schedule counter ABI')
    if meta.get('scope')!='calling_thread_source_work_counters_not_latency':errors.append('invalid immutable schedule counter scope')
    previous=None
    for label in ('after_warmups','after_measured_repeats'):
        values=meta.get(label)
        if not isinstance(values,dict) or set(values)!=set(FIELDS) or any(type(values[k]) is not int or values[k]<0 for k in FIELDS):
            errors.append('invalid immutable schedule counter snapshot');continue
        if previous and any(values[k]<previous[k] for k in FIELDS):errors.append('immutable schedule counters decreased')
        previous=values
    return errors
