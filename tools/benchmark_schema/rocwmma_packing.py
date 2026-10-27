"""Explicit rocWMMA source-work receipts; counters do not establish speedup."""
FIELDS=('a_pack_launches','b_pack_launches','direct_a_blocks','a_panel_hits','b_panel_hits')
POLICIES=('packed','direct-a','reuse','direct-a-reuse')


def packing_policy(capture):
    if capture.get('backend_selected')!='rocwmma':return None
    meta=capture.get('rocwmma_packing')
    return meta.get('policy') if isinstance(meta,dict) else 'packed'


def validate_rocwmma_packing(capture):
    meta=capture.get('rocwmma_packing')
    if meta is None:return []
    if not isinstance(meta,dict):return ['rocWMMA packing must be an object']
    errors=[];policy=meta.get('policy')
    if type(meta.get('abi_version')) is not int or meta['abi_version']!=1:errors.append('unknown rocWMMA packing ABI')
    if policy not in POLICIES:errors.append('unknown rocWMMA packing policy')
    if meta.get('scope')!='calling_thread_successful_submissions_not_bytes_or_time':errors.append('invalid rocWMMA packing scope')
    previous=None
    for label in ('after_warmups','after_measured_repeats'):
        snapshot=meta.get(label)
        if not isinstance(snapshot,dict) or set(snapshot)!=set(FIELDS) or any(type(snapshot[k]) is not int or snapshot[k]<0 for k in FIELDS):
            errors.append('invalid rocWMMA packing snapshot');continue
        if previous and any(snapshot[k]<previous[k] for k in FIELDS):errors.append('rocWMMA packing counters decreased')
        previous=snapshot
        if policy not in ('direct-a','direct-a-reuse') and snapshot['direct_a_blocks']:errors.append('disabled direct-A path recorded work')
        if policy not in ('reuse','direct-a-reuse') and (snapshot['a_panel_hits'] or snapshot['b_panel_hits']):errors.append('disabled panel reuse recorded hits')
        if capture.get('backend_selected')!='rocwmma' and any(snapshot.values()):errors.append('non-rocWMMA capture recorded rocWMMA work')
    return errors
