"""Logical ordering diagnostics must not masquerade as physical cache evidence."""
def validate_locality(capture):
    data=capture.get('locality_schedule')
    if data is None:return []
    if not isinstance(data,dict):return ['locality_schedule must be an object']
    errors=[]
    ints=('version','requested_order','applied_order','target_id','topology_version','modulus_group_size',
          'cache_domains_model','concurrent_groups_per_domain_model','cache_bytes_per_domain_model',
          'active_entries','directory_bytes','k_phases','requested_source_bytes','unique_source_bytes',
          'output_read_write_bytes','permutation_fingerprint')
    if any(type(data.get(k)) is not int or data[k]<0 for k in ints):return ['invalid locality integer metadata']
    if data['version']!=1 or data['topology_version']!=1 or data['modulus_group_size']!=1 or data.get('representation')!='resident_residue8':errors.append('unsupported locality representation/version')
    if data['requested_order'] not in range(4) or data['applied_order'] not in range(4) or data['target_id'] not in (0,942,950,1100):errors.append('invalid locality policy/target')
    if type(data.get('applied')) is not bool:errors.append('locality applied must be boolean')
    elif data['applied']:
        if not data['target_id'] or data['applied_order']==0 or data['applied_order']!=data['requested_order']:errors.append('invalid applied locality map')
    elif data['applied_order']!=0:errors.append('fallback must preserve ordinary order')
    if data.get('physical_affinity_established') is not False or data.get('promotion_eligible') is not False:errors.append('logical map does not establish physical affinity or promotion')
    if data.get('traffic_scope')!='active_arithmetic_only_excludes_pack_zero_fill_export':errors.append('invalid locality traffic scope')
    if data.get('actual_hbm_bytes') is not None or data.get('measured_cache_hit_rate') is not None:errors.append('source-work model cannot supply hardware counters')
    if not 1<=data['cache_domains_model']<=128 or not 1<=data['concurrent_groups_per_domain_model']<=65536 or not data['k_phases'] or not data['permutation_fingerprint']:errors.append('invalid locality model dimensions')
    if data['unique_source_bytes']>data['requested_source_bytes']:errors.append('unique source bytes exceed requested bytes')
    if data['k_phases']!=(capture.get('k',1)-1)//65536+1:errors.append('locality K-phase count mismatch')
    return errors
