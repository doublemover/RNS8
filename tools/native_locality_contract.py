"""Independent logical queue permutation/accounting; never physical placement."""
ORDERS = ("ordinary", "output-neighbor", "modulus", "same-output")


def model(tasks, order, target, unique_source_bytes):
    target_id = {"gfx942": 942, "gfx950": 950, "gfx1100": 1100}.get(target.split(":")[0], 0)
    applied = bool(target_id and order != "ordinary")
    per_task = [[(i, row, col, first) for row in range((t["m"]+15)//16)
                 for col in range((t["n"]+15)//16) for first in range(0, t["prefix"], t["group"])]
                for i, t in enumerate(tasks)]
    work = [items[q] for q in range(max(map(len, per_task))) for items in per_task if q < len(items)]
    def extents(w):
        i,r,c,p=w;t=tasks[i]
        return min(16,t["m"]-16*r),min(16,t["n"]-16*c),min(t["group"],t["prefix"]-p)
    def key(w):
        i,r,c,p=w;t=tasks[i];rows,cols,planes=extents(w)
        bucket=(bool(t.get("encoded_b")),int(t["unsigned_inputs"]),t["group"],planes,rows,cols,t["k"])
        axes={"output-neighbor":(r,p,c,i),"modulus":(p,r,c,i),"same-output":(r,c,p,i)}[order]
        return bucket,axes
    if applied:work.sort(key=key)
    requested=outputs=0
    for w in work:
        t=tasks[w[0]];rows,cols,planes=extents(w)
        requested+=rows*t["k"]*8+cols*(((t["k"]+15)//16)*16*planes if t.get("encoded_b") else t["k"]*8)
        outputs+=rows*cols*planes
    fingerprint=1469598103934665603
    for word in [target_id,ORDERS.index(order) if applied else 0,1,*(x for w in work for x in w)]:
        fingerprint=((fingerprint^word)*1099511628211)&((1<<64)-1)
    return dict(version=1,requested_order=order,applied_order=ORDERS.index(order) if applied else 0,
                applied=applied,target_id=target_id,topology_version=1,physical_affinity_established=False,
                active_entries=len(work),directory_bytes=len(work)*16,requested_source_bytes=requested,
                unique_source_bytes=unique_source_bytes,partial_output_bytes=outputs,permutation_fingerprint=fingerprint or 1)


def private_contract(case,target):
    fragments=case["fragments"] if case["candidate"]=="split_k" else 1
    tasks=[dict(case,k=case["k"]*(f+1)//fragments-case["k"]*f//fragments) for f in range(fragments)]
    result=model(tasks,case["locality_order"],target,(case["m"]+case["n"])*case["k"]*8)
    return dict(result,representation="raw_native64",modulus_group=case["group"],fragments=fragments)


def public_contract(case,target):
    tasks=[dict(t,encoded_b=case["encoded_b"]) for t in case["tasks"]]
    bs={t["b_slot"]:t for t in tasks}
    unique=sum(t["m"]*t["k"]*8 for t in tasks)+sum(t["n"]*((t["k"]+15)//16)*16*t["prefix"] if case["encoded_b"] else t["n"]*t["k"]*8 for t in bs.values())
    return model(tasks,case["locality_order"],target,unique)
