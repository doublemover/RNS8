"""Exact toy-domain basis/form/group/slab oracle with explicit resource costs.

Costs are supplied predictions in integer picoseconds, never hardware evidence.
The objective is serial grouped execution plus ordered prefix reconstruction.
No pruning based on a form's isolated latency is valid for this joint objective.
"""
from dataclasses import dataclass
from itertools import combinations, permutations, product
import math

from arithmetic_certificate import certificate_hash
from modulus_scheme_optimizer import Form, optimize


@dataclass(frozen=True)
class ScheduledForm:
    form: Form
    layout: str
    registers: int
    lds_bytes: int

    def signature(self):
        c = self.form.certificate
        # Conservative compatibility: exact opcode, target, compiler, operand
        # alphabet and fragment layout must match; K caps may differ.
        return (c.instruction, self.layout, len(c.planes), c.denominator)


def arithmetic_limits(c):
    """Separate dot and ordered weighted-epilogue caps, using integer intervals."""
    def cap(lo, hi, prior_lo, prior_hi, minimum, maximum):
        if prior_lo < minimum or prior_hi > maximum:
            return 0
        limits = []
        if lo < 0:
            limits.append((prior_lo - minimum) // -lo)
        if hi > 0:
            limits.append((maximum - prior_hi) // hi)
        return min(limits) if limits else None

    dot, epilogue = [], []
    running_lo = running_hi = 0
    for p in c.planes:
        products = [x * y for x in (min(p.lhs), max(p.lhs)) for y in (min(p.rhs), max(p.rhs))]
        lo, hi = min(products), max(products)
        dot.append(cap(lo, hi, c.prior_min, c.prior_max,
                       c.instruction.accumulator_min, c.instruction.accumulator_max))
        weighted_lo, weighted_hi = sorted((lo * p.weight, hi * p.weight))
        epilogue.append(cap(weighted_lo, weighted_hi, 0, 0, c.recombination_min, c.recombination_max))
        running_lo += weighted_lo
        running_hi += weighted_hi
        epilogue.append(cap(running_lo, running_hi, c.prior_min, c.prior_max,
                            c.recombination_min, c.recombination_max))
    def finite_min(items):
        return min((x for x in items if x is not None), default=None)
    dot_cap, ep_cap = finite_min(dot), finite_min(epilogue)
    return {"dot_accumulation_cap": dot_cap, "epilogue_composition_cap": ep_cap,
            "certified_slab_cap": min(x for x in (dot_cap, ep_cap, c.safe_k) if x is not None)}


def partitions(ids):
    """Unique set partitions, with stable physical plane IDs independent of order."""
    if not ids:
        yield ()
        return
    first, *rest = ids
    for tail in partitions(rest):
        yield ((first,),) + tail
        for i in range(len(tail)):
            yield tail[:i] + ((first,) + tail[i],) + tail[i + 1:]


def joint_optimize(forms, bound, signed, k, *, slabs, max_group_planes,
                   register_budget, lds_budget, group_launch_ps,
                   prefix_bit_ps, max_nodes=1000000, allowed_form_ids=None):
    """Enumerate all legal ordered bases, partitions and declared slab choices.

    prefix_bit_ps is the cost of reconstructing each cumulative prefix bit.
    Group compute costs sum per-channel work; only launches are shared. Real
    throughput/overlap requires a richer calibrated model, not an assumed win.
    allowed_form_ids can model a staged shortlist; run the unrestricted oracle
    separately to quantify lost optima from that shortlist.
    """
    if not 1 <= len(forms) <= 8:
        raise ValueError("joint oracle is bounded to 1..8 supplied forms")
    optimize([x.form for x in forms], bound, signed, k, max_nodes=1)
    integers = (max_group_planes, register_budget, lds_budget, group_launch_ps, prefix_bit_ps, max_nodes)
    if (any(type(x) is not int or x < 0 for x in integers) or not max_group_planes or not max_nodes
            or not slabs or len(set(slabs)) != len(slabs)
            or any(type(x) is not int or x <= 0 for x in slabs)):
        raise ValueError("invalid grouping, resource, cost, slab or search domain")
    for f in forms:
        if (not isinstance(f.layout, str) or not f.layout
                or any(type(x) is not int or x < 0 for x in (f.registers, f.lds_bytes))):
            raise ValueError("missing layout or resource vector")
    ids = tuple(range(len(forms))) if allowed_form_ids is None else tuple(allowed_form_ids)
    if len(set(ids)) != len(ids) or any(type(x) is not int or x not in range(len(forms)) for x in ids):
        raise ValueError("invalid staged shortlist")
    limits = [arithmetic_limits(f.form.certificate) for f in forms]
    hashes = [certificate_hash(f.form.certificate) for f in forms]
    threshold = bound * (2 if signed else 1)
    best = None
    visited = 0
    exhausted = False
    rejected = {"insufficient_range": 0, "noncoprime": 0, "incompatible_or_resource": 0,
                "no_certified_slab": 0}

    def consume():
        nonlocal visited, exhausted
        if visited == max_nodes:
            exhausted = True
            return False
        visited += 1
        return True

    for count in range(1, len(ids) + 1):
        for subset in combinations(ids, count):
            if not consume():
                break
            moduli = [forms[i].form.certificate.modulus for i in subset]
            if any(math.gcd(a, b) != 1 for a, b in combinations(moduli, 2)):
                rejected["noncoprime"] += 1
                continue
            if math.prod(moduli) <= threshold:
                rejected["insufficient_range"] += 1
                continue
            # Cost of logical basis order depends on cumulative prefix widths.
            # Physical group partition never changes these plane identities.
            orders = []
            for order in permutations(subset):
                if not consume():
                    break
                prefix, widths, products = 1, 0, []
                for i in order:
                    prefix *= forms[i].form.certificate.modulus
                    widths += prefix.bit_length()
                    products.append(prefix)
                orders.append((widths * prefix_bit_ps, order, products))
            if exhausted:
                break
            # Scheduling costs are explicitly independent of logical order in
            # this declared model, so retain the minimum exact order cost.
            order_cost, order, products = min(orders)
            for groups in partitions(list(subset)):
                if not consume():
                    break
                if any(len({forms[i].signature() for i in g}) != 1
                       or sum(len(forms[i].form.certificate.planes) for i in g) > max_group_planes
                       or sum(forms[i].registers for i in g) > register_budget
                       or sum(forms[i].lds_bytes for i in g) > lds_budget for g in groups):
                    rejected["incompatible_or_resource"] += 1
                    continue
                choices = [tuple(s for s in slabs if s <= min(limits[i]["certified_slab_cap"] for i in g))
                           for g in groups]
                if any(not x for x in choices):
                    rejected["no_certified_slab"] += 1
                    continue
                for selected_slabs in product(*choices):
                    if not consume():
                        break
                    cost = order_cost + sum(forms[i].form.conversion_ps + forms[i].form.reconstruction_ps
                                            for i in subset)
                    schedule = []
                    for group, slab in zip(groups, selected_slabs):
                        repeats = (k + slab - 1) // slab
                        cost += repeats * (group_launch_ps + sum(forms[i].form.slab_ps + forms[i].form.epilogue_ps
                                                                 for i in group))
                        schedule.append({"plane_ids": [order.index(i) for i in group],
                                         "form_ids": list(group), "slab": slab, "slab_count": repeats})
                    if best is None or cost < best["cost_ps"]:
                        best = {"cost_ps": cost, "ordered_form_ids": list(order),
                                "ordered_moduli": [forms[i].form.certificate.modulus for i in order],
                                "prefix_products": products, "schedule": schedule}
                if exhausted:
                    break
            if exhausted:
                break
        if exhausted:
            break
    return {"schema": "rns8_joint_scheme_oracle_v1", "best": best,
            "objective": "serial_groups_shared_launch_ordered_prefix_bits",
            "cost_cohort": forms[0].form.cost_cohort, "domain_hashes": hashes,
            "model": {"max_group_planes": max_group_planes, "register_budget": register_budget,
                      "lds_budget": lds_budget, "group_launch_ps": group_launch_ps,
                      "prefix_bit_ps": prefix_bit_ps, "k": k, "bound": bound, "signed": signed},
            "forms": [{"certificate_sha256": hashes[i], "layout": f.layout, "registers": f.registers,
                       "lds_bytes": f.lds_bytes, "conversion_ps": f.form.conversion_ps,
                       "slab_ps": f.form.slab_ps, "epilogue_ps": f.form.epilogue_ps,
                       "reconstruction_ps": f.form.reconstruction_ps} for i, f in enumerate(forms)],
            "allowed_form_ids": list(ids), "form_limits": limits, "slab_domain": list(slabs),
            "strict_product_threshold": threshold, "visited_nodes": visited,
            "search_complete": not exhausted, "optimal_within_declared_domain": not exhausted and best is not None,
            "lower_bound_ps": best["cost_ps"] if best is not None and not exhausted else 0,
            "rejections": rejected, "promotion_eligible": False, "runtime_ladder_changed": False}
