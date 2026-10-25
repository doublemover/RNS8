"""Independent tensor, liveness, sign and source-drift regressions; not run during development."""
import copy
from rank23_october55 import pinned, parse, verify, header, HEADER, allocate, evaluate, expand


def main():
    raw=pinned(); plans=parse(raw)
    assert header(raw,plans)==HEADER.read_text()
    proof=verify(plans)
    assert proof['additions']==[13,14,28] and proof['tensor_coefficients']==729
    for key in plans:
        bad=copy.deepcopy(plans);bad[key]['nodes'][0]['left_sign']*=-1
        try:verify(bad)
        except ValueError:pass
        else:raise AssertionError('altered tensor accepted')
    for modulus in range(2,257):
        schedules={k:allocate(p) for k,p in plans.items()}
        for seed in range(8):
            a=[(i*71+seed*11)%256-128 for i in range(9)]
            b=[(i*19+seed*23)%256-128 for i in range(9)]
            u=evaluate(schedules['U'],a,modulus);v=evaluate(schedules['V'],b,modulus)
            actual=evaluate(schedules['W'],[x*y for x,y in zip(u,v)],modulus)
            for r in range(3):
                for c in range(3):
                    expected=sum(a[r*3+q]*b[q*3+c] for q in range(3))%modulus
                    if expected>=(modulus+1)//2:expected-=modulus
                    assert actual[r*3+c]==expected
    # Ordered tensor expansion, rather than commuting A/B values, also covers
    # arbitrary noncommutative block rings. Explicit 2x2 matrix units witness
    # every ordered A_i*B_j coefficient; reversed factors would differ.
    matrices={key:expand(plan)[0] for key,plan in plans.items()}
    for a in range(9):
        for b in range(9):
            for out in range(9):
                coefficient=sum(matrices['U'][p][a]*matrices['V'][p][b]*matrices['W'][out][p] for p in range(23))
                assert coefficient==int(a//3==out//3 and b%3==out%3 and a%3==b//3)
    return 0


if __name__=='__main__':raise SystemExit(main())
