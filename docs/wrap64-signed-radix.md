# Carry-recoded wrap64 matrix candidate

The private direct-HIP entry `hip_direct_wrap_radix_wmma_device` admits compact
canonical byte-limb inputs/output for strict modulo 2^64 multiplication only.
It has two explicit candidates: signed carry-recode and native unsigned IU8.
Neither changes public dispatch, storage, or AUTO selection.

The signed view computes `(x + H) ^ H`, H=0x8080808080808080, once per loaded
cell per 16x16x16 tile. Unsigned addition propagates carries before bytes become
signed digits. Its weighted digits sum to x modulo 2^64. Exactly 36 pairs with
i+j<8 remain; higher pairs vanish in the ring. Both lanes in each half-wave
load the replicated RDNA3 fragment mapping. Each matrix instruction starts from
zero, uses 16 padded K terms, and disables clamp. The signed bound is 16*16384;
the unsigned control bound is 16*65025. Each exact INT32 result converts through
signed64 to unsigned64 before shifting by 8*(i+j); accumulation wraps only in
the intended unsigned ring. No INT32 diagonal sums are formed.

The implementation stages 256 full recoded A and B cells in 4096 shared bytes,
reusing them across all pairs. Input/output cell byte numbering is explicitly
little-endian through shifts; no alignment-sensitive native loads or signed
out-of-range casts are needed. Row/column/K tails are zero-masked, including
recode(0)=0. Output stores reconstruct canonical low64 bytes. The native unsigned
control uses identical tiles, staging, loop and output boundaries without recode.

Admission checks complete compact spans and rejects output overlap with either
input. Read-only A/B alias is legal. A four-byte device capability allocation
and readback precede output writes. Only gfx1100/1101/1102 with the actual IU8
builtin are admitted; unsupported targets cannot report successful computation.
The synchronous private call reports whether output writes may have begun;
a failed issued launch invalidates the output. No persistent recoded cache or
borrowed-stream contract is introduced.

`tools/data/wrap64_signed_radix_certificate.json` names the exact arithmetic
schedule and layout version. `arithmetic_certificate.py` checks the ring
construction, omitted terms, exact pair bounds and unsigned folding independently
of C++ arithmetic. The certificate never grants runtime dispatch eligibility.
Host tests cover shifted exhaustive 16-bit recodes, full-width adversarial and
random products, independent byte/U64/Boost oracles and broken recodes. GPU
fixtures cover tails, canaries, aliases, budgets and K around tile, 4096,
33025, 65536 and 131071 boundaries. Tests and compilation remain pending.

This first implementation deliberately folds after every 16-term MMA. It does
not claim a 65536-term accumulation optimization, ISA qualification, or a speedup.
The longer mathematical signed cap of 131071 and unsigned cap of 33025 are
certificate bounds, not current schedule lengths. Matched comparison against
direct-HIP v4 and corrected rocWMMA, ISA/resource receipts, and sustained
end-to-end benchmark evidence remain required before any measured integration.

The authored [matched calibration driver](wrap64-calibration.md) supplies
resident, roundtrip and cold-input controls with common canonical byte limbs,
hook-disabled production timing and complete outside references. Its long-K
cases explicitly omit rocWMMA above that existing control's K32768 cap.
The new driver, collector, case sets and fixtures have not run.
