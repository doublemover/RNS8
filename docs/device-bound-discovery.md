# Native device bound discovery

The private synchronous direct-HIP producer supports explicit `tile_max` and
`axis_l1` methods for 16 by 16 output tiles. The original method scans the maximum
absolute A and B values in each row/column tile and uses K times their product.
The L1 method scans each logical row of A and column of B, retaining the largest
absolute value and largest exact absolute sum in each axis tile. It admits the
minimum of K*amax*bmax, amax*max_column_L1(B), and max_row_L1(A)*bmax. Each term is
independently conservative for every cell in the output tile.

Sums use two unsigned limbs: K <= UINT32_MAX gives a sum below 2^96. A sum times
a native magnitude is below 2^160 and uses three limbs. Signed magnitudes handle
INT64_MIN by unsigned subtraction. Prefix selection uses strict signed/unsigned
uniqueness, with zero and unadmitted status distinct. There is no saturation that
could falsely suggest a smaller bound. Native arrays are read once per selected
axis scan; L1 summaries occupy 24 bytes per axis tile versus eight for maxima.

The caller explicitly selects the method and supplies its scratch budget. No
benefit-based default selection is claimed. Scan, prefix selection and device
bucket compaction have distinct timing labels. This standalone diagnostic retains
its synchronous rejected-count readback. The composed
[adaptive native consumer](adaptive-native-product.md) now uses device offsets for
its optional compact worker traversal, completes missing planes exactly and
commits full residue output. Its explicit public retained-native owner binds the
input materialization to matrix generations; raw discovery records alone remain
usable only while exact inputs are unchanged. Reusable summary caches and general
planner-selected consumers are still separate integration work.

Authored Boost checks cover full-width sums/products, carry boundaries through
UINT32_MAX K, signed minima, unsigned maxima, zeros, mutation, padded tails,
outliers and both methods' exact bound/prefix/bucket outputs. Registered applicable cases are covered by the
[dated all-feature validation](local-all-features-20261023.md); smaller proved prefixes do not establish a net speedup.

## Explicit K-panel L1 refinement

`panel_l1` partitions K into caller-selected panels of 1..65536 elements (default
256). Each axis-tile/panel keeps the same exact maximum and L1 summary. A tile's
bound is the sum over panels of the minimum of panel_length*amax*bmax,
amax*B_column_L1 and A_row_L1*bmax. Every term independently bounds every cell
on that panel; summing disjoint panels is conservative by the triangle inequality.
Three limbs preserve all carries: total K remains at most UINT32_MAX, so the
native product bound stays below 2^160. All-zero products from disjoint nonzero
panels can therefore be proved without forming an M-by-N magnitude matrix.

The producer scans native inputs once, stores 24 bytes per axis tile per panel,
and performs one bounded panel loop per output tile. Scratch admission charges
all summaries and compaction arrays. At most 65535 panels are admitted by this
launch layout; larger requested grids report unsupported. `native_bound_panel_l1_scan`
labels the new scan, while selection/compaction retain separate timing labels.
Required prefixes and zero buckets use the existing strict uniqueness test.

This policy need not dominate whole-axis L1: maximizing different rows within
each panel can lose cross-panel information. The caller chooses the route; no
cost-based default or net speedup is asserted. Authored CPU multiword and GPU
Boost fixtures include signed minima/unsigned maxima, exact carries, partial
panels, disjoint K support, mutation, canaries and invalid panel/budget rejection.
The extension and test executable compiled with VS2026/HIP 7.1 for gfx1100.
The dated record owns applicable tests at its named source/configuration. New
compiler/source changes await the consolidated campaign; per-method performance
and broader target qualification remain separate gates.
