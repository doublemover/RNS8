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
bucket compaction have distinct timing labels. The existing synchronous rejected
count readback remains; an indirect consumer is still pending. Raw records are
usable only while the exact inputs remain unchanged; generation-bound public
proof ownership, reusable summaries, K-panel refinement, selected conversion and
matrix-engine consumers remain separate integration work.

Authored Boost checks cover full-width sums/products, carry boundaries through
UINT32_MAX K, signed minima, unsigned maxima, zeros, mutation, padded tails,
outliers and both methods' exact bound/prefix/bucket outputs. Compilation and
execution are pending; smaller proved prefixes do not establish a net speedup.
