# Composable Kernel Backend

Reserved for future CK grouped GEMM and fused epilogue experiments.

This directory intentionally contains no backend implementation yet. CK must
remain feature-detected and optional, with exact CPU differential validation for
any enabled path.

`RNS8_ENABLE_CK` must keep failing fast until a real CK correctness backend
exists. Header discovery, CMake discovery, and optional compile probes are
evidence only; they must not compile placeholder backend code or satisfy
correctness.
