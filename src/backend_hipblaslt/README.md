# hipBLASLt Backend

Reserved for the future feature-detected hipBLASLt accelerator path.

No hipBLASLt code is implemented in the current Phase 0/1 scaffold. Adding code
here must not make hipBLASLt a correctness requirement, and every accepted path
must have CPU differential coverage before performance claims.

`RNS8_ENABLE_HIPBLASLT` must keep failing fast until a real correctness backend
exists. Dependency checks and probe presets may collect hipBLASLt component
evidence, but they must not compile a placeholder backend or satisfy
correctness.
