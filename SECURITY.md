# Security Policy

RNS8 is pre-1.0 research software and is not yet a production-hardened
cryptographic or distributed-compute component.

## Reporting

Please report security issues privately through GitHub's vulnerability reporting
flow for this repository when available. If that is unavailable, open a minimal
issue that does not include exploit details and ask for a private contact path.

## Scope

Security-relevant issues include:

- Memory safety bugs in public C/C++ APIs.
- Host/device buffer size overflows.
- Incorrect validation that could silently change exact arithmetic semantics.
- Unsafe handling of cache, environment, or generated artifact paths.

Performance-only regressions and unsupported-backend failures should use the
normal issue templates.

## Support Window

Only the current default branch and active release PR stack are in scope before
1.0. There are no long-term support branches yet.
