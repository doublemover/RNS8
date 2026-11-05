#pragma once

#include <cstdint>
#include <iosfwd>
#include <utility>

namespace rns8::detail::trace {

enum class Phase : uint32_t {
  preparation,
  source_pack,
  proof,
  dispatch,
  reduction,
  export_result,
  completion,
  publication,
  oracle,
  cleanup,
  public_call
};
enum class Role : uint32_t { unclassified, setup, warmup, measured, oracle, teardown };
enum class Outcome : uint32_t { host_scope_closed, completed, failed, abandoned };
enum class OperationNamespace : uint32_t { unclassified, benchmark_iteration, prepared_epoch };
struct Tag {
  uint64_t operation = 0, iteration = 0;
  Role role = Role::unclassified;
  OperationNamespace operation_namespace = OperationNamespace::unclassified;
};
struct Token {
  uint64_t range = 0;
  uint32_t slot = UINT32_MAX;
  bool active = false;
};
struct Parent {
  uint64_t range = 0;
  bool available = false;
};

// The provider is optional and loaded only after explicit process opt-in.
// No profiler headers/types appear in an installed public consumer.
const char* provider_status() noexcept;
void write_snapshot(std::ostream&);

#if defined(RNS8_ENABLE_ROCTX) && RNS8_ENABLE_ROCTX
Token begin(Phase, Tag, Parent parent = {}) noexcept;
void finish(Token&, Outcome) noexcept;
Tag current_tag() noexcept;
Tag exchange_tag(Tag) noexcept;
Parent current_parent() noexcept;
Parent exchange_parent(Parent) noexcept;
uint64_t next_operation() noexcept;

class Scope {
public:
  explicit Scope(Phase phase) noexcept : Scope(phase, current_tag()) {}
  Scope(Phase phase, Tag tag) noexcept
      : token_(begin(phase, tag, current_parent())),
        parent_(token_.active ? exchange_parent({token_.range, true}) : Parent{}),
        previous_(exchange_tag(tag)) {}
  ~Scope() {
    exchange_tag(previous_);
    if (token_.active)
      exchange_parent(parent_);
    finish(token_, Outcome::host_scope_closed);
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

private:
  Token token_;
  Parent parent_;
  Tag previous_;
};

// Start/stop ranges carry their own tag and may finish on another thread.
// Their lifetime is independent of a submitter's nested host stack.
class AsyncRange {
public:
  AsyncRange() noexcept = default;
  ~AsyncRange() { finish(token_, Outcome::abandoned); }
  AsyncRange(const AsyncRange&) = delete;
  AsyncRange& operator=(const AsyncRange&) = delete;
  AsyncRange(AsyncRange&& other) noexcept : token_(std::exchange(other.token_, {})) {}
  AsyncRange& operator=(AsyncRange&& other) noexcept {
    if (this != &other) {
      finish(token_, Outcome::abandoned);
      token_ = std::exchange(other.token_, {});
    }
    return *this;
  }
  void start(Phase phase, Tag tag) noexcept {
    finish(token_, Outcome::abandoned);
    token_ = begin(phase, tag, current_parent());
  }
  void complete(Outcome outcome) noexcept { finish(token_, outcome); }

private:
  Token token_;
};
class IterationScope {
public:
  IterationScope(Role role, uint64_t iteration) noexcept
      : scope_(Phase::public_call,
               {next_operation(), iteration, role, OperationNamespace::benchmark_iteration}) {}

private:
  Scope scope_;
};
#else
class Scope {
public:
  explicit Scope(Phase) noexcept {}
  Scope(Phase, Tag) noexcept {}
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};
class AsyncRange {
public:
  AsyncRange() noexcept = default;
  AsyncRange(const AsyncRange&) = delete;
  AsyncRange& operator=(const AsyncRange&) = delete;
  AsyncRange(AsyncRange&&) noexcept = default;
  AsyncRange& operator=(AsyncRange&&) noexcept = default;
  void start(Phase, Tag) noexcept {}
  void complete(Outcome) noexcept {}
};
class IterationScope {
public:
  IterationScope(Role, uint64_t) noexcept {}
};
#endif

} // namespace rns8::detail::trace
