#pragma once

#include <atomic>
#include <cstdint>

namespace tgfxrn {

// Per-tick decision for a display link that pauses itself once nothing animates.
class IdleGate {
 public:
  enum class Tick { Draw,
                    Skip,
                    Pause };

  void wake() {
    wakeGeneration_.fetch_add(1, std::memory_order_release);
  }

  // Display-link thread only, so the pause decision cannot race its own state.
  Tick tick(bool drawPending) {
    if (drawPending) {
      return Tick::Skip;
    }
    uint64_t generation = wakeGeneration_.load(std::memory_order_acquire);
    if (generation != lastWakeGeneration_) {
      lastWakeGeneration_ = generation;
      idleTicks_ = 0;
      return Tick::Draw;
    }
    if (++idleTicks_ <= kIdleTicksBeforePause) {
      return Tick::Draw;
    }
    // A wake raced the pause decision: skip instead, the next tick draws.
    return wakeGeneration_.load(std::memory_order_acquire) == generation
               ? Tick::Pause
               : Tick::Skip;
  }

 private:
  // Ticks without a wake before pausing; the grace lets a transient draw failure retry.
  static constexpr uint32_t kIdleTicksBeforePause = 3;

  std::atomic<uint64_t> wakeGeneration_{0};
  uint64_t lastWakeGeneration_ = 0;
  uint32_t idleTicks_ = 0;
};

}  // namespace tgfxrn
