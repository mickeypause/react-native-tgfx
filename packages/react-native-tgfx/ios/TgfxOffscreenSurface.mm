#import <Foundation/Foundation.h>

#include "OffscreenSurface.hpp"

namespace tgfxrn {

void renderOffscreenAutoreleased(const std::weak_ptr<OffscreenSurface>& next) {
  // Detached C++ threads do not drain autoreleased Metal objects themselves.
  @autoreleasepool {
    if (auto surface = next.lock()) {
      surface->renderNow();
    }
  }
}

}  // namespace tgfxrn
