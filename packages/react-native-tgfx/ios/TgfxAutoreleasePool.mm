#include "OffscreenRender.hpp"

// What @autoreleasepool compiles to (Clang's ARC spec, "Runtime support"),
// called directly so the pool can span a C++ scope.
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void* pool);

namespace tgfxrn {

AutoreleasePool::AutoreleasePool() : pool_(objc_autoreleasePoolPush()) {}

AutoreleasePool::~AutoreleasePool() {
  objc_autoreleasePoolPop(pool_);
}

}  // namespace tgfxrn
