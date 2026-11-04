#import "TgfxMetalView.h"

#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "IdleGate.hpp"
#include "Snapshot.hpp"
#include "TgfxRenderer.hpp"
#include "tgfx/core/ColorSpace.h"
#include "tgfx/gpu/metal/MetalWindow.h"

#if TARGET_OS_SIMULATOR
// The simulator's CAMetalLayer throws on the XR formats; 8-bit still carries P3, with less precision.
static const MTLPixelFormat kWidePixelFormat = MTLPixelFormatBGRA8Unorm;
#else
static const MTLPixelFormat kWidePixelFormat = MTLPixelFormatBGRA10_XR_sRGB;
#endif

// The `token` is the encoded buffer's owning reference, so ArrayBuffer.wrap needs no copy.
void TgfxSnapshotRelease(void* token) {
  delete static_cast<std::shared_ptr<tgfx::Data>*>(token);
}

@implementation TgfxMetalView {
  std::unique_ptr<tgfxrn::TgfxRenderer> _renderer;
  // Captured on the main thread; -[UIView layer] itself is main-thread-only.
  CAMetalLayer* _layer;
  CADisplayLink* _displayLink;
  dispatch_queue_t _renderQueue;
  std::atomic<bool> _drawPending;
  tgfxrn::IdleGate _idleGate;
  std::atomic<bool> _windowBound;
  // The space the bound window was built in. Render queue only.
  bool _boundP3;
  // Set once the display link is gone, so a snapshot staged after that is rejected, not hung.
  std::atomic<bool> _released;
  std::atomic<double> _drawableWidth;
  std::atomic<double> _drawableHeight;
  // Layer properties are UIKit state, so the render queue asks the main thread to
  // write them and binds only once the written config matches what it wants.
  std::atomic<bool> _layerWantsP3;
  std::atomic<bool> _layerConfigPending;
  bool _pendingRecycleClear;
}

+ (Class)layerClass {
  return [CAMetalLayer class];
}

- (CAMetalLayer*)metalLayer {
  return _layer;
}

- (instancetype)initWithFrame:(CGRect)frame {
  if (self = [super initWithFrame:frame]) {
    [self commonInit];
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  if (self = [super initWithCoder:coder]) {
    [self commonInit];
  }
  return self;
}

- (void)commonInit {
  _renderer = std::make_unique<tgfxrn::TgfxRenderer>();
  // The link pauses when idle, so a change nobody committed has to unpause it. Weak: the view owns it.
  __weak __typeof(self) weakSelf = self;
  _renderer->setWakeCallback([weakSelf] { [weakSelf wakeUp]; });
  _windowBound = false;
  _boundP3 = false;
  _released = false;
  _drawPending = false;
  _drawableWidth = 0;
  _drawableHeight = 0;
  _renderQueue = dispatch_queue_create("tgfx.render", DISPATCH_QUEUE_SERIAL);
  dispatch_set_target_queue(
      _renderQueue, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
  self.contentScaleFactor = UIScreen.mainScreen.nativeScale;
  self.opaque = NO;
  _layer = (CAMetalLayer*)self.layer;
  CAMetalLayer* layer = _layer;
  layer.device = MTLCreateSystemDefaultDevice();
  layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  // sRGB is pre-applied here so the common bind never has to hop back to the main thread.
  CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  layer.colorspace = srgb;
  CGColorSpaceRelease(srgb);
  _layerWantsP3 = false;
  _layerConfigPending = false;
  _pendingRecycleClear = false;
  layer.contentsScale = self.contentScaleFactor;

  _displayLink = [CADisplayLink displayLinkWithTarget:self
                                             selector:@selector(render)];
  [_displayLink addToRunLoop:NSRunLoop.mainRunLoop
                     forMode:NSRunLoopCommonModes];
}

- (void)layoutSubviews {
  [super layoutSubviews];
  CGFloat scale = self.contentScaleFactor;
  CGSize size = self.bounds.size;
  CAMetalLayer* layer = [self metalLayer];
  layer.contentsScale = scale;
  layer.drawableSize = CGSizeMake(size.width * scale, size.height * scale);
  _drawableWidth = size.width * scale;
  _drawableHeight = size.height * scale;
  if (_renderer != nullptr) {
    _renderer->setPixelRatio(static_cast<float>(scale));
    _renderer->onResize();
  }
  [self wakeUp];
}

- (void)bindWindowIfNeeded {
  if (_renderer == nullptr) {
    return;
  }
  bool wantsP3 = _renderer->wantsDisplayP3();
  if (_windowBound) {
    if (_boundP3 == wantsP3) {
      return;
    }
    // Recycled from a canvas in the other space: a window keeps its space, so build another.
    _renderer->setWindow(nullptr);
    _windowBound = false;
  }
  if (_drawableWidth <= 0 || _drawableHeight <= 0) {
    return;
  }
  // The layer's own space too, or the surface is P3 while the layer still presents sRGB.
  if (_layerWantsP3.load() != wantsP3) {
    if (!_layerConfigPending.exchange(true)) {
      dispatch_async(dispatch_get_main_queue(), ^{
        // +1 from Create; the layer retains it, so this scope has to give its reference back.
        CGColorSpaceRef cgSpace = CGColorSpaceCreateWithName(
            wantsP3 ? kCGColorSpaceDisplayP3 : kCGColorSpaceSRGB);
        self->_layer.colorspace = cgSpace;
        CGColorSpaceRelease(cgSpace);
        self->_layer.pixelFormat =
            wantsP3 ? kWidePixelFormat : MTLPixelFormatBGRA8Unorm;
        self->_layerWantsP3 = wantsP3;
        self->_layerConfigPending = false;
        [self wakeUp];
      });
    }
    return;
  }
  auto space = wantsP3 ? tgfx::ColorSpace::DisplayP3() : tgfx::ColorSpace::SRGB();
  auto window = tgfx::MetalWindow::MakeFrom(_layer, nullptr, space);
  if (window != nullptr) {
    _renderer->setWindow(window);
    _boundP3 = wantsP3;
    _windowBound = true;
  }
}

- (void)render {
  auto tick = _idleGate.tick(_drawPending.load(std::memory_order_acquire));
  if (tick == tgfxrn::IdleGate::Tick::Pause) {
    _displayLink.paused = YES;
  }
  if (tick != tgfxrn::IdleGate::Tick::Draw) {
    return;
  }
  if (_drawPending.exchange(true)) {
    return;
  }
  dispatch_async(_renderQueue, ^{
    [self bindWindowIfNeeded];
    bool snapshotPending = false;
    if (self->_renderer != nullptr) {
      self->_renderer->draw();
      snapshotPending = self->_renderer->snapshotPending();
    }
    self->_drawPending = false;
    // A GPU transfer needs another tick, which the idle gate would otherwise pause before.
    if (snapshotPending) {
      [self wakeUp];
    }
  });
}

- (void)wakeUp {
  _idleGate.wake();
  if (_displayLink == nil) {
    return;
  }
  if (NSThread.isMainThread) {
    _displayLink.paused = NO;
    return;
  }
  dispatch_async(dispatch_get_main_queue(), ^{
    self->_displayLink.paused = NO;
  });
}

- (void)setScene:(const double*)words
           count:(NSInteger)count
         strings:(NSArray<NSString*>*)strings
     awaitValues:(BOOL)awaitValues {
  if (_renderer == nullptr || words == nullptr || count <= 0) {
    // As on Android: this drops the string-table delta too, which JS cannot resend.
    return;
  }
  std::vector<std::string> table;
  table.reserve(strings.count);
  for (NSString* string in strings) {
    table.emplace_back(string.UTF8String);
  }
  _renderer->setScene(std::span<const double>(words, static_cast<size_t>(count)),
                      table, awaitValues == YES);
  [self wakeUp];
}

- (void)setOpaqueSurface:(BOOL)opaque {
  self.opaque = opaque;
  [self metalLayer].opaque = opaque;
  if (_renderer != nullptr) {
    _renderer->setOpaque(opaque);
  }
  [self wakeUp];
}

// Deferred while on screen: painting the still-visible view empty flashed white during unmount.
- (void)clearScene {
  if (self.window != nil) {
    _pendingRecycleClear = true;
    return;
  }
  [self clearSceneNow];
}

- (void)clearSceneNow {
  _pendingRecycleClear = false;
  if (_renderer != nullptr) {
    _renderer->clearScene();
  }
  [self wakeUp];
}

- (void)didMoveToWindow {
  [super didMoveToWindow];
  if (self.window == nil && _pendingRecycleClear) {
    [self clearSceneNow];
  }
}

- (void)setValues:(const double*)values count:(int32_t)count {
  if (_renderer != nullptr && values != nullptr) {
    _renderer->setValues(std::vector<double>(values, values + count));
  }
  [self wakeUp];
}

- (void)setStringValues:(const double*)indices
                  count:(int32_t)count
                 values:(NSArray<NSString*>*)values {
  if (_renderer != nullptr && indices != nullptr && values != nil) {
    std::vector<double> slots(indices, indices + count);
    std::vector<std::string> strings;
    strings.reserve(values.count);
    for (NSString* value in values) {
      const char* utf = value.UTF8String;
      strings.emplace_back(utf != nullptr ? utf : "");
    }
    _renderer->setStringValues(slots, strings);
  }
  [self wakeUp];
}

- (void)setViewportZoom:(float)zoom x:(float)x y:(float)y {
  // Only on a real move: an idle tick would otherwise unpause the display link
  // every frame for a scene that has not changed.
  if (_renderer != nullptr && _renderer->setViewport(zoom, x, y)) {
    [self wakeUp];
  }
}

- (void)setDisplayP3:(BOOL)displayP3 {
  if (_renderer != nullptr) {
    _renderer->setColorSpace(displayP3 == YES);
    [self requestRender];
  }
}

- (NSArray<NSNumber*>*)hitTestX:(float)x y:(float)y precise:(BOOL)precise {
  if (_renderer == nullptr) {
    return @[];
  }
  auto hits = _renderer->hitTest(x, y, precise == YES);
  NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:hits.size()];
  for (double value : hits) {
    [out addObject:@(value)];
  }
  return out;
}

- (void)snapshotWithFormat:(NSString*)format
                   quality:(int)quality
                      rect:(CGRect)rect
                     scale:(float)scale
                      path:(NSString*)path
                completion:(TgfxSnapshotCompletion)completion {
  if (_renderer == nullptr || _released) {
    completion(nullptr, 0, nullptr, 0, 0,
               @"[tgfx] snapshot: the canvas is no longer mounted");
    return;
  }
  tgfxrn::SnapshotRequest request;
  request.format = format.UTF8String;
  request.quality = quality;
  request.x = static_cast<float>(rect.origin.x);
  request.y = static_cast<float>(rect.origin.y);
  request.width = static_cast<float>(rect.size.width);
  request.height = static_cast<float>(rect.size.height);
  request.scale = scale;
  request.path = path != nil ? path.UTF8String : "";
  auto deliver = [completion](std::shared_ptr<tgfx::Data> data, int width, int height,
                              const std::string& error) {
    if (!error.empty()) {
      completion(nullptr, 0, nullptr, 0, 0, @(error.c_str()));
      return;
    }
    if (data == nullptr) {
      completion(nullptr, 0, nullptr, width, height, nil);
      return;
    }
    auto* token = new std::shared_ptr<tgfx::Data>(std::move(data));
    completion(const_cast<void*>((*token)->data()),
               static_cast<NSInteger>((*token)->size()), token, width, height, nil);
  };
  // Readbacks never touch the framebuffer-only drawable (see startSnapshots), so no layer work here.
  _renderer->requestSnapshot(request, deliver);
  [self wakeUp];
}

- (void)requestRender {
  [self wakeUp];
}

- (void)releaseRenderer {
  [_displayLink invalidate];
  _displayLink = nil;
  _windowBound = false;
  _released = true;
  if (_renderer == nullptr) {
    return;
  }
  dispatch_sync(_renderQueue, ^{
    self->_renderer->releaseWindow();
  });
}

- (void)dealloc {
  [self releaseRenderer];
}

@end
