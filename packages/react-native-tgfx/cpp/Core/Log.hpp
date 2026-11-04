#pragma once

#if defined(__ANDROID__)
#include <android/log.h>
#define TGFXRN_LOGW(...) \
  __android_log_print(ANDROID_LOG_WARN, "react-native-tgfx", __VA_ARGS__)
#define TGFXRN_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "react-native-tgfx", __VA_ARGS__)
#else
#include <cstdio>
#define TGFXRN_LOGW(...)                          \
  do {                                            \
    std::fprintf(stderr, "[react-native-tgfx] "); \
    std::fprintf(stderr, __VA_ARGS__);            \
    std::fprintf(stderr, "\n");                   \
  } while (0)
#define TGFXRN_LOGI(...) TGFXRN_LOGW(__VA_ARGS__)
#endif
