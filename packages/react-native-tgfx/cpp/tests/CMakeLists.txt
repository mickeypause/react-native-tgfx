cmake_minimum_required(VERSION 3.18.0)
project(TgfxSceneTest CXX)

set(CMAKE_CXX_STANDARD 20)
set(TGFX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tgfx")

set(TGFX_USE_METAL ON CACHE BOOL "" FORCE)
set(TGFX_USE_OPENGL OFF CACHE BOOL "" FORCE)
set(TGFX_BUILD_LAYERS ON CACHE BOOL "" FORCE)
set(TGFX_BUILD_SVG ON CACHE BOOL "" FORCE)
set(TGFX_BUILD_PDF ON CACHE BOOL "" FORCE)
set(TGFX_BUILD_HELLO2D OFF CACHE BOOL "" FORCE)
add_subdirectory(${TGFX_DIR} ${CMAKE_BINARY_DIR}/tgfx)

add_executable(SceneTest SceneTest.cpp ../Scene/TgfxScene.cpp ../Path/SvgPath.cpp ../Image/SvgImage.cpp ../Image/ImageRegistry.cpp ../Offscreen/OffscreenRender.cpp ../Offscreen/OffscreenSurface.cpp ../Export/PdfExport.cpp ../Export/SvgExport.cpp
               ../TextLayout/FontRegistry.cpp ../TextLayout/TextLayout.cpp ../TextLayout/TextShaping.cpp ../Path/PathOps.cpp ../Buffers/BufferRegistry.cpp)
if(APPLE)
    enable_language(OBJCXX)
    set_property(TARGET SceneTest PROPERTY OBJCXX_STANDARD 20)
    target_sources(SceneTest PRIVATE ../../ios/TgfxAutoreleasePool.mm)
    target_link_libraries(SceneTest "-framework Foundation")
endif()
# tgfx's src too: the EXIF test names JpegCodec rather than going through
# ImageCodec::MakeFrom, which would fall through to the platform decoder and
# prove nothing about the embedded one.
target_include_directories(SceneTest PRIVATE ".." "../Buffers" "../Core" "../Export" "../Image" "../Offscreen" "../Path" "../Render" "../Scene" "../TextLayout" "../third_party/harfbuzz" "${TGFX_DIR}/include" "${TGFX_DIR}/src")
# Real font faces to register, shared with the example app so the tests exercise
# files that ship in this repo rather than whatever the host happens to install.
target_compile_definitions(SceneTest PRIVATE
    TGFXRN_TEST_TMP_DIR="${CMAKE_BINARY_DIR}"
    TGFXRN_TEST_FONT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../apps/example/src/assets/fonts")
target_link_libraries(SceneTest tgfx)

add_executable(SnapshotTest SnapshotTest.cpp)
target_include_directories(SnapshotTest PRIVATE ".." "../Buffers" "../Core" "../Export" "../Image" "../Offscreen" "../Path" "../Render" "../Scene" "../TextLayout" "${TGFX_DIR}/include")
target_link_libraries(SnapshotTest tgfx)

add_executable(AllocBench AllocBench.cpp ../Scene/TgfxScene.cpp ../Path/SvgPath.cpp ../Image/ImageRegistry.cpp
               ../Render/TgfxFrameSink.cpp ../TextLayout/FontRegistry.cpp ../TextLayout/TextLayout.cpp ../TextLayout/TextShaping.cpp ../Buffers/BufferRegistry.cpp)
target_include_directories(AllocBench PRIVATE ".." "../Buffers" "../Core" "../Export" "../Image" "../Offscreen" "../Path" "../Render" "../Scene" "../TextLayout" "../third_party/harfbuzz" "${TGFX_DIR}/include")
target_link_libraries(AllocBench tgfx)
