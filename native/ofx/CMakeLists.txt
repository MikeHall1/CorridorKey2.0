cmake_minimum_required(VERSION 3.20)
project(CorridorKeyCached LANGUAGES CXX)
add_library(CorridorKeyCached MODULE adapter.cpp)
target_compile_features(CorridorKeyCached PRIVATE cxx_std_17)
target_include_directories(CorridorKeyCached PRIVATE ../../third_party/OpenFX/include)
set_target_properties(CorridorKeyCached PROPERTIES PREFIX "" SUFFIX ".ofx")
if(MSVC)
  target_compile_options(CorridorKeyCached PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(CorridorKeyCached PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
add_executable(OfxCacheProbe cache_probe.cpp)
target_compile_features(OfxCacheProbe PRIVATE cxx_std_17)
add_executable(OfxHostTest host_test.cpp adapter.cpp)
target_compile_features(OfxHostTest PRIVATE cxx_std_17)
target_include_directories(OfxHostTest PRIVATE ../../third_party/OpenFX/include)
if(WIN32)
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The target Windows OFX bundle requires x86_64.")
  endif()
  set(CK_BUNDLE "${CMAKE_BINARY_DIR}/CorridorKeyCached.ofx.bundle")
  set_target_properties(CorridorKeyCached PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CK_BUNDLE}/Contents/Win64>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CK_BUNDLE}/Contents/Win64>")
  add_custom_command(TARGET CorridorKeyCached POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CK_BUNDLE}/Contents/Resources"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE" "${CK_BUNDLE}/Contents/Resources/LICENSE"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/../../THIRD_PARTY_NOTICES.md" "${CK_BUNDLE}/Contents/Resources/THIRD_PARTY_NOTICES.md"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/OpenFX/LICENSE.md" "${CK_BUNDLE}/Contents/Resources/OpenFX-LICENSE.md")
else()
  message(STATUS "Diagnostic ABI build only; Windows/Resolve and Apple GPU inference acceptance NOT VERIFIED.")
endif()
