cmake_minimum_required(VERSION 3.16)
project(qnn_minimal CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# QNN SDK 경로
set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
if(NOT QNN_SDK_ROOT)
  # message(FATAL_ERROR "QNN_SDK_ROOT not set")
  set(QNN_SDK_ROOT "/workspace/qairt/2.37.0.250724")
endif()

# QNN include root
set(QNN_INCLUDE_DIR ${QNN_SDK_ROOT}/include/QNN)

# QNN lib root (CPU backend 예시)
if(ANDROID)
  message(STATUS "Building for ANDROID")
  set(QNN_LIB_DIR ${QNN_SDK_ROOT}/lib/aarch64-android)
else()
  message(STATUS "Building for x86")
  set(QNN_LIB_DIR ${QNN_SDK_ROOT}/lib/x86_64-linux-clang)
endif()
# -----------------------------
# QnnSystem imported target
# -----------------------------
add_library(QNN::System SHARED IMPORTED)

set_target_properties(QNN::System PROPERTIES
  IMPORTED_LOCATION
    ${QNN_LIB_DIR}/libQnnSystem.so
  INTERFACE_INCLUDE_DIRECTORIES
    ${QNN_INCLUDE_DIR}
)

# -----------------------------
# executable
# -----------------------------
if(ANDROID)
  set(QNN_APP_TARGET QnnRun)
  add_executable(${QNN_APP_TARGET} QnnLinearRun.cpp
                                   QnnSetup.cpp
                                   QnnUtils.cpp
                                   QnnSharedBuffer.cpp)
else()
  set(QNN_APP_TARGET QnnAOT)
  add_executable(${QNN_APP_TARGET} QnnLinearAOT.cpp
                                   QnnSetup.cpp
                                   QnnUtils.cpp)
endif()

target_link_libraries(${QNN_APP_TARGET}
  PRIVATE
  QNN::System
)

target_include_directories(${QNN_APP_TARGET} PRIVATE ./)