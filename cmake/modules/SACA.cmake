# SACA Module

message(STATUS "Build with SACA support")
# file(GLOB RUNTIME_CUDA_SRCS src/runtime/saca/*.cc)
list(APPEND RUNTIME_SRCS src/runtime/saca/saca_module.cc)
# list(APPEND COMPILER_SRCS src/runtime/saca/saca_module.cc)
list(APPEND COMPILER_SRCS src/target/opt/build_saca_on.cc)







