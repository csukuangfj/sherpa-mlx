set(sources
  circular-buffer.cc
  file-utils.cc
  parse-options.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  text-utils.cc
  vad-model-config.cc
  vad-model.cc
  voice-activity-detector.cc
  wave-reader.cc
  wave-writer.cc
)

# Always static build
add_library(sherpa-mlx-core STATIC ${sources})
target_include_directories(
  sherpa-mlx-core
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    $ENV{MLX_CMAKE_DIR}/include
)

set_target_properties(
    sherpa-mlx-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)

set(MLX_DIR $ENV{MLX_CMAKE_DIR}/share/cmake/MLX)
find_package(MLX CONFIG REQUIRED)
target_link_libraries(
  sherpa-mlx-core
  PUBLIC
  mlx
  )

if(UNIX AND NOT APPLE)
  # It works on GitHub actions without linking to pthread or dl
  # but it fails on my ubuntu
  find_package(Threads REQUIRED)
  target_link_libraries(sherpa-mlx-core
    PUBLIC
      Threads::Threads
      dl
  )
endif()

add_executable(sherpa-mlx-vad sherpa-mlx-vad.cc)

target_link_libraries(sherpa-mlx-vad sherpa-mlx-core)
target_link_libraries(sherpa-mlx-vad "-Wl,-rpath,$ENV{MLX_CMAKE_DIR}/lib")

set(srcs_to_check)
foreach(s IN LISTS sources)
  list(APPEND srcs_to_check ${CMAKE_CURRENT_LIST_DIR}/${s})
endforeach()

# For clang-tidy
add_custom_target(
  clang-tidy-check
  clang-tidy -p ${CMAKE_BINARY_DIR}/compile_commands.json --config-file ${PROJECT_SOURCE_DIR}/.clang-tidy ${srcs_to_check}
  DEPENDS ${sources})

add_custom_target(check DEPENDS clang-tidy-check)
