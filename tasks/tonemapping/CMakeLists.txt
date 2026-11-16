set(TONEMAPPING_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")

add_executable(tonemapping
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  RockField.cpp
  Sky.cpp
  ToneMapper.cpp
)
target_compile_definitions(tonemapping PRIVATE
  TONEMAPPING_ASSETS_ROOT="${TONEMAPPING_ASSETS_DIR}/"
)
target_link_libraries(tonemapping PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(tonemapping
  shaders/terrain.vert
  shaders/terrain.frag
  shaders/generate.comp
  shaders/rocks.vert
  shaders/rocks.frag
  shaders/fullscreen.vert
  shaders/lighting.frag
  shaders/decal.frag
  shaders/decal_generate.comp
  shaders/sky_bake.comp
  shaders/luma_tiles.comp
  shaders/luma_range.comp
  shaders/luma_histogram.comp
  shaders/auto_exposure.comp
  shaders/post.frag
)

target_compile_definitions(tonemapping PRIVATE TONEMAPPING_TEXTURES_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/textures/\")
