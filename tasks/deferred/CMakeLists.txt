set(DEFERRED_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")

add_executable(deferred
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  RockField.cpp
)
target_compile_definitions(deferred PRIVATE
  DEFERRED_ASSETS_ROOT="${DEFERRED_ASSETS_DIR}/"
)
target_link_libraries(deferred PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(deferred
  shaders/terrain.vert
  shaders/terrain.frag
  shaders/generate.comp
  shaders/rocks.vert
  shaders/rocks.frag
  shaders/fullscreen.vert
  shaders/lighting.frag
  shaders/decal.frag
  shaders/decal_generate.comp
)
