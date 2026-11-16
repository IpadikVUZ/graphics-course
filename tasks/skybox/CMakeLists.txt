set(SKYBOX_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")

add_executable(skybox
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  RockField.cpp
  Sky.cpp
)
target_compile_definitions(skybox PRIVATE
  SKYBOX_ASSETS_ROOT="${SKYBOX_ASSETS_DIR}/"
)
target_link_libraries(skybox PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(skybox
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
)

target_compile_definitions(skybox PRIVATE SKYBOX_TEXTURES_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/textures/\")
