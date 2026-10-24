add_executable(terrain
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
)

target_link_libraries(terrain PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(terrain
  shaders/terrain.vert
  shaders/terrain.frag
  shaders/generate.comp
)
