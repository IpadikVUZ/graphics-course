#pragma once
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/ComputePipeline.hpp>
#include <glm/glm.hpp>
#include "wsi/Keyboard.hpp"
#include "FramePacket.hpp"

class WorldRenderer
{
public:
  void allocateResources(glm::uvec2 swapchain_resolution);
  void resizeDepth(glm::uvec2 swapchain_resolution);
  void loadShaders();
  void setupPipelines(vk::Format swapchain_format);
  void debugInput(const Keyboard&);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_view);

private:
  struct NoiseSettings
  {
    float scale = 3.2f;
    float persistence = 0.52f;
    float lacunarity = 2.05f;
    float heightScale = 1.f;
    float islandRadius = 0.96f;
    int octaves = 6;
    int seed = 17;
    float shore = 0.63f;
  } noise;

  struct DrawSettings
  {
    glm::mat4 viewProj{1.0f};
    glm::vec4 light{25.f, 45.f, 20.f, 4.5f};
    glm::vec4 eye{0.f, 30.f, 70.f, 0.f};
    glm::vec4 detail{14.f, 0.7f, 0.f, 0.12f};
    glm::vec4 levels{1.6f, 6.8f, 9.6f, 1.0f};
  } draw;
  static_assert(sizeof(DrawSettings) == 128);
  static_assert(sizeof(NoiseSettings) == 32);

  void generateMaps(vk::CommandBuffer cmd_buf);
  static constexpr uint32_t kGridCells = 256;
  static constexpr uint32_t kIndexCount = kGridCells * kGridCells * 6;

  etna::Image depth;
  etna::Image heightMap;
  etna::Image albedoAtlas;
  etna::Image roughnessAtlas;
  etna::Image detailHeightAtlas;
  etna::Sampler sampler;
  etna::Buffer indices;
  etna::GraphicsPipeline drawPipeline{};
  etna::ComputePipeline generatePipeline{};
  glm::uvec2 resolution{};
  bool pendingGeneration = true;
};
