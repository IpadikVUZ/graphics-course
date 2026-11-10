#pragma once
#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>
#include <array>
#include "wsi/Keyboard.hpp"
#include "FramePacket.hpp"
#include "RenderSettings.hpp"
#include "RockField.hpp"

class WorldRenderer
{
public:
  void allocateResources(glm::uvec2 swapchainResolution);
  void resizeDepth(glm::uvec2 swapchainResolution);
  void loadShaders();
  void setupPipelines(vk::Format swapchainFormat);
  void debugInput(const Keyboard&);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(vk::CommandBuffer cmd, vk::Image target, vk::ImageView targetView, uint32_t frameSlot);

private:
  static constexpr uint32_t kGridCells = 256;
  static constexpr uint32_t kIndexCount = kGridCells * kGridCells * 6;
  static constexpr uint32_t kDecalCount = 5;

  struct GBuffer
  {
    etna::Image albedoAO;
    etna::Image normalRM;
  };
  struct DecalControl
  {
    glm::vec3 center;
    float yawDeg;
    glm::vec3 halfSize;
    int texture;
    float opacity;
    float normalStrength;
    float roughnessStrength;
    glm::vec3 tint;
    bool enabled;
  };

  void generateMaps(vk::CommandBuffer cmd);
  void generateDecalTextures(vk::CommandBuffer cmd);
  void writeDecalBuffer(uint32_t frameSlot);
  void drawTerrain(vk::CommandBuffer cmd, bool depthOnly);

  NoiseSettings noise;
  DrawSettings draw;
  LightingSettings lighting;
  RockField rocks;

  etna::Image depth;

  std::array<std::array<GBuffer, 2>, 2> gBuffers;
  etna::Image heightMap;
  etna::Image albedoAtlas;
  etna::Image roughnessAtlas;
  etna::Image detailHeightAtlas;
  etna::Image decalAlbedo;
  etna::Image decalNormals;
  etna::Image decalRoughness;
  etna::Sampler sampler;
  etna::Buffer indices;
  std::array<etna::Buffer, 2> decalGpuBuffers;

  etna::GraphicsPipeline depthPipeline{};
  std::array<etna::GraphicsPipeline, 2> geometryPipelines{};
  std::array<etna::GraphicsPipeline, 2> decalPipelines{};
  etna::GraphicsPipeline lightingPipeline{};
  etna::ComputePipeline generatePipeline{};
  etna::ComputePipeline generateDecalPipeline{};

  std::array<DecalControl, kDecalCount> decals{{
    {{ -17.f, 3.f,  -8.f},  18.f, {6.0f, 10.f, 4.5f}, 0, 0.93f, 0.78f, 0.82f, {1.f, 1.f, 1.f}, true},
    {{  -2.f, 4.f,  10.f}, -24.f, {5.4f, 10.f, 6.0f}, 1, 0.92f, 0.82f, 0.85f, {1.f, 1.f, 1.f}, true},
    {{  14.f, 4.f,   4.f},  35.f, {6.0f, 10.f, 4.0f}, 2, 0.96f, 0.66f, 0.85f, {1.f, 1.f, 1.f}, true},
    {{  -6.f, 4.f, -23.f},  12.f, {8.0f, 10.f, 6.0f}, 3, 0.83f, 0.40f, 1.00f, {1.f, 1.f, 1.f}, true},
    {{  22.f, 4.f, -16.f}, -42.f, {5.5f, 10.f, 5.0f}, 1, 0.76f, 0.75f, 0.70f, {1.f, 1.f, 1.f}, true},
  }};
  glm::uvec2 resolution{};
  bool pendingGeneration = true;
  bool pendingDecalTextures = true;
  bool compactGBuffer = true;
  bool decalsEnabled = true;
  float globalDecalOpacity = 1.0f;
  float globalDecalNormals = 1.0f;
  float globalDecalRoughness = 1.0f;
};
