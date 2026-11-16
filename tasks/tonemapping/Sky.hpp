#pragma once

#include <etna/ComputePipeline.hpp>
#include <etna/Image.hpp>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <array>
#include <cstdint>

class Sky
{
public:
  void allocateResources();
  void loadShaders();
  void setupPipeline();
  void drawGui();
  void bakeIfNeeded(vk::CommandBuffer cmd);
  etna::ImageBinding getBinding() const;
  glm::vec3 sunDirection() const;

private:
  static constexpr uint32_t kCubeSize = 2048;
  static constexpr uint32_t kFaceCount = 6;
  static constexpr uint32_t kTileSize = 256;
  static constexpr uint32_t kTilesPerFace = (kCubeSize / kTileSize) * (kCubeSize / kTileSize);
  static constexpr uint32_t kTilesPerFrame = 8;

  struct BakeParams
  {
    glm::vec4 sunDirection;
    glm::vec4 sunColor;
    glm::vec4 clouds;
    glm::vec4 layer;
    glm::vec4 tileInfo;
  };
  static_assert(sizeof(BakeParams) == 80);

  std::array<etna::Image, 2> cubemaps;
  uint32_t readyIndex = 0;
  uint32_t bakeIndex = 1;
  uint32_t nextTile = 0;
  std::array<vk::ImageLayout, 2> imageLayouts{vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined};
  vk::UniqueSampler cubemapSampler;
  etna::ComputePipeline bakePipeline{};

  glm::vec3 sunColor{1.0f, 0.94f, 0.84f};
  float sunAzimuth = 34.f;
  float sunElevation = 55.f;
  float sunIntensity = 3.4f;
  float coverage = 0.28f;
  float cloudDensity = 0.40f;
  float cloudScale = 0.065f;
  float cloudAbsorption = 0.72f;
  float cloudBottom = 14.f;
  float cloudTop = 30.f;
  int mainSteps = 10;
  int shadowSteps = 3;
  bool pendingBake = true;
  bool firstBake = true;
};
