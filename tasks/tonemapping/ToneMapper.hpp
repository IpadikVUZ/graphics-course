#pragma once

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <array>
#include <chrono>
#include <cstdint>

class ToneMapper
{
public:
  void allocateResources(glm::uvec2 resolution);
  void resize(glm::uvec2 resolution);
  void loadShaders();
  void setupPipeline(vk::Format swapchainFormat);
  void drawGui();
  void prepare(vk::CommandBuffer cmd);
  void measure(vk::CommandBuffer cmd, const etna::Image& hdr, uint32_t frameSlot);
  void draw(vk::CommandBuffer cmd, const etna::Image& hdr, uint32_t frameSlot);

private:
  static constexpr uint32_t kLutSize = 32;
  static constexpr uint32_t kHistogramBins = 128;

  struct FrameGpu
  {
    etna::Buffer groupRanges;
    etna::Buffer frameRange;
    etna::Buffer histogram;
    etna::Buffer exposure;
  };

  std::array<FrameGpu, 2> frames;
  etna::Image lut;
  etna::Buffer lutStaging;
  vk::UniqueSampler sampler;
  vk::UniqueSampler lutSampler;
  etna::ComputePipeline tilesPipeline{};
  etna::ComputePipeline rangePipeline{};
  etna::ComputePipeline histogramPipeline{};
  etna::ComputePipeline exposurePipeline{};
  etna::GraphicsPipeline postPipeline{};
  glm::uvec2 resolution{0, 0};
  bool lutPending = true;
  bool firstFrame = true;
  bool swapchainIsSrgb = false;
  bool enableToneMapping = true;
  bool autoExposure = true;
  bool colorGrading = true;
  bool filmic = true;
  bool useSharedHistogram = true;
  float exposureEV = 0.18f;
  float adaptationSpeed = 0.9f;
  float autoExposureStrength = 0.15f;
  float elapsedSeconds = 1.f / 60.f;
  std::chrono::steady_clock::time_point lastUpdate{};
};
