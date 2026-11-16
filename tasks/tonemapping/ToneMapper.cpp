#include "ToneMapper.hpp"
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace
{
void computeBarrier(vk::CommandBuffer cmd)
{
  const vk::MemoryBarrier2 barrier{
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
  };
  cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

etna::Buffer storageBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, const char* name)
{
  return etna::get_context().createBuffer(etna::Buffer::CreateInfo{
    .size = size,
    .bufferUsage = usage,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    .allocationCreate = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .name = name,
  });
}
}

void ToneMapper::allocateResources(glm::uvec2 res)
{
  auto& ctx = etna::get_context();
  lut = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{kLutSize, kLutSize, kLutSize},
    .name = "color_grading_32x32x32",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
    .type = vk::ImageType::e3D,
  });

  const std::string path = std::string(TONEMAPPING_ASSETS_ROOT) + "filmic_32_rgba8.bin";
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  constexpr vk::DeviceSize byteCount = kLutSize * kLutSize * kLutSize * 4;
  if (!file || file.tellg() != static_cast<std::streamoff>(byteCount))
    throw std::runtime_error("Missing or invalid color LUT: " + path);
  file.seekg(0);
  lutStaging = storageBuffer(byteCount, vk::BufferUsageFlagBits::eTransferSrc, "3D_LUT_upload");
  file.read(reinterpret_cast<char*>(lutStaging.data()), static_cast<std::streamsize>(byteCount));
  if (!file) throw std::runtime_error("Cannot read color LUT: " + path);

  sampler = ctx.getDevice().createSamplerUnique(vk::SamplerCreateInfo{
    .magFilter = vk::Filter::eNearest,
    .minFilter = vk::Filter::eNearest,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .addressModeU = vk::SamplerAddressMode::eClampToEdge,
    .addressModeV = vk::SamplerAddressMode::eClampToEdge,
    .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    .maxLod = 0.f,
  }).value;
  lutSampler = ctx.getDevice().createSamplerUnique(vk::SamplerCreateInfo{
    .magFilter = vk::Filter::eLinear,
    .minFilter = vk::Filter::eLinear,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .addressModeU = vk::SamplerAddressMode::eClampToEdge,
    .addressModeV = vk::SamplerAddressMode::eClampToEdge,
    .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    .maxLod = 0.f,
  }).value;
  lutPending = true;
  resize(res);
}

void ToneMapper::resize(glm::uvec2 res)
{
  resolution = res;
  uint32_t groupsX = (res.x + 15) / 16;
  uint32_t groupsY = (res.y + 15) / 16;
  vk::DeviceSize groupBytes = vk::DeviceSize(groupsX) * groupsY * sizeof(glm::vec2);
  for (auto& f : frames)
  {
    f.groupRanges = storageBuffer(groupBytes, vk::BufferUsageFlagBits::eStorageBuffer, "luma_tile_ranges");
    f.frameRange = storageBuffer(sizeof(glm::vec4), vk::BufferUsageFlagBits::eStorageBuffer, "luma_range");
    f.histogram = storageBuffer(kHistogramBins * sizeof(uint32_t),
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                vk::BufferUsageFlagBits::eTransferDst, "luma_histogram");
    f.exposure = storageBuffer(sizeof(glm::vec4), vk::BufferUsageFlagBits::eStorageBuffer, "exposure_per_frame");
    const glm::vec4 initialExposure(1.f, 1.f, 0.f, 0.f);
    std::memcpy(f.exposure.data(), &initialExposure, sizeof(initialExposure));
  }
  firstFrame = true;
  lastUpdate = {};
}

void ToneMapper::loadShaders()
{
  etna::create_program("tone_luma_tiles", {TONEMAPPING_SHADERS_ROOT "luma_tiles.comp.spv"});
  etna::create_program("tone_luma_range", {TONEMAPPING_SHADERS_ROOT "luma_range.comp.spv"});
  etna::create_program("tone_luma_histogram", {TONEMAPPING_SHADERS_ROOT "luma_histogram.comp.spv"});
  etna::create_program("tone_auto_exposure", {TONEMAPPING_SHADERS_ROOT "auto_exposure.comp.spv"});
  etna::create_program("tone_post", {TONEMAPPING_SHADERS_ROOT "post.frag.spv",
                                     TONEMAPPING_SHADERS_ROOT "fullscreen.vert.spv"});
}

void ToneMapper::setupPipeline(vk::Format format)
{
  swapchainIsSrgb = format == vk::Format::eB8G8R8A8Srgb || format == vk::Format::eR8G8B8A8Srgb;
  auto& manager = etna::get_context().getPipelineManager();
  tilesPipeline = manager.createComputePipeline("tone_luma_tiles", {});
  rangePipeline = manager.createComputePipeline("tone_luma_range", {});
  histogramPipeline = manager.createComputePipeline("tone_luma_histogram", {});
  exposurePipeline = manager.createComputePipeline("tone_auto_exposure", {});
  auto info = etna::GraphicsPipeline::CreateInfo{
    .rasterizationConfig = vk::PipelineRasterizationStateCreateInfo{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .lineWidth = 1.f,
    },
    .fragmentShaderOutput = {.colorAttachmentFormats = {format}},
  };
  info.depthConfig.depthTestEnable = vk::False;
  info.depthConfig.depthWriteEnable = vk::False;
  postPipeline = manager.createGraphicsPipeline("tone_post", info);
}

void ToneMapper::drawGui()
{
  ImGui::SeparatorText("Tone mapping and color grading");
  ImGui::Checkbox("Enable tone mapping", &enableToneMapping);
  ImGui::Checkbox("Auto exposure", &autoExposure);
  if (autoExposure)
  {
    ImGui::SliderFloat("Auto exposure influence", &autoExposureStrength, 0.f, 1.f);
    ImGui::SliderFloat("Eye adaptation speed", &adaptationSpeed, 0.2f, 4.f);
  }
  ImGui::Checkbox("Filmic curve (ACES fit)", &filmic);
  ImGui::Checkbox("Enable 3D LUT", &colorGrading);
  ImGui::Checkbox("Shared-memory histogram", &useSharedHistogram);
  ImGui::SliderFloat("Exposure compensation (EV)", &exposureEV, -3.f, 3.f);
  ImGui::TextDisabled("HDR: B10G11R11 UFLOAT | histogram: 128 log2-luminance bins");
  ImGui::TextDisabled("Auto exposure is metered with highlights and smoothed in EV");
  ImGui::TextDisabled("Shared on/off: compare ToneHistogram GPU zones in Tracy");
}

void ToneMapper::prepare(vk::CommandBuffer cmd)
{
  if (!lutPending) return;
  etna::set_state(cmd, lut.get(), vk::PipelineStageFlagBits2::eTransfer,
                  vk::AccessFlagBits2::eTransferWrite,
                  vk::ImageLayout::eTransferDstOptimal, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd);
  const std::array<vk::BufferImageCopy, 1> region{{
    vk::BufferImageCopy{
      .bufferOffset = 0,
      .imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageExtent = vk::Extent3D{kLutSize, kLutSize, kLutSize},
    }
  }};
  cmd.copyBufferToImage(lutStaging.get(), lut.get(), vk::ImageLayout::eTransferDstOptimal, region);
  etna::set_state(cmd, lut.get(), vk::PipelineStageFlagBits2::eFragmentShader,
                  vk::AccessFlagBits2::eShaderSampledRead,
                  vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd);
  lutPending = false;
}

void ToneMapper::measure(vk::CommandBuffer cmd, const etna::Image& hdr, uint32_t slot)
{
  auto& frame = frames[slot % 2];
  auto& previous = frames[(slot + 1) % 2];
  const uint32_t groupsX = (resolution.x + 15) / 16;
  const uint32_t groupsY = (resolution.y + 15) / 16;
  const uint32_t groupCount = groupsX * groupsY;

  const vk::MemoryBarrier2 fromPreviousFrame{
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
  };
  cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &fromPreviousFrame});

  auto now = std::chrono::steady_clock::now();
  elapsedSeconds = lastUpdate.time_since_epoch().count() == 0 ? (1.f / 60.f) :
                   std::clamp(std::chrono::duration<float>(now - lastUpdate).count(), 0.001f, 0.1f);
  lastUpdate = now;

  {
    ETNA_PROFILE_GPU(cmd, ToneMinMaxTiles);
    auto shader = etna::get_shader_program("tone_luma_tiles");
    auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, hdr.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{1, frame.groupRanges.genBinding()},
    }, BarrierBehavior::eSuppressBarriers);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, tilesPipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, tilesPipeline.getVkPipelineLayout(), 0,
                           {set.getVkSet()}, {});
    cmd.dispatch(groupsX, groupsY, 1);
  }
  computeBarrier(cmd);
  {
    ETNA_PROFILE_GPU(cmd, ToneMinMaxReduce);
    auto shader = etna::get_shader_program("tone_luma_range");
    auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, frame.groupRanges.genBinding()},
      etna::Binding{1, frame.frameRange.genBinding()},
    });
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, rangePipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rangePipeline.getVkPipelineLayout(), 0,
                           {set.getVkSet()}, {});
    cmd.pushConstants(rangePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                      0, sizeof(groupCount), &groupCount);
    cmd.dispatch(1, 1, 1);
  }
  computeBarrier(cmd);

  cmd.fillBuffer(frame.histogram.get(), 0, kHistogramBins * sizeof(uint32_t), 0u);
  const vk::MemoryBarrier2 cleared{
    .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
  };
  cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &cleared});
  {
    ETNA_PROFILE_GPU(cmd, ToneHistogram);
    auto shader = etna::get_shader_program("tone_luma_histogram");
    auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, hdr.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{1, frame.frameRange.genBinding()},
      etna::Binding{2, frame.histogram.genBinding()},
    }, BarrierBehavior::eSuppressBarriers);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, histogramPipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, histogramPipeline.getVkPipelineLayout(), 0,
                           {set.getVkSet()}, {});
    int shared = useSharedHistogram ? 1 : 0;
    cmd.pushConstants(histogramPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                      0, sizeof(shared), &shared);
    cmd.dispatch(groupsX, groupsY, 1);
  }
  computeBarrier(cmd);
  {
    ETNA_PROFILE_GPU(cmd, ToneAutoExposure);
    auto shader = etna::get_shader_program("tone_auto_exposure");
    auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, frame.frameRange.genBinding()},
      etna::Binding{1, frame.histogram.genBinding()},
      etna::Binding{2, previous.exposure.genBinding()},
      etna::Binding{3, frame.exposure.genBinding()},
    });
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, exposurePipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, exposurePipeline.getVkPipelineLayout(), 0,
                           {set.getVkSet()}, {});
    const std::array<glm::vec4, 2> controls = {
      glm::vec4(elapsedSeconds, adaptationSpeed, exposureEV, firstFrame ? 1.f : 0.f),
      glm::vec4(autoExposure ? 1.f : 0.f, autoExposureStrength, 0.f, 0.f),
    };
    cmd.pushConstants(exposurePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                      0, sizeof(controls), controls.data());
    cmd.dispatch(1, 1, 1);
  }
  firstFrame = false;
  const vk::MemoryBarrier2 barrier{
    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
  };
  cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

void ToneMapper::draw(vk::CommandBuffer cmd, const etna::Image& hdr, uint32_t slot)
{
  auto shader = etna::get_shader_program("tone_post");
  auto descriptor = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, hdr.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{1, lut.genBinding(lutSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
                        etna::Image::ViewParams{.type = vk::ImageViewType::e3D})},
    etna::Binding{2, frames[slot % 2].exposure.genBinding()},
  }, BarrierBehavior::eSuppressBarriers);
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, postPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, postPipeline.getVkPipelineLayout(), 0,
                         {descriptor.getVkSet()}, {});
  glm::vec4 controls(enableToneMapping ? 1.f : 0.f, filmic ? 1.f : 0.f,
                     colorGrading ? 1.f : 0.f, swapchainIsSrgb ? 1.f : 0.f);
  cmd.pushConstants(postPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eFragment,
                    0, sizeof(controls), &controls);
  cmd.draw(3, 1, 0, 0);
}
