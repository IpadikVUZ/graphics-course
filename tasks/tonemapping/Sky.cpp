#include "Sky.hpp"
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <utility>

void Sky::allocateResources()
{
  auto& ctx = etna::get_context();
  auto info = etna::Image::CreateInfo{
    .extent = vk::Extent3D{kCubeSize, kCubeSize, 1},
    .name = "procedural_sky_rgba16f_cubemap",
    .format = vk::Format::eR16G16B16A16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  };
  info.layers = kFaceCount;
  info.flags = vk::ImageCreateFlagBits::eCubeCompatible;
  for (auto& cubemap : cubemaps) cubemap = ctx.createImage(info);

  cubemapSampler = ctx.getDevice().createSamplerUnique(vk::SamplerCreateInfo{
    .magFilter = vk::Filter::eLinear,
    .minFilter = vk::Filter::eLinear,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .addressModeU = vk::SamplerAddressMode::eClampToEdge,
    .addressModeV = vk::SamplerAddressMode::eClampToEdge,
    .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    .maxLod = 0.f,
  }).value;
  firstBake = true;
  pendingBake = true;
  nextTile = 0;
  readyIndex = 0;
  bakeIndex = 1;
  imageLayouts = {vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined};
}

void Sky::loadShaders()
{
  etna::create_program("sky_bake", {TONEMAPPING_SHADERS_ROOT "sky_bake.comp.spv"});
}

void Sky::setupPipeline()
{
  bakePipeline = etna::get_context().getPipelineManager().createComputePipeline("sky_bake", {});
}

void Sky::drawGui()
{
  ImGui::SeparatorText("Procedural sky");
  bool changed = false;
  changed |= ImGui::SliderFloat("Sun azimuth", &sunAzimuth, -180.f, 180.f);
  changed |= ImGui::SliderFloat("Sun elevation", &sunElevation, 1.f, 89.f);
  changed |= ImGui::ColorEdit3("Sun color", &sunColor.x);
  changed |= ImGui::SliderFloat("Sun intensity", &sunIntensity, 0.f, 15.f);
  changed |= ImGui::SliderFloat("Cloud coverage", &coverage, 0.f, 1.f);
  changed |= ImGui::SliderFloat("Cloud density", &cloudDensity, 0.f, 2.f);
  changed |= ImGui::SliderFloat("Cloud scale", &cloudScale, 0.015f, 0.20f);
  changed |= ImGui::SliderFloat("Cloud absorption", &cloudAbsorption, 0.f, 3.f);
  changed |= ImGui::SliderFloat("Cloud layer bottom", &cloudBottom, 4.f, 48.f);
  changed |= ImGui::SliderFloat("Cloud layer top", &cloudTop, 8.f, 80.f);
  if (cloudTop < cloudBottom + 2.f)
  {
    cloudTop = cloudBottom + 2.f;
    changed = true;
  }
  changed |= ImGui::SliderInt("Ray marching steps", &mainSteps, 4, 40);
  changed |= ImGui::SliderInt("Shadow ray steps", &shadowSteps, 1, 8);
  if (ImGui::Button("Rebake sky")) changed = true;
  ImGui::TextDisabled("RGBA16F, 6 x 2048 x 2048; baked only when settings change");
  if (pendingBake && !firstBake)
    ImGui::Text("Baking clouds: %.0f%%", 100.f * float(nextTile) / float(kTilesPerFace * kFaceCount));
  if (changed)
  {
    pendingBake = true;
    nextTile = 0;
  }
}

glm::vec3 Sky::sunDirection() const
{
  const float az = glm::radians(sunAzimuth);
  const float el = glm::radians(sunElevation);
  return glm::normalize(glm::vec3(std::cos(el) * std::cos(az),
                                   std::sin(el), std::cos(el) * std::sin(az)));
}

void Sky::bakeIfNeeded(vk::CommandBuffer cmd)
{
  if (!firstBake && !pendingBake) return;
  ETNA_PROFILE_GPU(cmd, SkyBake);
  const auto transition = [&](uint32_t index, vk::ImageLayout destination,
                              vk::PipelineStageFlags2 dstStages, vk::AccessFlags2 dstAccess) {
    const vk::ImageLayout previous = imageLayouts[index];
    const vk::ImageMemoryBarrier2 barrier{
      .srcStageMask = previous == vk::ImageLayout::eGeneral
        ? vk::PipelineStageFlags2(vk::PipelineStageFlagBits2::eComputeShader)
        : (previous == vk::ImageLayout::eShaderReadOnlyOptimal
          ? vk::PipelineStageFlags2(vk::PipelineStageFlagBits2::eFragmentShader)
          : vk::PipelineStageFlags2(vk::PipelineStageFlagBits2::eTopOfPipe)),
      .srcAccessMask = previous == vk::ImageLayout::eGeneral
        ? vk::AccessFlags2(vk::AccessFlagBits2::eShaderStorageWrite)
        : (previous == vk::ImageLayout::eShaderReadOnlyOptimal
          ? vk::AccessFlags2(vk::AccessFlagBits2::eShaderSampledRead) : vk::AccessFlags2{}),
      .dstStageMask = dstStages,
      .dstAccessMask = dstAccess,
      .oldLayout = previous,
      .newLayout = destination,
      .image = cubemaps[index].get(),
      .subresourceRange = vk::ImageSubresourceRange{
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, kFaceCount},
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
    });
    imageLayouts[index] = destination;
  };

  const auto program = etna::get_shader_program("sky_bake");
  const auto shaderWrite = vk::AccessFlags2(vk::AccessFlagBits2::eShaderStorageWrite);
  const auto computeStage = vk::PipelineStageFlags2(vk::PipelineStageFlagBits2::eComputeShader);
  const auto fragmentStage = vk::PipelineStageFlags2(vk::PipelineStageFlagBits2::eFragmentShader);
  const auto shaderRead = vk::AccessFlags2(vk::AccessFlagBits2::eShaderSampledRead);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, bakePipeline.getVkPipeline());

  const glm::vec3 sunDir = sunDirection();
  BakeParams params{
    glm::vec4(sunDir, sunIntensity),
    glm::vec4(sunColor, 1.f),
    glm::vec4(coverage, cloudDensity, cloudScale, cloudAbsorption),
    glm::vec4(cloudBottom, cloudTop, float(mainSteps), float(shadowSteps)),
    glm::vec4(0.0f),
  };


  if (firstBake)
  {
    transition(readyIndex, vk::ImageLayout::eGeneral, computeStage, shaderWrite);
    const auto previewDescriptor = etna::create_descriptor_set(program.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, cubemaps[readyIndex].genBinding({}, vk::ImageLayout::eGeneral,
                          etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray})},
    }, BarrierBehavior::eSuppressBarriers);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, bakePipeline.getVkPipelineLayout(),
                           0, {previewDescriptor.getVkSet()}, {});
    params.clouds.x = 0.f;
    params.tileInfo = glm::vec4(0.f, 0.f, -1.f, 0.f);
    cmd.pushConstants(bakePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                      0, sizeof(params), &params);
    cmd.dispatch(kCubeSize / 8, kCubeSize / 8, kFaceCount);
    transition(readyIndex, vk::ImageLayout::eShaderReadOnlyOptimal, fragmentStage, shaderRead);
    params.clouds.x = coverage;
    firstBake = false;
  }

  if (!pendingBake) return;
  transition(bakeIndex, vk::ImageLayout::eGeneral, computeStage, shaderWrite);
  const auto bakeDescriptor = etna::create_descriptor_set(program.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, cubemaps[bakeIndex].genBinding({}, vk::ImageLayout::eGeneral,
                        etna::Image::ViewParams{.type = vk::ImageViewType::e2DArray})},
  }, BarrierBehavior::eSuppressBarriers);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, bakePipeline.getVkPipelineLayout(),
                         0, {bakeDescriptor.getVkSet()}, {});

  constexpr uint32_t tilesPerAxis = kCubeSize / kTileSize;
  constexpr uint32_t totalTiles = kTilesPerFace * kFaceCount;
  for (uint32_t n = 0; n < kTilesPerFrame && nextTile < totalTiles; ++n, ++nextTile)
  {
    const uint32_t face = nextTile / kTilesPerFace;
    const uint32_t tile = nextTile % kTilesPerFace;
    params.tileInfo = glm::vec4(float(tile % tilesPerAxis * kTileSize),
                                float(tile / tilesPerAxis * kTileSize), float(face), 0.f);
    cmd.pushConstants(bakePipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eCompute,
                      0, sizeof(params), &params);
    cmd.dispatch(kTileSize / 8, kTileSize / 8, 1);
  }
  if (nextTile == totalTiles)
  {
    transition(bakeIndex, vk::ImageLayout::eShaderReadOnlyOptimal, fragmentStage, shaderRead);
    std::swap(readyIndex, bakeIndex);
    pendingBake = false;
      nextTile = 0;
  }
}

etna::ImageBinding Sky::getBinding() const
{
  return cubemaps[readyIndex].genBinding(cubemapSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
                            etna::Image::ViewParams{.type = vk::ImageViewType::eCube});
}
