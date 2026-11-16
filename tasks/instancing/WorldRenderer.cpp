#include "WorldRenderer.hpp"
#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

void WorldRenderer::resizeDepth(glm::uvec2 res)
{
  resolution = res;
  depth = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{res.x, res.y, 1},
    .name = "terrain_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });
}

void WorldRenderer::allocateResources(glm::uvec2 res)
{
  resizeDepth(res);
  auto& ctx = etna::get_context();
  heightMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{1024, 1024, 1},
    .name = "terrain_height_R32_SFLOAT",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });
  const auto createMaterialAtlas = [&](const char* name, vk::Format format) {
    auto info = etna::Image::CreateInfo{
      .extent = vk::Extent3D{kAtlasWidth, kAtlasHeight, 1},
      .name = name,
      .format = format,
      .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
    };
    info.mipLevels = kTextureMipLevels;
    return ctx.createImage(info);
  };
  albedoAtlas = createMaterialAtlas("PBR_albedo_sand_grass_rock_snow", vk::Format::eR8G8B8A8Srgb);
  roughnessAtlas = createMaterialAtlas("PBR_roughness_sand_grass_rock_snow", vk::Format::eR8Unorm);
  detailHeightAtlas = createMaterialAtlas("PBR_detail_height_sand_grass_rock_snow", vk::Format::eR8Unorm);

  constexpr std::array<const char*, 3> files = {
    "albedo_mips.bin", "roughness_mips.bin", "detail_height_mips.bin"
  };
  constexpr std::array<uint32_t, 3> bytesPerPixel = {4, 1, 1};
  for (size_t i = 0; i < files.size(); ++i)
  {
    vk::DeviceSize size = 0;
    for (uint32_t mip = 0; mip < kTextureMipLevels; ++mip)
      size += vk::DeviceSize(kAtlasWidth >> mip) * (kAtlasHeight >> mip) * bytesPerPixel[i];

    const auto path = std::filesystem::path(INSTANCING_TEXTURES_ROOT) / files[i];
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(size))
      throw std::runtime_error("Missing or invalid terrain texture: " + path.string());
    input.seekg(0);

    textureStaging[i] = ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = size,
      .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = files[i],
    });
    input.read(reinterpret_cast<char*>(textureStaging[i].data()), static_cast<std::streamsize>(size));
    if (!input)
      throw std::runtime_error("Could not read terrain texture: " + path.string());
  }
  materialSampler = ctx.getDevice().createSamplerUnique(vk::SamplerCreateInfo{
    .magFilter = vk::Filter::eLinear,
    .minFilter = vk::Filter::eLinear,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .addressModeU = vk::SamplerAddressMode::eClampToEdge,
    .addressModeV = vk::SamplerAddressMode::eClampToEdge,
    .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    .maxLod = static_cast<float>(kTextureMipLevels - 1),
  }).value;
  sampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "terrain_linear_sampler"});

  std::vector<uint32_t> meshIndices;
  meshIndices.reserve(kIndexCount);
  for (uint32_t y = 0; y < kGridCells; ++y)
    for (uint32_t x = 0; x < kGridCells; ++x)
    {
      const uint32_t a = y * (kGridCells + 1) + x;
      const uint32_t b = a + 1;
      const uint32_t c = a + kGridCells + 1;
      const uint32_t d = c + 1;
      meshIndices.insert(meshIndices.end(), {a, c, b, b, c, d});
    }
  indices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = static_cast<vk::DeviceSize>(meshIndices.size() * sizeof(uint32_t)),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    .allocationCreate =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    .name = "terrain_256x256_index_grid",
  });
  std::memcpy(indices.data(), meshIndices.data(), meshIndices.size() * sizeof(uint32_t));
  rocks.allocateResources();
  pendingGeneration = true;
}

void WorldRenderer::loadShaders()
{
  rocks.loadShaders();
  etna::create_program("terrain_generate", {INSTANCING_SHADERS_ROOT "generate.comp.spv"});
  etna::create_program("terrain_surface",
                       {INSTANCING_SHADERS_ROOT "terrain.frag.spv",
                        INSTANCING_SHADERS_ROOT "terrain.vert.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  auto& manager = etna::get_context().getPipelineManager();
  generatePipeline = manager.createComputePipeline("terrain_generate", {});
  rocks.setupPipeline(swapchain_format);
  drawPipeline = manager.createGraphicsPipeline("terrain_surface", etna::GraphicsPipeline::CreateInfo{
    .rasterizationConfig = vk::PipelineRasterizationStateCreateInfo{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .lineWidth = 1.0f,
    },
    .fragmentShaderOutput = {
      .colorAttachmentFormats = {swapchain_format},
      .depthAttachmentFormat = vk::Format::eD32Sfloat,
    },
  });
}

void WorldRenderer::debugInput(const Keyboard&) {}

void WorldRenderer::update(const FramePacket& packet)
{
  const float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);
  draw.viewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  draw.eye = glm::vec4(packet.mainCam.position, 0.f);
  rocks.update(draw, noise);
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Terrain");
  ImGui::SeparatorText("Noise");
  ImGui::SliderInt("Octaves", &noise.octaves, 1, 10);
  ImGui::SliderFloat("Scale", &noise.scale, 0.5f, 12.f);
  ImGui::SliderFloat("Persistence", &noise.persistence, 0.2f, 0.85f);
  ImGui::SliderFloat("Lacunarity", &noise.lacunarity, 1.5f, 3.5f);
  if (ImGui::Button("Regenerate"))
  {
    pendingGeneration = true;
    rocks.invalidate();
  }

  ImGui::SeparatorText("Lighting");
  ImGui::SliderFloat3("Light position", &draw.light.x, -100.f, 100.f);
  ImGui::SliderFloat("Light intensity", &draw.light.w, 0.1f, 20.f);

  ImGui::SeparatorText("Materials");
  ImGui::SliderFloat("Texture tiling", &draw.detail.x, 2.f, 40.f);
  ImGui::SliderFloat("Height detail", &draw.detail.y, 0.f, 2.5f);
  rocks.drawGui();
  ImGui::End();
}

void WorldRenderer::generateMaps(vk::CommandBuffer cmd)
{
  const auto toStorage = [&](etna::Image& image) {
    etna::set_state(cmd, image.get(), vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderStorageWrite,
                    vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
  };
  toStorage(heightMap);
  etna::flush_barriers(cmd);

  auto shader = etna::get_shader_program("terrain_generate");
  auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, heightMap.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
  });
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, generatePipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                         generatePipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
  cmd.pushConstants(generatePipeline.getVkPipelineLayout(),
                    vk::ShaderStageFlagBits::eCompute, 0, sizeof(noise), &noise);
  cmd.dispatch(32, 32, 1);

  const auto toSampled = [&](etna::Image& image) {
    etna::set_state(cmd, image.get(),
                    vk::PipelineStageFlagBits2::eVertexShader |
                      vk::PipelineStageFlagBits2::eFragmentShader,
                    vk::AccessFlagBits2::eShaderSampledRead,
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::ImageAspectFlagBits::eColor);
  };
  toSampled(heightMap);
  etna::flush_barriers(cmd);
  pendingGeneration = false;
}

void WorldRenderer::uploadTextures(vk::CommandBuffer cmd)
{
  const std::array<etna::Image*, 3> atlases = {&albedoAtlas, &roughnessAtlas, &detailHeightAtlas};
  constexpr std::array<uint32_t, 3> bytesPerPixel = {4, 1, 1};

  const auto transition = [&](vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                              vk::PipelineStageFlags2 sourceStage, vk::AccessFlags2 sourceAccess,
                              vk::PipelineStageFlags2 targetStage, vk::AccessFlags2 targetAccess) {
    std::array<vk::ImageMemoryBarrier2, 3> barriers;
    for (size_t i = 0; i < atlases.size(); ++i)
      barriers[i] = vk::ImageMemoryBarrier2{
        .srcStageMask = sourceStage,
        .srcAccessMask = sourceAccess,
        .dstStageMask = targetStage,
        .dstAccessMask = targetAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .image = atlases[i]->get(),
        .subresourceRange = vk::ImageSubresourceRange{
          vk::ImageAspectFlagBits::eColor, 0, kTextureMipLevels, 0, 1},
      };
    cmd.pipelineBarrier2(vk::DependencyInfo{
      .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
      .pImageMemoryBarriers = barriers.data(),
    });
  };

  transition(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
             vk::PipelineStageFlagBits2::eTopOfPipe, {},
             vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);

  for (size_t i = 0; i < atlases.size(); ++i)
  {
    std::array<vk::BufferImageCopy, kTextureMipLevels> regions;
    vk::DeviceSize offset = 0;
    for (uint32_t mip = 0; mip < kTextureMipLevels; ++mip)
    {
      const uint32_t width = kAtlasWidth >> mip;
      const uint32_t height = kAtlasHeight >> mip;
      regions[mip] = vk::BufferImageCopy{
        .bufferOffset = offset,
        .imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, 0, 1},
        .imageExtent = vk::Extent3D{width, height, 1},
      };
      offset += vk::DeviceSize(width) * height * bytesPerPixel[i];
    }
    cmd.copyBufferToImage(textureStaging[i].get(), atlases[i]->get(),
                          vk::ImageLayout::eTransferDstOptimal, regions);
  }

  transition(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
             vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
             vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
  pendingTextureUpload = false;
}

void WorldRenderer::renderWorld(vk::CommandBuffer cmd, vk::Image target, vk::ImageView view, uint32_t frameSlot)
{
  if (pendingTextureUpload) uploadTextures(cmd);
  if (pendingGeneration) generateMaps(cmd);

  auto shader = etna::get_shader_program("terrain_surface");
  auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, heightMap.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{1, albedoAtlas.genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{2, roughnessAtlas.genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{3, detailHeightAtlas.genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
  });
  etna::RenderTargetState renderTargets(
    cmd,
    {{0, 0}, {resolution.x, resolution.y}},
    {{.image = target, .view = view}},
    {.image = depth.get(), .view = depth.getView({})});

  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, drawPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         drawPipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
  cmd.pushConstants(drawPipeline.getVkPipelineLayout(),
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                    0, sizeof(draw), &draw);
  cmd.bindIndexBuffer(indices.get(), 0, vk::IndexType::eUint32);
  cmd.drawIndexed(kIndexCount, 1, 0, 0, 0);
  rocks.render(cmd, heightMap, sampler, draw, frameSlot);
}
