#include "WorldRenderer.hpp"
#include "PipelineDepth.hpp"
#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <cstring>
#include <algorithm>
#include <vector>
#include <span>

namespace
{
constexpr std::array<vk::Format, 2> kFormats = {
  vk::Format::eR8G8B8A8Unorm, vk::Format::eR16G16B16A16Sfloat
};
constexpr vk::PipelineStageFlags2 kDepthStages =
    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
constexpr vk::AccessFlags2 kDepthRead = vk::AccessFlagBits2::eDepthStencilAttachmentRead;

void viewport(vk::CommandBuffer cmd, glm::uvec2 res)
{
  cmd.setViewport(0, {vk::Viewport{0.f, 0.f, float(res.x), float(res.y), 0.f, 1.f}});
  cmd.setScissor(0, {vk::Rect2D{{0, 0}, {res.x, res.y}}});
}

void beginRender(vk::CommandBuffer cmd, glm::uvec2 res,
                 std::span<const vk::ImageView> colorViews, bool clearColors,
                 vk::ImageView depthView = {}, bool clearDepth = false)
{
  std::array<vk::RenderingAttachmentInfo, 2> colors{};
  for (size_t i = 0; i < colorViews.size(); ++i)
  {
    colors[i].imageView = colorViews[i];
    colors[i].imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colors[i].loadOp = clearColors ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eDontCare;
    colors[i].storeOp = vk::AttachmentStoreOp::eStore;
    colors[i].clearValue.color = vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 0.f});
  }
  vk::RenderingAttachmentInfo depth{};
  if (depthView)
  {
    depth.imageView = depthView;
    depth.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    depth.loadOp = clearDepth ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
    depth.storeOp = vk::AttachmentStoreOp::eStore;
    depth.clearValue.depthStencil = vk::ClearDepthStencilValue{1.f, 0};
  }
  vk::RenderingInfo info{};
  info.renderArea = vk::Rect2D{{0, 0}, {res.x, res.y}};
  info.layerCount = 1;
  info.colorAttachmentCount = uint32_t(colorViews.size());
  info.pColorAttachments = colorViews.empty() ? nullptr : colors.data();
  info.pDepthAttachment = depthView ? &depth : nullptr;
  cmd.beginRendering(info);
  viewport(cmd, res);
}

void toSampled(vk::CommandBuffer cmd, const etna::Image& image, vk::ImageAspectFlagBits aspect)
{
  etna::set_state(cmd, image.get(), vk::PipelineStageFlagBits2::eFragmentShader,
                  vk::AccessFlagBits2::eShaderSampledRead,
                  aspect == vk::ImageAspectFlagBits::eDepth ?
                      vk::ImageLayout::eDepthStencilReadOnlyOptimal :
                      vk::ImageLayout::eShaderReadOnlyOptimal,
                  aspect);
}

void toColorAttachment(vk::CommandBuffer cmd, const etna::Image& image)
{
  etna::set_state(cmd, image.get(), vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                  vk::AccessFlagBits2::eColorAttachmentWrite,
                  vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
}

vk::PipelineRasterizationStateCreateInfo rasterState()
{
  return vk::PipelineRasterizationStateCreateInfo{
    .polygonMode = vk::PolygonMode::eFill,
    .cullMode = vk::CullModeFlagBits::eNone,
    .frontFace = vk::FrontFace::eCounterClockwise,
    .lineWidth = 1.f,
  };
}
}

void WorldRenderer::resizeDepth(glm::uvec2 res)
{
  resolution = res;
  auto& ctx = etna::get_context();
  depth = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{res.x, res.y, 1},
    .name = "terrain_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment |
                  vk::ImageUsageFlagBits::eSampled,
  });
  for (uint32_t format = 0; format < 2; ++format)
    for (uint32_t ping = 0; ping < 2; ++ping)
    {
      auto createGImage = [&](const char* name) {
        return ctx.createImage(etna::Image::CreateInfo{
          .extent = vk::Extent3D{res.x, res.y, 1},
          .name = name,
          .format = kFormats[format],
          .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                        vk::ImageUsageFlagBits::eSampled,
        });
      };
      gBuffers[format][ping].albedoAO = createGImage("G0_albedo_AO");
      gBuffers[format][ping].normalRM = createGImage("G1_octNormal_roughness_metalness");
    }
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
  albedoAtlas = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{1024, 256, 1},
    .name = "PBR_albedo_sand_grass_rock_snow",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });
  roughnessAtlas = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{1024, 256, 1},
    .name = "PBR_roughness_sand_grass_rock_snow",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });
  detailHeightAtlas = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{1024, 256, 1},
    .name = "PBR_detail_height_sand_grass_rock_snow",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
  });
  auto createDecalTexture = [&](const char* name) {
    return ctx.createImage(etna::Image::CreateInfo{
      .extent = vk::Extent3D{1024, 256, 1},
      .name = name,
      .format = vk::Format::eR8G8B8A8Unorm,
      .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    });
  };
  decalAlbedo = createDecalTexture("decals_albedo_opacity_atlas");
  decalNormals = createDecalTexture("decals_tangent_normals_atlas");
  decalRoughness = createDecalTexture("decals_roughness_atlas");
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
  for (auto& data : decalGpuBuffers)
    data = ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = vk::DeviceSize(kDecalCount * sizeof(DecalGpu)),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .allocationCreate = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .name = "projective_decal_boxes",
    });
  rocks.allocateResources();
  pendingGeneration = true;
  pendingDecalTextures = true;
}

void WorldRenderer::loadShaders()
{
  rocks.loadShaders();
  etna::create_program("terrain_generate", {DEFERRED_SHADERS_ROOT "generate.comp.spv"});
  etna::create_program("decal_generate", {DEFERRED_SHADERS_ROOT "decal_generate.comp.spv"});
  etna::create_program("terrain_depth", {DEFERRED_SHADERS_ROOT "terrain.vert.spv"});
  etna::create_program("terrain_surface", {DEFERRED_SHADERS_ROOT "terrain.frag.spv",
                                             DEFERRED_SHADERS_ROOT "terrain.vert.spv"});
  etna::create_program("projective_decal", {DEFERRED_SHADERS_ROOT "decal.frag.spv",
                                             DEFERRED_SHADERS_ROOT "fullscreen.vert.spv"});
  etna::create_program("deferred_lighting", {DEFERRED_SHADERS_ROOT "lighting.frag.spv",
                                               DEFERRED_SHADERS_ROOT "fullscreen.vert.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchainFormat)
{
  auto& manager = etna::get_context().getPipelineManager();
  generatePipeline = manager.createComputePipeline("terrain_generate", {});
  generateDecalPipeline = manager.createComputePipeline("decal_generate", {});
  rocks.setupPipelines();

  auto depthInfo = etna::GraphicsPipeline::CreateInfo{
    .rasterizationConfig = rasterState(),
    .fragmentShaderOutput = {.colorAttachmentFormats = {},
                             .depthAttachmentFormat = vk::Format::eD32Sfloat},
  };
  depth_pipeline::configure(depthInfo, true, true, vk::CompareOp::eLess);
  depthPipeline = manager.createGraphicsPipeline("terrain_depth", depthInfo);

  for (uint32_t i = 0; i < 2; ++i)
  {
    auto surfaceInfo = etna::GraphicsPipeline::CreateInfo{
      .rasterizationConfig = rasterState(),
      .fragmentShaderOutput = {.colorAttachmentFormats = {kFormats[i], kFormats[i]},
                               .depthAttachmentFormat = vk::Format::eD32Sfloat},
    };
    depth_pipeline::configure(surfaceInfo, true, false, vk::CompareOp::eEqual);
    geometryPipelines[i] = manager.createGraphicsPipeline("terrain_surface", surfaceInfo);

    auto decalInfo = etna::GraphicsPipeline::CreateInfo{
      .rasterizationConfig = rasterState(),
      .fragmentShaderOutput = {.colorAttachmentFormats = {kFormats[i], kFormats[i]}},
    };
    depth_pipeline::configure(decalInfo, false, false, vk::CompareOp::eAlways);
    decalPipelines[i] = manager.createGraphicsPipeline("projective_decal", decalInfo);
  }
  auto lightingInfo = etna::GraphicsPipeline::CreateInfo{
    .rasterizationConfig = rasterState(),
    .fragmentShaderOutput = {.colorAttachmentFormats = {swapchainFormat}},
  };
  depth_pipeline::configure(lightingInfo, false, false, vk::CompareOp::eAlways);
  lightingPipeline = manager.createGraphicsPipeline("deferred_lighting", lightingInfo);
}

void WorldRenderer::debugInput(const Keyboard&) {}

void WorldRenderer::update(const FramePacket& packet)
{
  const float aspect = static_cast<float>(resolution.x) / static_cast<float>(resolution.y);
  draw.viewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  draw.eye = glm::vec4(packet.mainCam.position, 0.f);
  lighting.invViewProj = glm::inverse(draw.viewProj);
  lighting.eye = draw.eye;
  lighting.light = draw.light;
  lighting.options = glm::vec4(draw.detail.w, draw.levels.w, 0.f, 0.f);
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

  ImGui::SeparatorText("G-buffer and GPU profiling");
  ImGui::Checkbox("Compact G-buffer (RGBA8)", &compactGBuffer);
  const int bytesPerPixel = compactGBuffer ? 8 : 16;
  const double mib = double(resolution.x) * resolution.y * (bytesPerPixel + 4) / (1024.0 * 1024.0);
  ImGui::Text("G-buffer: %d B/px + depth: 4 B/px", bytesPerPixel);
  ImGui::Text("Active attachments: %.2f MiB at %u x %u", mib, resolution.x, resolution.y);

  ImGui::SeparatorText("Projective decals");
  ImGui::Checkbox("Enable decals", &decalsEnabled);
  ImGui::SliderFloat("Decal opacity", &globalDecalOpacity, 0.f, 1.f);
  ImGui::SliderFloat("Normal intensity", &globalDecalNormals, 0.f, 2.f);
  ImGui::SliderFloat("Roughness intensity", &globalDecalRoughness, 0.f, 2.f);
  constexpr const char* titles[kDecalCount] = {"Soil", "Moss", "Crack", "Wet patch", "More moss"};
  for (uint32_t i = 0; i < kDecalCount; ++i)
  {
    ImGui::PushID(int(i));
    auto& d = decals[i];
    if (ImGui::TreeNode(titles[i]))
    {
      ImGui::Checkbox("Enabled", &d.enabled);
      ImGui::SliderFloat("Opacity", &d.opacity, 0.f, 1.f);
      ImGui::SliderFloat("Normal", &d.normalStrength, 0.f, 2.f);
      ImGui::SliderFloat("Roughness", &d.roughnessStrength, 0.f, 2.f);
      ImGui::TreePop();
    }
    ImGui::PopID();
  }
  rocks.drawGui();
  ImGui::End();
}

void WorldRenderer::writeDecalBuffer(uint32_t slot)
{
  std::array<DecalGpu, kDecalCount> gpu{};
  for (uint32_t i = 0; i < kDecalCount; ++i)
  {
    const auto& d = decals[i];
    glm::mat4 m = glm::translate(glm::mat4(1.f), d.center);
    m = glm::rotate(m, glm::radians(d.yawDeg), glm::vec3(0.f, 1.f, 0.f));
    m = glm::scale(m, glm::max(d.halfSize, glm::vec3(0.1f)));
    gpu[i].boxToWorld = m;
    gpu[i].worldToBox = glm::inverse(m);
    gpu[i].tint = glm::vec4(d.tint, 1.f);
    gpu[i].controls = glm::vec4(d.opacity * (d.enabled ? 1.f : 0.f),
                               d.normalStrength, d.roughnessStrength, float(d.texture));
  }
  std::memcpy(decalGpuBuffers[slot].data(), gpu.data(), sizeof(gpu));
}

void WorldRenderer::generateMaps(vk::CommandBuffer cmd)
{
  const auto toStorage = [&](etna::Image& image) {
    etna::set_state(cmd, image.get(), vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderStorageWrite,
                    vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
  };
  toStorage(heightMap);
  toStorage(albedoAtlas);
  toStorage(roughnessAtlas);
  toStorage(detailHeightAtlas);
  etna::flush_barriers(cmd);

  auto shader = etna::get_shader_program("terrain_generate");
  auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, heightMap.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
    etna::Binding{1, albedoAtlas.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
    etna::Binding{2, roughnessAtlas.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
    etna::Binding{3, detailHeightAtlas.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
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
  toSampled(albedoAtlas);
  toSampled(roughnessAtlas);
  toSampled(detailHeightAtlas);
  etna::flush_barriers(cmd);
  pendingGeneration = false;
}

void WorldRenderer::generateDecalTextures(vk::CommandBuffer cmd)
{
  for (etna::Image* image : {&decalAlbedo, &decalNormals, &decalRoughness})
    etna::set_state(cmd, image->get(), vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderStorageWrite,
                    vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd);
  auto shader = etna::get_shader_program("decal_generate");
  auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, decalAlbedo.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
    etna::Binding{1, decalNormals.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
    etna::Binding{2, decalRoughness.genBinding(vk::Sampler{}, vk::ImageLayout::eGeneral)},
  });
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, generateDecalPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, generateDecalPipeline.getVkPipelineLayout(),
                         0, {set.getVkSet()}, {});
  cmd.dispatch(64, 16, 1);
  for (etna::Image* image : {&decalAlbedo, &decalNormals, &decalRoughness})
    toSampled(cmd, *image, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd);
  pendingDecalTextures = false;
}

void WorldRenderer::drawTerrain(vk::CommandBuffer cmd, bool depthOnly)
{
  auto shader = etna::get_shader_program(depthOnly ? "terrain_depth" : "terrain_surface");
  auto set = depthOnly ? etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, heightMap.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
  }) : etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, heightMap.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{1, albedoAtlas.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{2, roughnessAtlas.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{3, detailHeightAtlas.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
  });
  auto& pipeline = depthOnly ? depthPipeline : geometryPipelines[compactGBuffer ? 0 : 1];
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(),
                         0, {set.getVkSet()}, {});
  vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex;
  if (!depthOnly) stages |= vk::ShaderStageFlagBits::eFragment;
  cmd.pushConstants(pipeline.getVkPipelineLayout(), stages, 0, sizeof(draw), &draw);
  cmd.bindIndexBuffer(indices.get(), 0, vk::IndexType::eUint32);
  cmd.drawIndexed(kIndexCount, 1, 0, 0, 0);
}

void WorldRenderer::renderWorld(vk::CommandBuffer cmd, vk::Image target, vk::ImageView view, uint32_t frameSlot)
{
  if (pendingGeneration) generateMaps(cmd);
  if (pendingDecalTextures) generateDecalTextures(cmd);
  frameSlot %= 2;
  writeDecalBuffer(frameSlot);
  const uint32_t fmt = compactGBuffer ? 0 : 1;
  const vk::ImageView depthView = depth.getView({});

  {
    ETNA_PROFILE_GPU(cmd, DepthPrepass);
    etna::set_state(cmd, depth.get(), kDepthStages, kDepthRead |
                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                    vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageAspectFlagBits::eDepth);
    etna::flush_barriers(cmd);
    beginRender(cmd, resolution, {}, false, depthView, true);
    drawTerrain(cmd, true);
    rocks.render(cmd, heightMap, sampler, draw, frameSlot, true, fmt);
    cmd.endRendering();
  }

  {
    ETNA_PROFILE_GPU(cmd, GBufferPass);
    auto& gb = gBuffers[fmt][0];
    toColorAttachment(cmd, gb.albedoAO);
    toColorAttachment(cmd, gb.normalRM);
    etna::set_state(cmd, depth.get(), kDepthStages, kDepthRead,
                    vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageAspectFlagBits::eDepth);
    etna::flush_barriers(cmd);
    const std::array<vk::ImageView,2> colors{gb.albedoAO.getView({}), gb.normalRM.getView({})};
    beginRender(cmd, resolution, colors, true, depthView);
    drawTerrain(cmd, false);
    rocks.render(cmd, heightMap, sampler, draw, frameSlot, false, fmt);
    cmd.endRendering();
  }

  toSampled(cmd, depth, vk::ImageAspectFlagBits::eDepth);
  toSampled(cmd, gBuffers[fmt][0].albedoAO, vk::ImageAspectFlagBits::eColor);
  toSampled(cmd, gBuffers[fmt][0].normalRM, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd);
  uint32_t current = 0;

  if (decalsEnabled && globalDecalOpacity > 0.f)
  {
    ETNA_PROFILE_GPU(cmd, DecalPass);
    auto shader = etna::get_shader_program("projective_decal");
    for (uint32_t i = 0; i < kDecalCount; ++i)
    {
      if (!decals[i].enabled || decals[i].opacity <= 0.f) continue;
      auto& source = gBuffers[fmt][current];
      const uint32_t destinationIndex = current ^ 1u;
      auto& destination = gBuffers[fmt][destinationIndex];
      toColorAttachment(cmd, destination.albedoAO);
      toColorAttachment(cmd, destination.normalRM);
      etna::flush_barriers(cmd);
      const std::array<vk::ImageView,2> colors{
        destination.albedoAO.getView({}), destination.normalRM.getView({})};
      beginRender(cmd, resolution, colors, false);
      auto descriptor = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
        etna::Binding{0, source.albedoAO.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{1, source.normalRM.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{2, depth.genBinding(sampler.get(), vk::ImageLayout::eDepthStencilReadOnlyOptimal)},
        etna::Binding{3, decalAlbedo.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{4, decalNormals.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{5, decalRoughness.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        etna::Binding{6, decalGpuBuffers[frameSlot].genBinding()},
      });
      auto& p = decalPipelines[fmt];
      cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, p.getVkPipeline());
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, p.getVkPipelineLayout(),
                             0, {descriptor.getVkSet()}, {});
      const DecalPush push{lighting.invViewProj,
                           glm::vec4(float(resolution.x), float(resolution.y), 0.f, 0.f),
                           glm::vec4(float(i), globalDecalOpacity, globalDecalNormals, globalDecalRoughness)};
      cmd.pushConstants(p.getVkPipelineLayout(), vk::ShaderStageFlagBits::eFragment,
                        0, sizeof(push), &push);
      cmd.draw(3, 1, 0, 0);
      cmd.endRendering();
      toSampled(cmd, destination.albedoAO, vk::ImageAspectFlagBits::eColor);
      toSampled(cmd, destination.normalRM, vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(cmd);
      current = destinationIndex;
    }
  }

  {
    ETNA_PROFILE_GPU(cmd, DeferredLightPass);
    etna::set_state(cmd, target, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlagBits2::eColorAttachmentWrite,
                    vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(cmd);
    const std::array<vk::ImageView,1> colors{view};
    beginRender(cmd, resolution, colors, true);
    auto shader = etna::get_shader_program("deferred_lighting");
    auto descriptor = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, gBuffers[fmt][current].albedoAO.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{1, gBuffers[fmt][current].normalRM.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{2, depth.genBinding(sampler.get(), vk::ImageLayout::eDepthStencilReadOnlyOptimal)},
    });
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipelineLayout(),
                           0, {descriptor.getVkSet()}, {});

    lighting.light = draw.light;
    lighting.options = glm::vec4(draw.detail.w, draw.levels.w, 0.f, 0.f);
    cmd.pushConstants(lightingPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eFragment,
                      0, sizeof(lighting), &lighting);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
  }
}
