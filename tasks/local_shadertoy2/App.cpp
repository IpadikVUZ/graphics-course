#include "App.hpp"

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace
{

vk::ShaderModule loadShader(vk::Device device, const char* path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Cannot open compiled shader: " + std::string(path));

  auto size = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> code(size / sizeof(uint32_t));
  file.seekg(0);
  if (size % sizeof(uint32_t) != 0 || !file.read(reinterpret_cast<char*>(code.data()), size))
    throw std::runtime_error("Invalid SPIR-V shader: " + std::string(path));

  vk::ShaderModuleCreateInfo info{};
  info.codeSize = size;
  info.pCode = code.data();
  vk::ShaderModule module;
  ETNA_CHECK_VK_RESULT(device.createShaderModule(&info, nullptr, &module));
  return module;
}

vk::Pipeline createPipeline(vk::Device device, vk::PipelineLayout layout,
                            vk::ShaderModule vertex, vk::ShaderModule fragment)
{
  vk::PipelineShaderStageCreateInfo stages[2]{};
  stages[0].stage = vk::ShaderStageFlagBits::eVertex;
  stages[0].module = vertex;
  stages[0].pName = "main";
  stages[1].stage = vk::ShaderStageFlagBits::eFragment;
  stages[1].module = fragment;
  stages[1].pName = "main";

  vk::PipelineVertexInputStateCreateInfo vertexInput{};
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

  vk::PipelineViewportStateCreateInfo viewportState{};
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  vk::PipelineRasterizationStateCreateInfo rasterization{};
  rasterization.polygonMode = vk::PolygonMode::eFill;
  rasterization.cullMode = vk::CullModeFlagBits::eNone;
  rasterization.frontFace = vk::FrontFace::eCounterClockwise;
  rasterization.lineWidth = 1.0f;

  vk::PipelineMultisampleStateCreateInfo multisample{};
  multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

  vk::PipelineColorBlendAttachmentState colorAttachment{};
  colorAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                   vk::ColorComponentFlagBits::eG |
                                   vk::ColorComponentFlagBits::eB |
                                   vk::ColorComponentFlagBits::eA;
  vk::PipelineColorBlendStateCreateInfo colorBlend{};
  colorBlend.attachmentCount = 1;
  colorBlend.pAttachments = &colorAttachment;

  vk::DynamicState dynamicStates[] = {
    vk::DynamicState::eViewport, vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.dynamicStateCount = 2;
  dynamicState.pDynamicStates = dynamicStates;

  vk::Format format = vk::Format::eR8G8B8A8Unorm;
  vk::PipelineRenderingCreateInfo dynamicRendering{};
  dynamicRendering.colorAttachmentCount = 1;
  dynamicRendering.pColorAttachmentFormats = &format;

  vk::GraphicsPipelineCreateInfo info{};
  info.pNext = &dynamicRendering;
  info.stageCount = 2;
  info.pStages = stages;
  info.pVertexInputState = &vertexInput;
  info.pInputAssemblyState = &inputAssembly;
  info.pViewportState = &viewportState;
  info.pRasterizationState = &rasterization;
  info.pMultisampleState = &multisample;
  info.pColorBlendState = &colorBlend;
  info.pDynamicState = &dynamicState;
  info.layout = layout;

  vk::Pipeline pipeline;
  ETNA_CHECK_VK_RESULT(device.createGraphicsPipelines({}, 1, &info, nullptr, &pipeline));
  return pipeline;
}

void drawPass(vk::CommandBuffer cmd, vk::ImageView imageView, vk::Pipeline pipeline,
              vk::PipelineLayout layout, glm::uvec2 resolution, float time,
              vk::DescriptorSet descriptorSet = {})
{
  vk::RenderingAttachmentInfo attachment{};
  attachment.imageView = imageView;
  attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  attachment.loadOp = vk::AttachmentLoadOp::eDontCare;
  attachment.storeOp = vk::AttachmentStoreOp::eStore;

  vk::RenderingInfo rendering{};
  rendering.renderArea = vk::Rect2D{{0, 0}, {resolution.x, resolution.y}};
  rendering.layerCount = 1;
  rendering.colorAttachmentCount = 1;
  rendering.pColorAttachments = &attachment;

  cmd.beginRendering(rendering);
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
  vk::Viewport viewport{0.0f, 0.0f, float(resolution.x), float(resolution.y), 0.0f, 1.0f};
  vk::Rect2D scissor{{0, 0}, {resolution.x, resolution.y}};
  cmd.setViewport(0, 1, &viewport);
  cmd.setScissor(0, 1, &scissor);
  cmd.pushConstants(layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(float), &time);
  if (descriptorSet)
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, &descriptorSet, 0, nullptr);
  cmd.draw(3, 1, 0, 0);
  cmd.endRendering();
}

}

App::App()
  : resolution{1280, 720}
  , useVsync{true}
{
  // First, we need to initialize Vulkan, which is not trivial because
  // extensions are required for just about anything.
  {
    // GLFW tells us which extensions it needs to present frames to the OS window.
    // Actually rendering anything to a screen is optional in Vulkan, you can
    // alternatively save rendered frames into files, send them over network, etc.
    // Instance extensions do not depend on the actual GPU, only on the OS.
    auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();

    std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};

    // We also need the swapchain device extension to get access to the OS
    // window from inside of Vulkan on the GPU.
    // Device extensions require HW support from the GPU.
    // Generally, in Vulkan, we call the GPU a "device" and the CPU/OS combination a "host."
    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Etna does all of the Vulkan initialization heavy lifting.
    // You can skip figuring out how it works for now.
    etna::initialize(etna::InitParams{
      .applicationName = "Local Shadertoy",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .instanceExtensions = instanceExtensions,
      .deviceExtensions = deviceExtensions,
      // Replace with an index if etna detects your preferred GPU incorrectly
      .physicalDeviceIndexOverride = {},
      .numFramesInFlight = 1,
    });
  }

  // Next, we need a magical Etna helper to send commands to the GPU.
  // How it is actually performed is not trivial, but we can skip this for now.
  commandManager = etna::get_context().createPerFrameCmdMgr();

  // Now we can create an OS window
  osWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = resolution,
  });

  // But we also need to hook the OS window up to Vulkan manually!
  {
    // First, we ask GLFW to provide a "surface" for the window,
    // which is an opaque description of the area where we can actually render.
    auto surface = osWindow->createVkSurface(etna::get_context().getInstance());

    // Then we pass it to Etna to do the complicated work for us
    vkWindow = etna::get_context().createWindow(etna::Window::CreateInfo{
      .surface = std::move(surface),
    });

    // And finally ask Etna to create the actual swapchain so that we can
    // get (different) images each frame to render stuff into.
    // Here, we do not support window resizing, so we only need to call this once.
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });

    // Technically, Vulkan might fail to initialize a swapchain with the requested
    // resolution and pick a different one. This, however, does not occur on platforms
    // we support. Still, it's better to follow the "intended" path.
    resolution = {w, h};
  }

  auto& context = etna::get_context();
  auto device = context.getDevice();

  bufferAImage = context.createImage(etna::Image::CreateInfo{
    .extent = {resolution.x, resolution.y, 1},
    .name = "bufferA",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  bufferBImage = context.createImage(etna::Image::CreateInfo{
    .extent = {resolution.x, resolution.y, 1},
    .name = "bufferB",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  resultImage = context.createImage(etna::Image::CreateInfo{
    .extent = {resolution.x, resolution.y, 1},
    .name = "result",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
  });

  vk::ImageViewCreateInfo viewInfo{};
  viewInfo.viewType = vk::ImageViewType::e2D;
  viewInfo.format = vk::Format::eR8G8B8A8Unorm;
  viewInfo.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
  viewInfo.image = bufferAImage.get();
  ETNA_CHECK_VK_RESULT(device.createImageView(&viewInfo, nullptr, &bufferAView));
  viewInfo.image = bufferBImage.get();
  ETNA_CHECK_VK_RESULT(device.createImageView(&viewInfo, nullptr, &bufferBView));
  viewInfo.image = resultImage.get();
  ETNA_CHECK_VK_RESULT(device.createImageView(&viewInfo, nullptr, &resultView));

  vk::SamplerCreateInfo samplerInfo{};
  samplerInfo.magFilter = vk::Filter::eLinear;
  samplerInfo.minFilter = vk::Filter::eLinear;
  samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
  samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
  samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
  samplerInfo.maxLod = 0.0f;
  ETNA_CHECK_VK_RESULT(device.createSampler(&samplerInfo, nullptr, &sampler));

  vk::DescriptorSetLayoutBinding bindings[2]{};
  for (uint32_t i = 0; i < 2; ++i)
  {
    bindings[i].binding = i;
    bindings[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = vk::ShaderStageFlagBits::eFragment;
  }
  vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
  descriptorLayoutInfo.bindingCount = 2;
  descriptorLayoutInfo.pBindings = bindings;
  ETNA_CHECK_VK_RESULT(device.createDescriptorSetLayout(&descriptorLayoutInfo, nullptr, &descriptorLayout));

  vk::PushConstantRange pushRange{};
  pushRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
  pushRange.offset = 0;
  pushRange.size = sizeof(float);
  vk::PipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &descriptorLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushRange;
  ETNA_CHECK_VK_RESULT(device.createPipelineLayout(&layoutInfo, nullptr, &pipelineLayout));

  vk::DescriptorPoolSize poolSize{vk::DescriptorType::eCombinedImageSampler, 2};
  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  ETNA_CHECK_VK_RESULT(device.createDescriptorPool(&poolInfo, nullptr, &descriptorPool));

  vk::DescriptorSetAllocateInfo allocation{};
  allocation.descriptorPool = descriptorPool;
  allocation.descriptorSetCount = 1;
  allocation.pSetLayouts = &descriptorLayout;
  ETNA_CHECK_VK_RESULT(device.allocateDescriptorSets(&allocation, &descriptorSet));

  vk::DescriptorImageInfo descriptors[2] = {
    {sampler, bufferAView, vk::ImageLayout::eShaderReadOnlyOptimal},
    {sampler, bufferBView, vk::ImageLayout::eShaderReadOnlyOptimal}
  };
  vk::WriteDescriptorSet writes[2]{};
  for (uint32_t i = 0; i < 2; ++i)
  {
    writes[i].dstSet = descriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[i].pImageInfo = &descriptors[i];
  }
  device.updateDescriptorSets(2, writes, 0, nullptr);

  auto vertex = loadShader(device, LOCAL_SHADERTOY2_SHADERS_ROOT "fullscreen.vert.spv");
  auto bufferA = loadShader(device, LOCAL_SHADERTOY2_SHADERS_ROOT "buffer_a.frag.spv");
  auto bufferB = loadShader(device, LOCAL_SHADERTOY2_SHADERS_ROOT "buffer_b.frag.spv");
  auto fragment = loadShader(device, LOCAL_SHADERTOY2_SHADERS_ROOT "toy.frag.spv");

  bufferAPipeline = createPipeline(device, pipelineLayout, vertex, bufferA);
  bufferBPipeline = createPipeline(device, pipelineLayout, vertex, bufferB);
  pipeline = createPipeline(device, pipelineLayout, vertex, fragment);

  device.destroyShaderModule(vertex);
  device.destroyShaderModule(bufferA);
  device.destroyShaderModule(bufferB);
  device.destroyShaderModule(fragment);
}

App::~App()
{
  auto device = etna::get_context().getDevice();
  ETNA_CHECK_VK_RESULT(device.waitIdle());
  device.destroyPipeline(bufferAPipeline);
  device.destroyPipeline(bufferBPipeline);
  device.destroyPipeline(pipeline);
  device.destroyPipelineLayout(pipelineLayout);
  device.destroyDescriptorPool(descriptorPool);
  device.destroyDescriptorSetLayout(descriptorLayout);
  device.destroySampler(sampler);
  device.destroyImageView(bufferAView);
  device.destroyImageView(bufferBView);
  device.destroyImageView(resultView);
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();

    drawFrame();
  }

  // We need to wait for the GPU to execute the last frame before destroying
  // all resources and closing the application.
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::drawFrame()
{
  // First, get a command buffer to write GPU commands into.
  auto currentCmdBuf = commandManager->acquireNext();

  // Next, tell Etna that we are going to start processing the next frame.
  etna::begin_frame();

  // And now get the image we should be rendering the picture into.
  auto nextSwapchainImage = vkWindow->acquireNext();

  // When window is minimized, we can't render anything in Windows
  // because it kills the swapchain, so we skip frames in this case.
  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem, backbufferReadyForPresentSem] =
      *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      // First of all, we need to "initialize" th "backbuffer", aka the current swapchain
      // image, into a state that is appropriate for us working with it. The initial state
      // is considered to be "undefined" (aka "I contain trash memory"), by the way.
      // "Transfer" in vulkanese means "copy or blit".
      // Note that Etna sometimes calls this for you to make life simpler, read Etna's code!
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        // We are going to use the texture at the transfer stage...
        vk::PipelineStageFlagBits2::eTransfer,
        // ...to transfer-write stuff into it...
        vk::AccessFlagBits2::eTransferWrite,
        // ...and want it to have the appropriate layout.
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);
      // The set_state doesn't actually record any commands, they are deferred to
      // the moment you call flush_barriers.
      // As with set_state, Etna sometimes flushes on it's own.
      // Usually, flushes should be placed before "action", i.e. compute dispatches
      // and blit/copy operations.
      etna::flush_barriers(currentCmdBuf);

      static const auto startTime = std::chrono::steady_clock::now();
      float time = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

      etna::set_state(
        currentCmdBuf,
        bufferAImage.get(),
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(currentCmdBuf);
      drawPass(currentCmdBuf, bufferAView, bufferAPipeline, pipelineLayout, resolution, time);

      etna::set_state(
        currentCmdBuf,
        bufferBImage.get(),
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(currentCmdBuf);
      drawPass(currentCmdBuf, bufferBView, bufferBPipeline, pipelineLayout, resolution, time);

      etna::set_state(
        currentCmdBuf,
        bufferAImage.get(),
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::set_state(
        currentCmdBuf,
        bufferBImage.get(),
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::set_state(
        currentCmdBuf,
        resultImage.get(),
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(currentCmdBuf);
      drawPass(currentCmdBuf, resultView, pipeline, pipelineLayout, resolution, time, descriptorSet);

      etna::set_state(
        currentCmdBuf,
        resultImage.get(),
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferRead,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(currentCmdBuf);

      vk::ImageBlit region{};
      region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
      region.srcOffsets[1] = {
        static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1};
      region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
      region.dstOffsets[1] = {
        static_cast<int32_t>(resolution.x), static_cast<int32_t>(resolution.y), 1};

      currentCmdBuf.blitImage(
        resultImage.get(),
        vk::ImageLayout::eTransferSrcOptimal,
        backbuffer,
        vk::ImageLayout::eTransferDstOptimal,
        1,
        &region,
        vk::Filter::eNearest);

      // At the end of "rendering", we are required to change how the pixels of the
      // swpchain image are laid out in memory to something that is appropriate
      // for presenting to the window (while preserving the content of the pixels!).
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        // This looks weird, but is correct. Ask about it later.
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);
      // And of course flush the layout transition.
      etna::flush_barriers(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    // We are done recording GPU commands now and we can send them to be executed by the GPU.
    // Note that the GPU won't start executing our commands before the backbufferAvailableSem
    // semaphore is signalled, which will happen when the OS says that the next swapchain image
    // is ready, and the result image will be ready for present after backbufferReadyForPresent
    // is signalled by GPU
    auto renderingDone = commandManager->submit(
      std::move(currentCmdBuf),
      std::move(backbufferAvailableSem),
      std::move(backbufferReadyForPresentSem));

    // Finally, present the backbuffer the screen, but only after the GPU tells the OS
    // that it is done executing the command buffer via the renderingDone semaphore.
    const bool presented = vkWindow->present(std::move(renderingDone), backbufferView);

    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  // After a window us un-minimized, we need to restore the swapchain to continue rendering.
  if (!nextSwapchainImage && osWindow->getResolution() != glm::uvec2{0, 0})
  {
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });
    ETNA_VERIFY((resolution == glm::uvec2{w, h}));
  }
}
