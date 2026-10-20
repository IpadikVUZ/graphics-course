#pragma once

#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/Image.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  glm::uvec2 resolution;
  bool useVsync;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;

  etna::Image bufferAImage;
  etna::Image bufferBImage;
  etna::Image resultImage;

  vk::ImageView bufferAView;
  vk::ImageView bufferBView;
  vk::ImageView resultView;
  vk::Sampler sampler;
  vk::DescriptorSetLayout descriptorLayout;
  vk::DescriptorPool descriptorPool;
  vk::DescriptorSet descriptorSet;
  vk::PipelineLayout pipelineLayout;
  vk::Pipeline bufferAPipeline;
  vk::Pipeline bufferBPipeline;
  vk::Pipeline pipeline;
};
