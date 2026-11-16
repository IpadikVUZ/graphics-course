#pragma once
#include <etna/GraphicsPipeline.hpp>

namespace depth_pipeline
{
inline void configure(etna::GraphicsPipeline::CreateInfo& info, bool test, bool write, vk::CompareOp comparison)
{
  info.depthConfig.depthTestEnable = test ? vk::True : vk::False;
  info.depthConfig.depthWriteEnable = write ? vk::True : vk::False;
  info.depthConfig.depthCompareOp = comparison;
}
}
