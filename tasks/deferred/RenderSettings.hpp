#pragma once
#include <glm/glm.hpp>

struct NoiseSettings
{
  float scale = 3.2f;
  float persistence = 0.52f;
  float lacunarity = 2.05f;
  float heightScale = 1.f;
  float islandRadius = 0.96f;
  int octaves = 6;
  int seed = 17;
  float shore = 0.63f;
};

struct DrawSettings
{
  glm::mat4 viewProj{1.0f};
  glm::vec4 light{25.f, 45.f, 20.f, 4.5f};
  glm::vec4 eye{0.f, 30.f, 70.f, 0.f};
  glm::vec4 detail{14.f, 0.7f, 0.f, 0.12f};
  glm::vec4 levels{1.6f, 6.8f, 9.6f, 1.0f};
};

struct LightingSettings
{
  glm::mat4 invViewProj{1.f};
  glm::vec4 light{25.f, 45.f, 20.f, 4.5f};
  glm::vec4 eye{0.f, 30.f, 70.f, 0.f};

  glm::vec4 options{0.12f, 1.f, 0.f, 0.f};
};

struct DecalGpu
{
  glm::mat4 worldToBox{1.f};
  glm::mat4 boxToWorld{1.f};
  glm::vec4 tint{1.f};

  glm::vec4 controls{1.f, 1.f, 1.f, 0.f};
};

struct DecalPush
{
  glm::mat4 invViewProj{1.f};
  glm::vec4 screenSize{1280.f, 720.f, 0.f, 0.f};

  glm::vec4 settings{0.f, 1.f, 1.f, 1.f};
};

static_assert(sizeof(NoiseSettings) == 32);
static_assert(sizeof(DrawSettings) == 128);
static_assert(sizeof(LightingSettings) == 112);
static_assert(sizeof(DecalGpu) == 160);
static_assert(sizeof(DecalPush) == 96);
