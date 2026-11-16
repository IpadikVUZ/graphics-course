#version 450

struct Vertex {
    vec4 position;
    vec4 normal;
    vec4 tangent;
    vec4 uv;
};

layout(set = 0, binding = 0, std430) readonly buffer RockVertices {
    Vertex vertices[];
};

struct Instance {
    mat4 transform;
    vec4 color;
};

layout(set = 0, binding = 1, std430) readonly buffer RockInstances {
    Instance instances[];
};

layout(set = 0, binding = 2) uniform sampler2D heightMap;

layout(push_constant) uniform DrawSettings {
    mat4 viewProj;
    vec4 light;
    vec4 eye;
    vec4 detail;
    vec4 levels;
} params;

layout(location = 0) out vec3 oWorld;
layout(location = 1) out vec3 oNormal;
layout(location = 2) out vec4 oTangent;
layout(location = 3) out vec2 oUV;
layout(location = 4) out vec4 oColor;

void main() {
    Vertex v = vertices[gl_VertexIndex];
    Instance instance = instances[gl_InstanceIndex];
    vec3 world = (instance.transform * vec4(v.position.xyz, 1.0)).xyz;
    vec2 islandUV = clamp(instance.transform[3].xz / 100.0 + 0.5, vec2(0.0), vec2(1.0));
    world.y += max(textureLod(heightMap, islandUV, 0.0).r, 0.0);

    mat3 normalMatrix = mat3(instance.transform);
    oWorld = world;
    oNormal = normalize(normalMatrix * v.normal.xyz);
    oTangent = vec4(normalize(normalMatrix * v.tangent.xyz), v.tangent.w);
    oUV = v.uv.xy;
    oColor = instance.color;
    gl_Position = params.viewProj * vec4(world, 1.0);
}
