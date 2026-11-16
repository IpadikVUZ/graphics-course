#version 450

layout(set = 0, binding = 3) uniform sampler2D albedoMap;
layout(set = 0, binding = 4) uniform sampler2D normalMap;
layout(set = 0, binding = 5) uniform sampler2D armMap;

layout(location = 0) in vec3 iWorld;
layout(location = 1) in vec3 iNormal;
layout(location = 2) in vec4 iTangent;
layout(location = 3) in vec2 iUV;
layout(location = 4) in vec4 iColor;
layout(location = 0) out vec4 outAlbedoAO;
layout(location = 1) out vec4 outNormalRM;

vec2 octEncode(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 p = n.xy;
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(p, vec2(0.0)));
    return p * 0.5 + 0.5;
}

void main() {
    vec3 baseN = normalize(iNormal);
    vec3 T = normalize(iTangent.xyz - baseN * dot(baseN, iTangent.xyz));
    vec3 B = normalize(cross(baseN, T)) * iTangent.w;
    vec3 normalTS = texture(normalMap, iUV).xyz * 2.0 - 1.0;
    vec3 N = normalize(mat3(T, B, baseN) * normalTS);
    if (!gl_FrontFacing) N = -N;

    vec3 albedo = texture(albedoMap, iUV).rgb * iColor.rgb;
    vec3 arm = texture(armMap, iUV).rgb;
    float ao = clamp(arm.r, 0.0, 1.0);
    float roughness = clamp(arm.g, 0.08, 1.0);
    float metalness = clamp(arm.b, 0.0, 1.0);

    outAlbedoAO = vec4(albedo, ao);
    outNormalRM = vec4(octEncode(N), roughness, metalness);
}
