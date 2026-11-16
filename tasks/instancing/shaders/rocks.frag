#version 450

layout(set = 0, binding = 3) uniform sampler2D albedoMap;
layout(set = 0, binding = 4) uniform sampler2D normalMap;
layout(set = 0, binding = 5) uniform sampler2D armMap;

layout(push_constant) uniform DrawSettings {
    mat4 viewProj;
    vec4 light;
    vec4 eye;
    vec4 detail;
    vec4 levels;
} params;

layout(location = 0) in vec3 iWorld;
layout(location = 1) in vec3 iNormal;
layout(location = 2) in vec4 iTangent;
layout(location = 3) in vec2 iUV;
layout(location = 4) in vec4 iColor;
layout(location = 0) out vec4 outColor;

const float PI = 3.141592653589793;

float distributionGGX(float NoH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 0.00001);
}

float smithG1(float NoX, float k) {
    return NoX / max(NoX * (1.0 - k) + k, 0.00001);
}

vec3 fresnelSchlick(float VoH, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - VoH, 5.0);
}

void main() {
    vec3 baseN = normalize(iNormal);
    vec3 T = normalize(iTangent.xyz - baseN * dot(baseN, iTangent.xyz));
    vec3 B = normalize(cross(baseN, T)) * iTangent.w;
    mat3 tbn = mat3(T, B, baseN);

    vec3 normalTS = texture(normalMap, iUV).xyz * 2.0 - 1.0;
    vec3 N = normalize(tbn * normalTS);
    if (!gl_FrontFacing) N = -N;

    vec3 texAlbedo = texture(albedoMap, iUV).rgb;
    texAlbedo *= iColor.rgb;
    vec3 arm = texture(armMap, iUV).rgb;
    float ao = arm.r;
    float roughness = clamp(arm.g, 0.08, 1.0);
    float metallic = clamp(arm.b, 0.0, 1.0);

    vec3 V = normalize(params.eye.xyz - iWorld);
    vec3 L = normalize(params.light.xyz - iWorld);
    vec3 H = normalize(V + L);
    float NoL = max(dot(N, L), 0.0);
    float NoV = max(dot(N, V), 0.0001);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);

    vec3 f0 = mix(vec3(0.04), texAlbedo, metallic);
    vec3 F = fresnelSchlick(VoH, f0);
    float D = distributionGGX(NoH, roughness);
    float k = pow(roughness + 1.0, 2.0) / 8.0;
    float G = smithG1(NoL, k) * smithG1(NoV, k);
    vec3 specular = (D * G * F) / max(4.0 * NoL * NoV, 0.001);
    vec3 diffuse = (1.0 - F) * (1.0 - metallic) * texAlbedo / PI;

    float lightDistance = length(params.light.xyz - iWorld);
    float attenuation = 1.0 / (1.0 + 0.0015 * lightDistance * lightDistance);
    vec3 color = (diffuse + specular) * NoL * params.light.w * attenuation;
    color += texAlbedo * (0.18 * ao + params.detail.w * 0.5 * ao);
    color *= params.levels.w;
    color = color / (color + vec3(1.0));
    outColor = vec4(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
