#version 450
layout(set=0, binding=0) uniform sampler2D gAlbedoAO;
layout(set=0, binding=1) uniform sampler2D gNormalRM;
layout(set=0, binding=2) uniform sampler2D gDepth;
layout(set=0, binding=3) uniform samplerCube skyCubemap;

layout(push_constant) uniform LightingSettings {
    mat4 invViewProj;
    vec4 light;
    vec4 eye;
    vec4 options;
} params;
layout(location=0) out vec4 outColor;
const float PI = 3.141592653589793;

vec3 octDecode(vec2 e) {
    vec2 f = e * 2.0 - 1.0;
    vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}
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
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(gDepth, 0);
    float z = texelFetch(gDepth, p, 0).r;
    if (z >= 0.999999) {
        vec2 uv = (vec2(p) + 0.5) / vec2(size);
        vec4 worldFar = params.invViewProj * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
        vec3 rayDir = normalize(worldFar.xyz - params.eye.xyz * worldFar.w);
        outColor = vec4(max(textureLod(skyCubemap, rayDir, 0.0).rgb * params.options.y, vec3(0.0)), 1.0);
        return;
    }
    vec4 g0 = texelFetch(gAlbedoAO, p, 0);
    vec4 g1 = texelFetch(gNormalRM, p, 0);
    vec2 uv = (vec2(p) + 0.5) / vec2(size);
    vec4 h = params.invViewProj * vec4(uv * 2.0 - 1.0, z, 1.0);
    vec3 world = h.xyz / h.w;

    vec3 albedo = g0.rgb;
    float ao = g0.a;
    vec3 N = octDecode(g1.xy);
    float roughness = clamp(g1.z, 0.06, 1.0);
    float metalness = clamp(g1.w, 0.0, 1.0);

    vec3 V = normalize(params.eye.xyz - world);
    vec3 L = normalize(params.light.xyz);
    vec3 H = normalize(V + L);
    float NoL = max(dot(N, L), 0.0);
    float NoV = max(dot(N, V), 0.0001);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metalness);
    vec3 F = fresnelSchlick(VoH, F0);
    float D = distributionGGX(NoH, roughness);
    float k = pow(roughness + 1.0, 2.0) / 8.0;
    float G = smithG1(NoL, k) * smithG1(NoV, k);
    vec3 specular = (D * G * F) / max(4.0 * NoL * NoV, 0.001);
    vec3 diffuse = (vec3(1.0) - F) * (1.0 - metalness) * albedo / PI;
    vec3 direct = (diffuse + specular) * NoL * params.light.w;

    vec3 zenithSky = textureLod(skyCubemap, vec3(0.0, 1.0, 0.0), 0.0).rgb;
    vec3 upperHemisphere = mix(vec3(0.78, 0.89, 1.05), zenithSky * 1.25, 0.15);
    vec3 lowerHemisphere = vec3(0.49, 0.45, 0.40);
    vec3 hemiLight = mix(lowerHemisphere, upperHemisphere,
                         clamp(N.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 ambient = albedo * (1.0 - metalness) * ao * hemiLight * params.options.x * 1.85;
    vec3 ambientSpec = F0 * mix(vec3(0.05), upperHemisphere * 0.16,
                               1.0 - roughness) * ao;

    vec3 color = direct + ambient + ambientSpec;
    color *= params.options.y;
    outColor = vec4(max(color, vec3(0.0)), 1.0);
}
