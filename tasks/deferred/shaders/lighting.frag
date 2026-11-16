#version 450
layout(set=0, binding=0) uniform sampler2D gAlbedoAO;
layout(set=0, binding=1) uniform sampler2D gNormalRM;
layout(set=0, binding=2) uniform sampler2D gDepth;

layout(push_constant) uniform LightingSettings {
    mat4 invViewProj;
    vec4 light;
    vec4 eye;
    vec4 options;
} params;
layout(location = 0) out vec4 outColor;
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
        float t = float(p.y) / float(size.y);
        vec3 sky = mix(vec3(0.30, 0.46, 0.64), vec3(0.73, 0.82, 0.87), t);
        outColor = vec4(pow(sky, vec3(1.0/2.2)), 1.0);
        return;
    }
    vec4 g0 = texelFetch(gAlbedoAO, p, 0);
    vec4 g1 = texelFetch(gNormalRM, p, 0);
    vec2 uv = (vec2(p) + 0.5) / vec2(size);

    vec4 clip = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 h = params.invViewProj * clip;
    vec3 world = h.xyz / h.w;

    vec3 albedo = g0.rgb;
    float ao = g0.a;
    vec3 N = octDecode(g1.xy);
    float roughness = clamp(g1.z, 0.06, 1.0);
    float metalness = clamp(g1.w, 0.0, 1.0);

    vec3 V = normalize(params.eye.xyz - world);
    vec3 toLight = params.light.xyz - world;
    vec3 L = normalize(toLight);
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
    float d2 = dot(toLight, toLight);
    float attenuation = 1.0 / (1.0 + 0.0015 * d2);
    vec3 color = (diffuse + specular) * NoL * params.light.w * attenuation;
    color += albedo * params.options.x * ao;
    color *= params.options.y;
    color = color / (color + vec3(1.0));
    outColor = vec4(pow(max(color, vec3(0)), vec3(1.0/2.2)), 1.0);
}
