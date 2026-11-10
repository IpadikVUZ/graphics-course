#version 450
layout(push_constant) uniform DrawSettings {
    mat4 viewProj;
    vec4 light;
    vec4 eye;
    vec4 detail;
    vec4 levels;
} params;
layout(location = 0) in vec3 iWorld;
layout(location = 1) in vec3 iNormal;
layout(location = 2) in vec4 iColor;
layout(location = 0) out vec4 outAlbedoAO;
layout(location = 1) out vec4 outNormalRM;

vec2 octEncode(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 p = n.xy;
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(p, vec2(0.0)));
    return p * 0.5 + 0.5;
}
float hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
float noise3(vec3 p) {
    vec3 b = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(b), hash(b + vec3(1,0,0)), f.x),
            mix(hash(b + vec3(0,1,0)), hash(b + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(b + vec3(0,0,1)), hash(b + vec3(1,0,1)), f.x),
            mix(hash(b + vec3(0,1,1)), hash(b + vec3(1,1,1)), f.x), f.y), f.z);
}
void main() {
    vec3 N = normalize(gl_FrontFacing ? iNormal : -iNormal);
    float stone = noise3(iWorld * 2.7) * 0.65 + noise3(iWorld * 11.0) * 0.35;
    float mineral = smoothstep(0.57, 0.78, noise3(iWorld * 1.5));
    vec3 albedo = iColor.rgb * mix(0.72, 1.23, stone);
    albedo = mix(albedo, albedo * vec3(1.12, 0.94, 0.77), mineral * 0.20);
    float roughness = clamp(iColor.a + 0.15 * (stone - 0.5), 0.12, 1.0);
    outAlbedoAO = vec4(albedo, 1.0);
    outNormalRM = vec4(octEncode(N), roughness, 0.0);
}
