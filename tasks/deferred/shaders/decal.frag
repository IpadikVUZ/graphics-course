#version 450

layout(set=0, binding=0) uniform sampler2D srcAlbedoAO;
layout(set=0, binding=1) uniform sampler2D srcNormalRM;
layout(set=0, binding=2) uniform sampler2D depthTexture;
layout(set=0, binding=3) uniform sampler2D decalAlbedo;
layout(set=0, binding=4) uniform sampler2D decalNormals;
layout(set=0, binding=5) uniform sampler2D decalRoughness;

struct Decal {
    mat4 worldToBox;
    mat4 boxToWorld;
    vec4 tint;
    vec4 controls;
};
layout(set=0, binding=6, std430) readonly buffer Decals { Decal items[]; } decals;
layout(push_constant) uniform DecalPush {
    mat4 invViewProj;
    vec4 screenSize;
    vec4 settings;
} params;
layout(location=0) out vec4 outAlbedoAO;
layout(location=1) out vec4 outNormalRM;

vec3 octDecode(vec2 e) {
    vec2 f = e * 2.0 - 1.0;
    vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}
vec2 octEncode(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 p = n.xy;
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(p, vec2(0.0)));
    return p * 0.5 + 0.5;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 originalColor = texelFetch(srcAlbedoAO, pixel, 0);
    vec4 originalSurface = texelFetch(srcNormalRM, pixel, 0);
    outAlbedoAO = originalColor;
    outNormalRM = originalSurface;
    float depth = texelFetch(depthTexture, pixel, 0).r;
    if (depth >= 0.999999) return;

    vec2 uv = (vec2(pixel) + 0.5) / params.screenSize.xy;
    vec4 h = params.invViewProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec3 world = h.xyz / h.w;
    Decal decal = decals.items[int(params.settings.x)];
    vec3 local = (decal.worldToBox * vec4(world, 1.0)).xyz;
    if (any(greaterThan(abs(local), vec3(1.0)))) return;
    vec3 N = octDecode(originalSurface.xy);
    vec3 up = normalize(decal.boxToWorld[1].xyz);

    float tile = floor(decal.controls.w + 0.5);
    vec2 localUV = clamp(local.xz * 0.5 + 0.5, vec2(0.002), vec2(0.998));
    vec2 atlasUV = vec2((tile + localUV.x) * 0.25, localUV.y);
    vec4 paint = texture(decalAlbedo, atlasUV);
    vec4 bump = texture(decalNormals, atlasUV);
    vec4 rough = texture(decalRoughness, atlasUV);
    float alpha = clamp(paint.a * decal.controls.x * params.settings.y, 0.0, 1.0);
    if (alpha <= 0.0001) return;

    outAlbedoAO.rgb = mix(originalColor.rgb, paint.rgb * decal.tint.rgb, alpha);

    float normalAlpha = clamp(alpha * bump.a * decal.controls.y * params.settings.z, 0.0, 1.0);
    vec3 tang = normalize(bump.rgb * 2.0 - 1.0);
    vec3 X = normalize(decal.boxToWorld[0].xyz);
    vec3 Z = normalize(decal.boxToWorld[2].xyz);
    vec3 projectedNormal = normalize(tang.x * X + tang.y * Z + tang.z * up);
    outNormalRM.xy = octEncode(normalize(mix(N, projectedNormal, normalAlpha)));
    float roughAlpha = clamp(alpha * rough.a * decal.controls.z * params.settings.w, 0.0, 1.0);
    outNormalRM.z = mix(originalSurface.z, rough.r, roughAlpha);
}
