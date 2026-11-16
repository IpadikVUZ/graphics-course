#version 450
layout(set=0, binding=0) uniform sampler2D hdrImage;
layout(set=0, binding=1) uniform sampler3D colorLut;
layout(set=0, binding=2, std430) readonly buffer Exposure { vec4 values; } exposure;
layout(push_constant) uniform Settings { vec4 controls; } params;
layout(location=0) out vec4 outColor;

vec3 linearToSrgb(vec3 x) {
    vec3 lo = 12.92 * x;
    vec3 hi = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, greaterThan(x, vec3(0.0031308)));
}
vec3 srgbToLinear(vec3 x) {
    vec3 lo = x / 12.92;
    vec3 hi = pow(max((x + 0.055) / 1.055, vec3(0.0)), vec3(2.4));
    return mix(lo, hi, greaterThan(x, vec3(0.04045)));
}
vec3 acesFit(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    vec3 hdr = max(texelFetch(hdrImage, xy, 0).rgb, vec3(0.0));
    vec3 linearColor = hdr;
    if (params.controls.x > 0.5) {
        vec3 exposed = hdr * exposure.values.x;
        linearColor = params.controls.y > 0.5 ? acesFit(exposed) : exposed;
    }
    vec3 displayColor = linearToSrgb(clamp(linearColor, vec3(0.0), vec3(1.0)));
    if (params.controls.z > 0.5) {
        vec3 uvw = displayColor * (31.0 / 32.0) + (0.5 / 32.0);
        displayColor = texture(colorLut, uvw).rgb;
    }
    outColor = vec4(params.controls.w > 0.5 ? srgbToLinear(displayColor) : displayColor, 1.0);
}
