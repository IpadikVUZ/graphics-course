#version 450

layout(location = 0) out vec4 outColor;
layout(push_constant) uniform PushConstants { float time; } pc;

const vec3 iResolution = vec3(1280.0, 720.0, 1.0);
#define iTime pc.time

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;

    float pattern = 0.5 + 0.5 * sin(10.0 * uv.x + iTime);

    vec3 color = mix(
        vec3(1.0, 0.2, 0.1),
        vec3(1.0, 0.9, 0.1),
        pattern
    );

    fragColor = vec4(color, 1.0);
}

void main()
{
    mainImage(outColor, gl_FragCoord.xy);
}
