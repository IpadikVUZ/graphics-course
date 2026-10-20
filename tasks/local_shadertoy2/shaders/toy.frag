#version 450

layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D iChannel0;
layout(binding = 1) uniform sampler2D iChannel1;
layout(push_constant) uniform PushConstants { float time; } pc;

const vec3 iResolution = vec3(1280.0, 720.0, 1.0);
#define iTime pc.time

float star2d(vec2 point)
{
    float angle = radians(36.0);

    vec2 sideDir1 = vec2(cos(angle), -sin(angle));
    vec2 sideDir2 = vec2(-cos(angle), -sin(angle));

    point.x = abs(point.x);
    point -= 2.0 * max(dot(sideDir1, point), 0.0) * sideDir1;
    point -= 2.0 * max(dot(sideDir2, point), 0.0) * sideDir2;
    point.x = abs(point.x);
    point.y -= 1.0;

    vec2 edgeDir = 0.5 * vec2(-sideDir1.y, sideDir1.x) - vec2(0.0, 1.0);
    float projection = clamp(dot(point, edgeDir) / dot(edgeDir, edgeDir), 0.0, 1.0);

    return length(point - edgeDir * projection) * sign(point.y * edgeDir.x - point.x * edgeDir.y);
}

float star3d(vec3 point)
{
    float starShape = star2d(point.xy);
    float starThickness = abs(point.z) - 0.1;
    return max(starShape, starThickness);
}

mat2 rotation2d(float angle)
{
    float sineValue = sin(angle);
    float cosineValue = cos(angle);
    return mat2(cosineValue, -sineValue, sineValue, cosineValue);
}

float scene(vec3 point)
{
    point.xz *= rotation2d(iTime);
    return star3d(point);
}

float rayMarch(vec3 rayOrigin, vec3 rayDirection)
{
    float distanceTravelled = 0.0;

    for (int step = 0; step < 100; step++)
    {
        float distanceToScene = scene(rayOrigin + rayDirection * distanceTravelled);

        if (distanceToScene < 0.001) break;

        distanceTravelled += distanceToScene;
        if (distanceTravelled > 30.0) break;
    }

    return distanceTravelled;
}

vec3 getNormal(vec3 point)
{
    float epsilon = 0.001;

    return normalize(vec3(
        scene(point + vec3(epsilon, 0.0, 0.0)) - scene(point - vec3(epsilon, 0.0, 0.0)),
        scene(point + vec3(0.0, epsilon, 0.0)) - scene(point - vec3(0.0, epsilon, 0.0)),
        scene(point + vec3(0.0, 0.0, epsilon)) - scene(point - vec3(0.0, 0.0, epsilon))
    ));
}

const float PI = 3.14159265359;

float distributionGGX(vec3 normal, vec3 halfwayDirection, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(normal, halfwayDirection), 0.0);
    float NdotH2 = NdotH * NdotH;

    float denominator = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denominator * denominator);
}

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 normal, vec3 viewDirection, vec3 lightDirection, float roughness)
{
    float NdotV = max(dot(normal, viewDirection), 0.0);
    float NdotL = max(dot(normal, lightDirection), 0.0);

    float ggxView = geometrySchlickGGX(NdotV, roughness);
    float ggxLight = geometrySchlickGGX(NdotL, roughness);

    return ggxView * ggxLight;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;

    vec3 rayOrigin = vec3(0.0, 0.0, -3.5);
    vec3 rayDirection = normalize(vec3(uv, 1.0));

    float hitDistance = rayMarch(rayOrigin, rayDirection);
    vec3 finalColor = vec3(0);

    if (hitDistance < 30.0)
    {
        vec3 hitPoint = rayOrigin + rayDirection * hitDistance;
        vec3 normal = getNormal(hitPoint);

        vec3 lightPosition = vec3(2.0, 2.0, -8.0);
        vec3 lightDirection = normalize(lightPosition - hitPoint);
        vec3 viewDirection = normalize(rayOrigin - hitPoint);
        vec3 halfwayDirection = normalize(lightDirection + viewDirection);

        vec3 localPoint = hitPoint;
        localPoint.xz *= rotation2d(iTime);

        vec2 textureUv = localPoint.xy * 0.5 + 0.5;

        vec3 albedo = texture(iChannel0, textureUv).rgb;
        vec2 material = texture(iChannel1, textureUv).rg;

        float roughness = clamp(material.r, 0.05, 1.0);
        float metalness = clamp(material.g, 0.0, 1.0);

        vec3 F0 = mix(vec3(0.04), albedo, metalness);

        vec3 F = fresnelSchlick(max(dot(halfwayDirection, viewDirection), 0.0), F0);
        float D = distributionGGX(normal, halfwayDirection, roughness);
        float G = geometrySmith(normal, viewDirection, lightDirection, roughness);

        vec3 numerator = D * G * F;
        float denominator =
            4.0 * max(dot(normal, viewDirection), 0.0) *
            max(dot(normal, lightDirection), 0.0) + 0.0001;

        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metalness);

        float NdotL = max(dot(normal, lightDirection), 0.0);

        float lightDistance = length(lightPosition - hitPoint);
        vec3 radiance = vec3(200.0) / (lightDistance * lightDistance);

        vec3 ambient = 0.25 * albedo;

        finalColor = ambient + (kD * albedo / PI + specular) * radiance * NdotL;
    }

    fragColor = vec4(finalColor, 1.0);
}

void main()
{
    mainImage(outColor, vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y));
}
