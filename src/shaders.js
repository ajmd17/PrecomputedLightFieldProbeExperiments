export const gbufferVert = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uViewProj;
uniform mat4 uModel;

out vec3 vNormal;
out vec3 vColor;
out vec3 vWorldPos;

void main() {
    vec4 worldPos = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(uModel) * aNormal;
    vColor = aColor;
    gl_Position = uViewProj * worldPos;
}
`;

export const gbufferFrag = `#version 300 es
precision highp float;

in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPos;

uniform vec3 uProbePos;

layout(location = 0) out vec4 oRadiance;
layout(location = 1) out vec4 oNormal;
layout(location = 2) out float oDistance;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uProbePos - vWorldPos);
    // Orient normal to face the probe (visible surface orientation)
    if (dot(N, V) < 0.0) N = -N;

    oRadiance = vec4(vColor, 1.0);
    oNormal = vec4(N * 0.5 + 0.5, 1.0);
    oDistance = length(vWorldPos - uProbePos);
}
`;

export const octahedralVert = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aPosition;

out vec2 vTexCoord;

void main() {
    vTexCoord = aPosition * 0.5 + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

export const octahedralFrag = `#version 300 es
precision highp float;

uniform samplerCube uRadianceCubemap;
uniform samplerCube uNormalCubemap;
uniform samplerCube uDistanceCubemap;

in vec2 vTexCoord;

layout(location = 0) out vec4 oRadiance;
layout(location = 1) out vec4 oNormal;
layout(location = 2) out float oDistance;

vec3 octDecode(vec2 uv) {
    uv = uv * 2.0 - 1.0;
    vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    if (v.z < 0.0) {
        vec2 s = sign(uv);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

void main() {
    vec3 dir = octDecode(vTexCoord);
    oRadiance = texture(uRadianceCubemap, dir);
    oNormal = texture(uNormalCubemap, dir);
    oDistance = texture(uDistanceCubemap, dir).r;
}
`;

export const irradianceFilterFrag = `#version 300 es
precision highp float;

#define PI 3.14159265

uniform samplerCube uRadianceCubemap;
uniform int uSampleCount;

in vec2 vTexCoord;

layout(location = 0) out vec4 oIrradiance;

// Fibonacci sphere: returns direction uniformly distributed on sphere
vec3 fibSphere(int i, int N) {
    float phi = acos(1.0 - 2.0 * float(i + 1) / float(N + 1));
    float theta = PI * (1.0 + sqrt(5.0)) * float(i);
    return vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

vec3 octDecode(vec2 uv) {
    uv = uv * 2.0 - 1.0;
    vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    if (v.z < 0.0) {
        vec2 s = sign(uv);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// Build orthonormal basis from a single vector (N)
mat3 buildBasis(vec3 N) {
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    return mat3(T, B, N);
}

void main() {
    vec3 N = octDecode(vTexCoord);
    mat3 basis = buildBasis(N);
    vec3 irradiance = vec3(0.0);
    int Ns = uSampleCount;

    for (int i = 0; i < 1024; i++) {
        if (i >= Ns) break;
        vec3 L = fibSphere(i, Ns);
        // Only sample upper hemisphere (L.z > 0)
        if (L.z <= 0.0) L.z = -L.z;
        // Transform from local frame (z=up=N) to world
        vec3 L_world = basis * L;
        float NdotL = L.z;  // L.z = dot(N, L_world) since L is in local frame
        irradiance += texture(uRadianceCubemap, L_world).rgb * NdotL;
    }

    irradiance *= 2.0 / float(Ns);
    oIrradiance = vec4(irradiance, 1.0);
}
`;

export const varianceShadowFrag = `#version 300 es
precision highp float;

uniform samplerCube uDistanceCubemap;

in vec2 vTexCoord;

layout(location = 0) out vec4 oVSM;

vec3 octDecode(vec2 uv) {
    uv = uv * 2.0 - 1.0;
    vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    if (v.z < 0.0) {
        vec2 s = sign(uv);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

// Poisson-disk style samples for variance estimation
vec3 poissonSample(int i) {
    float phi = acos(1.0 - 2.0 * float(i + 1) / 33.0);
    float theta = 2.399963 * float(i);
    return vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

void main() {
    vec3 dir = octDecode(vTexCoord);

    float depth = texture(uDistanceCubemap, dir).r;
    float depthSq = depth * depth;

    // Average depth and depth^2 over a small kernel for variance estimation
    float avgDepth = depth;
    float avgDepthSq = depthSq;
    int kernelSize = 8;

    for (int i = 0; i < 16; i++) {
        if (i >= kernelSize) break;
        vec3 sampleDir = normalize(dir + poissonSample(i) * 0.05);
        float d = texture(uDistanceCubemap, sampleDir).r;
        avgDepth += d;
        avgDepthSq += d * d;
    }

    float invN = 1.0 / float(kernelSize + 1);
    avgDepth *= invN;
    avgDepthSq *= invN;

    float variance = max(avgDepthSq - avgDepth * avgDepth, 0.0);
    float standardDeviation = sqrt(variance);

    // Store: average depth, variance (for Chebyshev inequality in tracing)
    oVSM = vec4(avgDepth, variance, 0.0, 1.0);
}
`;
