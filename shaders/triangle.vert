#version 450

layout(set = 0, binding = 0) uniform CameraData {
    mat4 viewProjection;
} camera;

layout(std430, set = 0, binding = 2) readonly buffer ComputeOutput {
    vec4 values[];
} computeOutput;

layout(push_constant) uniform DrawData {
    vec4 tint;
} drawData;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUv;

layout(location = 0) out vec3 color;
layout(location = 1) out vec2 uv;

void main() {
    gl_Position = camera.viewProjection * vec4(inPosition, 1.0);
    color = inColor * computeOutput.values[0].rgb * drawData.tint.rgb;
    uv = inUv;
}
