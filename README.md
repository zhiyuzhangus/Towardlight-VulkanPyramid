# Vulkan Pyramid

A small native C++ sample that draws a colored 3D pyramid using the Vulkan API.
GLFW is used only to create the window and receive window events; rendering is
performed entirely with Vulkan (there is no OpenGL renderer).

[**Try the interactive pyramid in your browser**](https://zhiyuzhangus.github.io/ZRX---Vulkan-triangle-GLFW/)

Orbit, pan, zoom, and switch projections in a browser rendering of the same pyramid.

## Requirements

- Linux with a working Vulkan driver and desktop session
- CMake 3.20 or newer
- A C++17 compiler
- Vulkan SDK headers/loader
- GLFW 3.3 or newer
- GLM
- `glslangValidator`

On Ubuntu, the development packages can be installed with:

```sh
sudo apt install build-essential cmake libvulkan-dev libglfw3-dev libglm-dev glslang-tools
```

Install a Vulkan driver for your GPU as well. For Mesa-based Intel/AMD GPUs,
`mesa-vulkan-drivers` provides one; NVIDIA systems need the matching proprietary
driver. `vulkan-tools` is optional but provides `vulkaninfo` for diagnostics.

The program prints the Vulkan device it selected. It prefers a discrete GPU,
then an integrated GPU, and can use a CPU Vulkan implementation if that is the
only suitable device. Use **left-drag** to orbit, **middle/right-drag** to pan,
and **+/-** to zoom. **Ctrl+left-click** a visible point on the pyramid to set
the orbit and zoom pivot there; while holding **Ctrl**, left-drag to pan.
Press **R** to reset the camera, **P** for perspective projection, or **O**
for orthographic projection. Close the window to exit.
# ZRX

This responstory is writen and learned by Zhiyu,Zhang.
