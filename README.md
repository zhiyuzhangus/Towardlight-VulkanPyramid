# Towardlight-VulkanPyramid

A native C++17 learning application that draws the original colored 3D pyramid
using Vulkan 1.3. GLFW is used only for the window and events; rendering,
resource management, command execution, and synchronization use the native
Vulkan C API (there is no rendering engine or memory-allocation wrapper).

[**Try the interactive pyramid in your browser**](https://zhiyuzhangus.github.io/Towardlight-VulkanPyramid/)

The browser version remains an independent viewer. The native application
preserves the same pyramid geometry, projections, camera movement, zoom, and
mouse picking.

## Requirements and build

- Linux, a Vulkan 1.3-capable driver and desktop session
- CMake 3.20+, a C++17 compiler, Vulkan SDK headers/loader
- Khronos validation layer, GLFW 3.3+, GLM, and `glslangValidator`

On Ubuntu, install the development packages with:

```sh
sudo apt install build-essential cmake libvulkan-dev vulkan-validationlayers \
  libglfw3-dev libglm-dev glslang-tools
```

Build and run:

```sh
cmake -S . -B build-local
cmake --build build-local
./build-local/vulkan_triangle
```

GLSL sources are compiled to SPIR-V as part of the build. Validation is
required at runtime: the app reports warnings/errors through
`VK_EXT_debug_utils` and exits with an explicit error if the Khronos layer is
not installed. Synchronization validation is enabled when exposed by that
layer. The app prefers a discrete GPU, then an integrated GPU, then a CPU
Vulkan implementation.

Start directly in the dynamic-rendering path with
`./build-local/vulkan_triangle --dynamic-rendering`; **T** switches paths
while the application is running.

Controls: **left-drag** orbits, **middle/right-drag** pans, **+/-** zooms,
**Ctrl+left-click** picks the orbit pivot, and **Ctrl+left-drag** pans.
**R** resets, **P** selects perspective, **O** selects orthographic, and
**T** switches between render-pass and Vulkan 1.3 dynamic rendering.

## Vulkan study map

The scene and compute operation are intentionally minimal. Native object
creation, memory binding, command recording, and synchronization remain
explicit in `src/main.cpp`; shader interfaces are in `shaders/`.

### Initialization workflow

1. Check the Vulkan loader version, GLFW-required instance extensions,
   `VK_EXT_debug_utils`, and `VK_LAYER_KHRONOS_validation`.
2. Create a Vulkan 1.3 instance with a debug-messenger create-info in `pNext`,
   then create the GLFW surface.
3. Enumerate physical devices and query Vulkan 1.3 dynamic-rendering and
   synchronization2 features, properties, limits, queue families, swapchain
   support, and surface formats/modes. Select the highest-ranked suitable
   device.
4. Create one queue per unique graphics, present, compute, and transfer family.
   The roles share a family wherever possible; the compute output is concurrent
   between compute and graphics if those roles need different families.
5. Create role-specific command pools, a pipeline cache, descriptor layouts,
   and long-lived geometry, texture, uniform, and storage resources.
6. Create swapchain-dependent views, depth images, render-pass and
   dynamic-rendering pipelines, framebuffers, and image-indexed commands.

### CPU-to-GPU transfer workflow

The CPU defines 18 interleaved vertices (position, face color, UV) and 18
indices. It packs both into one host-visible staging buffer, flushing mapped
memory if that allocation is not coherent. A transfer command buffer copies
them to separate device-local vertex/index buffers. `vkCmdPipelineBarrier2`
publishes the writes to vertex/index input. A transfer fence confirms that the
staging allocation can be freed. The one-pixel white texture follows the same
staging pattern with `vkCmdCopyBufferToImage`, transitioning from `UNDEFINED` to
transfer-destination and then shader-read-only layout.

Camera matrices are written to persistently mapped host-visible uniform buffers
per frame. Per-frame device-local storage buffers are populated by compute and
read by the vertex shader.

### Graphics, compute, and synchronization workflow

Each frame first waits for its frame fence and acquires a swapchain image with
a binary semaphore. A compute primary command buffer binds its pipeline and
per-frame storage descriptor, then dispatches one invocation that writes a
neutral `vec4(1)` consumed by the vertex shader. `vkQueueSubmit2` signals a
monotonically increasing timeline semaphore value. The graphics submission
waits on that value at the vertex-shader stage and on the acquire semaphore
before all graphics commands, so even the explicit image-layout transition
cannot run ahead of acquisition. A buffer barrier makes the compute write
available to the vertex shader.

The graphics primary command buffer owns the rendering scope and executes a
secondary command buffer. The secondary binds the pipeline, descriptors,
vertex/index buffers, and small tint push constant, then calls
`vkCmdDrawIndexed`. Press **T** to select a render-pass pipeline or a
dynamic-rendering pipeline. A binary semaphore indexed by swapchain image
signals completion for presentation; it is reused only after that image is
acquired again. Frame fences protect per-frame CPU/GPU data, while
`imagesInFlight` fences protect per-image rendering resources.

### Resource relationship diagram

```text
GLFW window -> VkSurfaceKHR -> VkSwapchainKHR -> swapchain VkImage
                                              -> image view/framebuffer
                                              -> per-image present semaphore
                                              -> per-image depth image/view

CPU vertex/index arrays -> mapped staging VkBuffer
                        -> transfer commands -> device-local vertex/index VkBuffer
                        -> graphics secondary -> vkCmdDrawIndexed

Per-frame CPU camera -> mapped uniform VkBuffer -> graphics descriptor set
White texel -> staging VkBuffer -> VkImage -> VkImageView + VkSampler
                                  -> combined-image-sampler descriptor
Compute shader -> per-frame storage VkBuffer -> timeline + buffer barrier
                                              -> vertex shader reads same buffer

Graphics pipeline layout -> uniform + image sampler + storage descriptors + push constant
Compute pipeline layout -> compute storage descriptor
Graphics primary -> executes draw secondary -> graphics queue -> presentation queue
```

### Resource creation and destruction order

```text
Create:
instance/debug messenger -> surface -> physical device -> device/queues
-> command pools/pipeline cache/layouts -> staging/device buffers and images
-> descriptors/synchronization -> swapchain/views/depth/render pass/pipelines/framebuffers

Destroy (reverse dependencies):
wait for outstanding device work -> command buffers/framebuffers/pipelines/render pass
-> depth and swapchain views/semaphores/swapchain -> descriptor pools
-> sampler/texture view/image/memory -> uniform/storage/vertex/index buffers and memory
-> pipeline layouts/descriptor layouts/cache/command pools/sync objects -> device
-> surface/debug messenger -> instance
```

Swapchain recreation waits for a non-zero framebuffer size and outstanding
device work to retire, then rebuilds swapchain-dependent resources. Normal frame
submission does not use `vkQueueWaitIdle` or `vkDeviceWaitIdle`; device idle is
limited to shutdown and swapchain recreation.

### Scope notes

- The white 1x1 procedural texture preserves the face colors while exercising
  image upload and sampling.
- Compute intentionally performs only a neutral write, but its output is
  consumed by graphics rather than being unused demonstration infrastructure.
- Graphics and transfer roles normally share the graphics family. A separate
  compute family uses concurrent buffer sharing and timeline synchronization,
  so no buffer ownership transfer is required.
- Non-coherent host ranges are flushed using a whole mapped allocation, keeping
  ranges aligned to the physical device's non-coherent atom size.
- There is no GPU-to-CPU readback buffer in this scene, so memory invalidation
  is not applicable; add it before reading host-visible memory written by GPU work.
- The tested NVIDIA device uses queue family 0 for graphics, present, compute,
  and transfer. Separate compute/present families are handled through concurrent
  sharing and semaphore synchronization, but a dedicated-family device was not
  available for runtime validation.
- The native camera and picking math remains simple and the pyramid geometry
  unchanged.

## Author

This repository is written and learned by Zhiyu Zhang.
