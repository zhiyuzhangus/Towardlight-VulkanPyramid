#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kWindowWidth = 960;
constexpr uint32_t kWindowHeight = 640;
constexpr uint32_t kFramesInFlight = 2;
constexpr const char* kSwapchainExtension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr float kCameraRadius = 5.15f;
constexpr float kCameraYaw = 0.66f;
constexpr float kCameraPitch = 0.48f;
constexpr float kMouseSensitivity = 0.005f;
constexpr float kMinimumCameraDistance = 2.0f;
constexpr float kMaximumCameraDistance = 15.0f;

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                                 std::to_string(result) + ")");
    }
}

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    std::optional<uint32_t> compute;
    std::optional<uint32_t> transfer;

    bool complete() const {
        return graphics && present && compute && transfer;
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 uv;
};

struct CameraUniform {
    glm::mat4 viewProjection;
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE; 
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocationSize = 0;
    void* mapped = nullptr;
    bool coherent = false;
};

struct ImageAllocation {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

class VulkanPyramid {
public:
    void setDynamicRendering(bool enabled) {
        useDynamicRendering_ = enabled;
    }

    void run() {
        initWindow();
        try {
            initVulkan();
            mainLoop();
        } catch (...) {
            cleanupVulkan();
            glfwDestroyWindow(window_);
            glfwTerminate();
            throw;
        }
        cleanupVulkan();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    VkQueue transferQueue_ = VK_NULL_HANDLE;
    QueueFamilies queueFamilies_{};

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImageView> swapchainImageViews_;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    std::vector<ImageAllocation> depthImages_;
    std::vector<VkImageView> depthImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkSemaphore> presentReadySemaphores_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    VkCommandPool graphicsCommandPool_ = VK_NULL_HANDLE;
    VkCommandPool computeCommandPool_ = VK_NULL_HANDLE;
    VkCommandPool transferCommandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> primaryCommandBuffers_;
    std::vector<VkCommandBuffer> secondaryCommandBuffers_;
    std::array<VkCommandBuffer, kFramesInFlight> computeCommandBuffers_{};

    Buffer vertexBuffer_{};
    Buffer indexBuffer_{};
    std::array<Buffer, kFramesInFlight> cameraBuffers_{};
    std::array<Buffer, kFramesInFlight> computeOutputBuffers_{};
    ImageAllocation texture_{};
    VkImageView textureView_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout graphicsSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool graphicsDescriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kFramesInFlight> graphicsDescriptorSets_{};
    std::array<VkDescriptorSet, kFramesInFlight> computeDescriptorSets_{};
    VkPipelineLayout graphicsPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline dynamicRenderingPipeline_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

    std::array<VkSemaphore, kFramesInFlight> imageAvailableSemaphores_{};
    std::array<VkFence, kFramesInFlight> inFlightFences_{};
    std::vector<VkFence> imagesInFlight_;
    VkSemaphore computeTimeline_ = VK_NULL_HANDLE;
    uint64_t computeTimelineValue_ = 0;
    size_t currentFrame_ = 0;

    bool framebufferResized_ = false;
    bool orthographicView_ = false;
    bool useDynamicRendering_ = false;
    bool mouseDragging_ = false;
    int dragButton_ = GLFW_MOUSE_BUTTON_LEFT;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    float cameraYaw_ = kCameraYaw;
    float cameraPitch_ = kCameraPitch;
    float cameraDistance_ = kCameraRadius;
    glm::vec3 cameraTarget_{0.0f};

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void*) {
        const char* label =
            severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "WARNING";
        std::cerr << "[Vulkan validation " << label << "] "
                  << callbackData->pMessage << '\n';
        return VK_FALSE;
    }

    VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo() const {
        VkDebugUtilsMessengerCreateInfoEXT info{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = debugCallback;
        return info;
    }

    static void framebufferSizeCallback(GLFWwindow* window, int, int) {
        static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window))->framebufferResized_ = true;
    }

    static void keyCallback(GLFWwindow* window, int key, int, int action, int) {
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS && (key == GLFW_KEY_P || key == GLFW_KEY_O)) {
            app->orthographicView_ = key == GLFW_KEY_O;
            glfwSetWindowTitle(window, app->windowTitle().c_str());
        } else if (action == GLFW_PRESS && key == GLFW_KEY_T) {
            app->useDynamicRendering_ = !app->useDynamicRendering_;
            glfwSetWindowTitle(window, app->windowTitle().c_str());
        } else if (action == GLFW_PRESS && key == GLFW_KEY_R) {
            app->cameraYaw_ = kCameraYaw;
            app->cameraPitch_ = kCameraPitch;
            app->cameraDistance_ = kCameraRadius;
            app->cameraTarget_ = glm::vec3(0.0f);
        } else if ((action == GLFW_PRESS || action == GLFW_REPEAT) &&
                   (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)) {
            app->zoomCamera(0.9f);
        } else if ((action == GLFW_PRESS || action == GLFW_REPEAT) &&
                   (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)) {
            app->zoomCamera(1.0f / 0.9f);
        }
    }

    std::string windowTitle() const {
        return std::string("Vulkan Pyramid - ") +
               (orthographicView_ ? "Orthographic" : "Perspective") + " / " +
               (useDynamicRendering_ ? "Dynamic Rendering (T)" : "Render Pass (T)");
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
        if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_MIDDLE &&
            button != GLFW_MOUSE_BUTTON_RIGHT) {
            return;
        }
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS) {
            app->mouseDragging_ = true;
            app->dragButton_ = button;
            glfwGetCursorPos(window, &app->lastCursorX_, &app->lastCursorY_);
            const bool controlHeld =
                glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            if (button == GLFW_MOUSE_BUTTON_LEFT && controlHeld) {
                app->pickOrbitPivot(app->lastCursorX_, app->lastCursorY_);
            }
        } else if (action == GLFW_RELEASE && app->dragButton_ == button) {
            app->mouseDragging_ = false;
        }
    }

    static void cursorPositionCallback(GLFWwindow* window, double x, double y) {
        auto* app = static_cast<VulkanPyramid*>(glfwGetWindowUserPointer(window));
        if (!app->mouseDragging_) {
            return;
        }
        const double deltaX = x - app->lastCursorX_;
        const double deltaY = y - app->lastCursorY_;
        app->lastCursorX_ = x;
        app->lastCursorY_ = y;
        if (app->dragButton_ == GLFW_MOUSE_BUTTON_LEFT) {
            const bool controlHeld =
                glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
            if (controlHeld) {
                app->panCamera(static_cast<float>(deltaX), static_cast<float>(deltaY));
            } else {
                app->cameraYaw_ -= static_cast<float>(deltaX) * kMouseSensitivity;
                app->cameraPitch_ = std::clamp(
                    app->cameraPitch_ + static_cast<float>(deltaY) * kMouseSensitivity,
                    -1.45f, 1.45f);
            }
        } else {
            app->panCamera(static_cast<float>(deltaX), static_cast<float>(deltaY));
        }
    }

    void zoomCamera(float factor) {
        cameraDistance_ = std::clamp(cameraDistance_ * factor,
                                     kMinimumCameraDistance, kMaximumCameraDistance);
    }

    glm::vec3 cameraPosition() const {
        const float horizontalRadius = cameraDistance_ * std::cos(cameraPitch_);
        return cameraTarget_ + glm::vec3(
            horizontalRadius * std::sin(cameraYaw_),
            cameraDistance_ * std::sin(cameraPitch_),
            horizontalRadius * std::cos(cameraYaw_));
    }

    glm::mat4 viewProjection() const {
        const float aspect = static_cast<float>(swapchainExtent_.width) /
                             static_cast<float>(swapchainExtent_.height);
        const glm::mat4 view = glm::lookAt(cameraPosition(), cameraTarget_,
                                           glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection;
        if (orthographicView_) {
            const float halfHeight = 1.8f * cameraDistance_ / kCameraRadius;
            const float halfWidth = halfHeight * aspect;
            projection = glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight,
                                    0.1f, 20.0f);
        } else {
            projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 20.0f);
        }
        projection[1][1] *= -1.0f;
        return projection * view;
    }

    void pickOrbitPivot(double cursorX, double cursorY) {
        int windowWidth = 0;
        int windowHeight = 0;
        glfwGetWindowSize(window_, &windowWidth, &windowHeight);
        if (windowWidth <= 0 || windowHeight <= 0 ||
            swapchainExtent_.width == 0 || swapchainExtent_.height == 0) {
            return;
        }
        const float framebufferX =
            static_cast<float>(cursorX * swapchainExtent_.width / windowWidth);
        const float framebufferY =
            static_cast<float>(cursorY * swapchainExtent_.height / windowHeight);
        const float ndcX = 2.0f * framebufferX / swapchainExtent_.width - 1.0f;
        const float ndcY = 2.0f * framebufferY / swapchainExtent_.height - 1.0f;
        const glm::mat4 inverseViewProjection = glm::inverse(viewProjection());
        glm::vec4 nearPoint = inverseViewProjection * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
        glm::vec4 farPoint = inverseViewProjection * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
        if (std::abs(nearPoint.w) < 1e-6f || std::abs(farPoint.w) < 1e-6f) {
            return;
        }
        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;
        const glm::vec3 rayOrigin =
            orthographicView_ ? glm::vec3(nearPoint) : cameraPosition();
        const glm::vec3 rayDirection = glm::normalize(glm::vec3(farPoint - nearPoint));
        static const std::array<glm::vec3, 18> triangles = {{
            {0.0f, 0.9f, 0.0f}, {-0.7f, -0.7f, -0.7f}, {0.7f, -0.7f, -0.7f},
            {0.0f, 0.9f, 0.0f}, {0.7f, -0.7f, -0.7f}, {0.7f, -0.7f, 0.7f},
            {0.0f, 0.9f, 0.0f}, {0.7f, -0.7f, 0.7f}, {-0.7f, -0.7f, 0.7f},
            {0.0f, 0.9f, 0.0f}, {-0.7f, -0.7f, 0.7f}, {-0.7f, -0.7f, -0.7f},
            {-0.7f, -0.7f, -0.7f}, {0.7f, -0.7f, 0.7f}, {0.7f, -0.7f, -0.7f},
            {-0.7f, -0.7f, -0.7f}, {-0.7f, -0.7f, 0.7f}, {0.7f, -0.7f, 0.7f}
        }};
        float nearestDistance = std::numeric_limits<float>::infinity();
        glm::vec3 nearestPoint{};
        bool hit = false;
        for (size_t i = 0; i < triangles.size(); i += 3) {
            const glm::vec3 edge1 = triangles[i + 1] - triangles[i];
            const glm::vec3 edge2 = triangles[i + 2] - triangles[i];
            const glm::vec3 crossDirection = glm::cross(rayDirection, edge2);
            const float determinant = glm::dot(edge1, crossDirection);
            if (std::abs(determinant) < 1e-6f) {
                continue;
            }
            const float inverseDeterminant = 1.0f / determinant;
            const glm::vec3 fromVertex = rayOrigin - triangles[i];
            const float u = glm::dot(fromVertex, crossDirection) * inverseDeterminant;
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            const glm::vec3 crossFromVertex = glm::cross(fromVertex, edge1);
            const float v = glm::dot(rayDirection, crossFromVertex) * inverseDeterminant;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            const float distance =
                glm::dot(edge2, crossFromVertex) * inverseDeterminant;
            if (distance > 0.0f && distance < nearestDistance) {
                nearestDistance = distance;
                nearestPoint = rayOrigin + rayDirection * distance;
                hit = true;
            }
        }
        if (hit) {
            const glm::vec3 offset = cameraPosition() - nearestPoint;
            const float distance = glm::length(offset);
            if (distance <= 1e-6f) {
                return;
            }
            cameraTarget_ = nearestPoint;
            cameraDistance_ = std::clamp(distance, kMinimumCameraDistance,
                                         kMaximumCameraDistance);
            cameraYaw_ = std::atan2(offset.x, offset.z);
            cameraPitch_ = std::clamp(
                std::asin(std::clamp(offset.y / distance, -1.0f, 1.0f)),
                -1.45f, 1.45f);
        }
    }

    void panCamera(float deltaX, float deltaY) {
        const float horizontalRadius = cameraDistance_ * std::cos(cameraPitch_);
        const glm::vec3 cameraOffset(
            horizontalRadius * std::sin(cameraYaw_),
            cameraDistance_ * std::sin(cameraPitch_),
            horizontalRadius * std::cos(cameraYaw_));
        const glm::vec3 forward = glm::normalize(-cameraOffset);
        const glm::vec3 right =
            glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 cameraUp = glm::cross(right, forward);
        const float unitsPerPixel =
            orthographicView_
                ? (3.6f * cameraDistance_ / kCameraRadius) /
                      static_cast<float>(swapchainExtent_.height)
                : (2.0f * cameraDistance_ * std::tan(glm::radians(22.5f))) /
                      static_cast<float>(swapchainExtent_.height);
        cameraTarget_ += right * (-deltaX * unitsPerPixel) +
                         cameraUp * (deltaY * unitsPerPixel);
    }

    void initWindow() {
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("GLFW initialization failed");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        const std::string title = windowTitle();
        window_ = glfwCreateWindow(static_cast<int>(kWindowWidth),
                                   static_cast<int>(kWindowHeight), title.c_str(),
                                   nullptr, nullptr);
        if (window_ == nullptr) {
            glfwTerminate();
            throw std::runtime_error("GLFW could not create a window");
        }
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
        glfwSetKeyCallback(window_, keyCallback);
        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetCursorPosCallback(window_, cursorPositionCallback);
    }

    bool hasInstanceExtension(const char* name) const {
        uint32_t count = 0;
        checkVk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
                "vkEnumerateInstanceExtensionProperties");
        std::vector<VkExtensionProperties> extensions(count);
        checkVk(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()),
                "vkEnumerateInstanceExtensionProperties");
        return std::any_of(extensions.begin(), extensions.end(),
                           [name](const VkExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName, name) == 0;
                           });
    }

    bool hasInstanceLayer(const char* name) const {
        uint32_t count = 0;
        checkVk(vkEnumerateInstanceLayerProperties(&count, nullptr),
                "vkEnumerateInstanceLayerProperties");
        std::vector<VkLayerProperties> layers(count);
        checkVk(vkEnumerateInstanceLayerProperties(&count, layers.data()),
                "vkEnumerateInstanceLayerProperties");
        return std::any_of(layers.begin(), layers.end(),
                           [name](const VkLayerProperties& layer) {
                               return std::strcmp(layer.layerName, name) == 0;
                           });
    }

    bool hasLayerExtension(const char* layer, const char* name) const {
        uint32_t count = 0;
        checkVk(vkEnumerateInstanceExtensionProperties(layer, &count, nullptr),
                "vkEnumerateInstanceExtensionProperties (validation layer)");
        std::vector<VkExtensionProperties> extensions(count);
        checkVk(vkEnumerateInstanceExtensionProperties(layer, &count, extensions.data()),
                "vkEnumerateInstanceExtensionProperties (validation layer)");
        return std::any_of(extensions.begin(), extensions.end(),
                           [name](const VkExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName, name) == 0;
                           });
    }

    // The CPU verifies Vulkan 1.3, GLFW surface extensions, debug-utils, and
    // validation, then creates the instance and its persistent diagnostic callback.
    void createInstance() {
        uint32_t loaderVersion = VK_API_VERSION_1_0;
        auto enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        if (enumerateVersion != nullptr) {
            checkVk(enumerateVersion(&loaderVersion), "vkEnumerateInstanceVersion");
        }
        if (VK_VERSION_MAJOR(loaderVersion) < 1 ||
            (VK_VERSION_MAJOR(loaderVersion) == 1 && VK_VERSION_MINOR(loaderVersion) < 3)) {
            throw std::runtime_error("Vulkan 1.3 loader is required");
        }
        if (!hasInstanceLayer(kValidationLayer)) {
            throw std::runtime_error("Required validation layer is unavailable: " +
                                     std::string(kValidationLayer));
        }
        if (!hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            throw std::runtime_error("Required instance extension is unavailable: " +
                                     std::string(VK_EXT_DEBUG_UTILS_EXTENSION_NAME));
        }
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions =
            glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions");
        }
        std::vector<const char*> extensions(glfwExtensions,
                                            glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        const bool synchronizationValidationAvailable =
            hasLayerExtension(kValidationLayer, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        if (synchronizationValidationAvailable) {
            extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        }
        for (const char* extension : extensions) {
            if (std::strcmp(extension, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) != 0 &&
                !hasInstanceExtension(extension)) {
                throw std::runtime_error("Required instance extension is unavailable: " +
                                         std::string(extension));
            }
        }
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Vulkan Pyramid";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 2, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;
        VkDebugUtilsMessengerCreateInfoEXT debugInfo = debugMessengerInfo();
        const VkValidationFeatureEnableEXT enabledFeature =
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        VkValidationFeaturesEXT validationFeatures{
            VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
        validationFeatures.enabledValidationFeatureCount = 1;
        validationFeatures.pEnabledValidationFeatures = &enabledFeature;
        validationFeatures.pNext = &debugInfo;
        VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfo.pNext = synchronizationValidationAvailable
                               ? static_cast<const void*>(&validationFeatures)
                               : static_cast<const void*>(&debugInfo);
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &kValidationLayer;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        checkVk(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
        if (synchronizationValidationAvailable) {
            std::cout << "Synchronization validation enabled\n";
        }
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (createMessenger == nullptr) {
            throw std::runtime_error("vkCreateDebugUtilsMessengerEXT is unavailable");
        }
        checkVk(createMessenger(instance_, &debugInfo, nullptr, &debugMessenger_),
                "vkCreateDebugUtilsMessengerEXT");
    }

    QueueFamilies findQueueFamilies(VkPhysicalDevice candidate) const {
        QueueFamilies families;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, properties.data());
        for (uint32_t i = 0; i < count; ++i) {
            const VkQueueFlags flags = properties[i].queueFlags;
            if (!families.graphics && (flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                families.graphics = i;
            }
            VkBool32 supportsPresent = VK_FALSE;
            checkVk(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_,
                                                         &supportsPresent),
                    "vkGetPhysicalDeviceSurfaceSupportKHR");
            if (!families.present && supportsPresent == VK_TRUE) {
                families.present = i;
            }
            if (!families.compute && (flags & VK_QUEUE_COMPUTE_BIT) != 0) {
                families.compute = i;
            }
            if (!families.transfer && (flags & VK_QUEUE_TRANSFER_BIT) != 0) {
                families.transfer = i;
            }
        }
        if (families.graphics &&
            (properties[*families.graphics].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            families.compute = families.graphics;
        }
        if (families.graphics) {
            // Graphics-capable queues support transfer commands even when the
            // transfer flag is not separately advertised; keep uploads local.
            families.transfer = families.graphics;
        }
        return families;
    }

    bool hasDeviceExtension(VkPhysicalDevice candidate, const char* name) const {
        uint32_t count = 0;
        checkVk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr),
                "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> extensions(count);
        checkVk(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count,
                                                     extensions.data()),
                "vkEnumerateDeviceExtensionProperties");
        return std::any_of(extensions.begin(), extensions.end(),
                           [name](const VkExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName, name) == 0;
                           });
    }

    SwapchainSupport querySwapchainSupport(VkPhysicalDevice candidate) const {
        SwapchainSupport support;
        checkVk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate, surface_,
                                                          &support.capabilities),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        uint32_t count = 0;
        checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &count, nullptr),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        support.formats.resize(count);
        if (count > 0) {
            checkVk(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &count,
                                                         support.formats.data()),
                    "vkGetPhysicalDeviceSurfaceFormatsKHR");
        }
        count = 0;
        checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &count, nullptr),
                "vkGetPhysicalDeviceSurfacePresentModesKHR");
        support.presentModes.resize(count);
        if (count > 0) {
            checkVk(vkGetPhysicalDeviceSurfacePresentModesKHR(
                        candidate, surface_, &count, support.presentModes.data()),
                    "vkGetPhysicalDeviceSurfacePresentModesKHR");
        }
        return support;
    }

    bool isDeviceSuitable(VkPhysicalDevice candidate) const {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (VK_VERSION_MAJOR(properties.apiVersion) < 1 ||
            (VK_VERSION_MAJOR(properties.apiVersion) == 1 &&
             VK_VERSION_MINOR(properties.apiVersion) < 3)) {
            return false;
        }
        const QueueFamilies families = findQueueFamilies(candidate);
        if (!families.complete() || !hasDeviceExtension(candidate, kSwapchainExtension)) {
            return false;
        }
        VkPhysicalDeviceVulkan13Features features13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features features12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features12.pNext = &features13;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features12;
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        if (features12.timelineSemaphore != VK_TRUE ||
            features13.dynamicRendering != VK_TRUE ||
            features13.synchronization2 != VK_TRUE ||
            properties.limits.maxPushConstantsSize < sizeof(glm::vec4) ||
            properties.limits.maxComputeWorkGroupInvocations < 1) {
            return false;
        }
        const SwapchainSupport support = querySwapchainSupport(candidate);
        return !support.formats.empty() && !support.presentModes.empty();
    }

    // Inspect relevant features/limits before choosing; device suitability also
    // requires Vulkan 1.3, swapchain, compute, transfer, and present-capable queues.
    void pickPhysicalDevice() {
        uint32_t count = 0;
        checkVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                "vkEnumeratePhysicalDevices");
        if (count == 0) {
            throw std::runtime_error("No Vulkan-capable physical device was found");
        }
        std::vector<VkPhysicalDevice> devices(count);
        checkVk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
                "vkEnumeratePhysicalDevices");
        int bestScore = -1;
        for (VkPhysicalDevice candidate : devices) {
            if (!isDeviceSuitable(candidate)) {
                continue;
            }
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            int score = 1;
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score = 3;
            } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score = 2;
            }
            if (score > bestScore) {
                bestScore = score;
                physicalDevice_ = candidate;
                physicalDeviceProperties_ = properties;
                queueFamilies_ = findQueueFamilies(candidate);
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "No suitable Vulkan 1.3 device with required queues, swapchain, "
                "dynamic rendering, and synchronization2 support");
        }
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);
        std::cout << "Using Vulkan device: " << physicalDeviceProperties_.deviceName
                  << " (API " << VK_VERSION_MAJOR(physicalDeviceProperties_.apiVersion)
                  << '.' << VK_VERSION_MINOR(physicalDeviceProperties_.apiVersion) << ")\n"
                  << "Queue families: graphics=" << *queueFamilies_.graphics
                  << ", present=" << *queueFamilies_.present
                  << ", compute=" << *queueFamilies_.compute
                  << ", transfer=" << *queueFamilies_.transfer << '\n'
                  << "Limits: push constants="
                  << physicalDeviceProperties_.limits.maxPushConstantsSize
                  << " bytes; uniform range="
                  << physicalDeviceProperties_.limits.maxUniformBufferRange
                  << " bytes; compute invocations="
                  << physicalDeviceProperties_.limits.maxComputeWorkGroupInvocations
                  << "; image dimension="
                  << physicalDeviceProperties_.limits.maxImageDimension2D << '\n';
    }

    // Enable the queried Vulkan 1.3 features and create one queue per unique family;
    // roles sharing a family intentionally share the family queue and need no ownership transfer.
    void createLogicalDevice() {
        const std::set<uint32_t> uniqueFamilies = {
            *queueFamilies_.graphics, *queueFamilies_.present,
            *queueFamilies_.compute, *queueFamilies_.transfer};
        const float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo queueInfo{
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueInfo.queueFamilyIndex = family;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueInfos.push_back(queueInfo);
        }
        const char* extensions[] = {kSwapchainExtension};
        VkPhysicalDeviceVulkan13Features features13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features13.dynamicRendering = VK_TRUE;
        features13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceVulkan12Features features12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features12.timelineSemaphore = VK_TRUE;
        features12.pNext = &features13;
        VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        info.pNext = &features12;
        info.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
        info.pQueueCreateInfos = queueInfos.data();
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = extensions;
        checkVk(vkCreateDevice(physicalDevice_, &info, nullptr, &device_),
                "vkCreateDevice");
        vkGetDeviceQueue(device_, *queueFamilies_.graphics, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, *queueFamilies_.present, 0, &presentQueue_);
        vkGetDeviceQueue(device_, *queueFamilies_.compute, 0, &computeQueue_);
        vkGetDeviceQueue(device_, *queueFamilies_.transfer, 0, &transferQueue_);
    }

    // Pools are family-owned, so role-specific pools remain valid even if roles alias
    // one family. Resettable pools support primary/secondary recording and reuse.
    void createCommandPools() {
        auto createPool = [this](uint32_t family, VkCommandPool* pool,
                                 const char* operation) {
            VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            info.queueFamilyIndex = family;
            checkVk(vkCreateCommandPool(device_, &info, nullptr, pool), operation);
        };
        createPool(*queueFamilies_.graphics, &graphicsCommandPool_,
                   "vkCreateCommandPool (graphics)");
        createPool(*queueFamilies_.compute, &computeCommandPool_,
                   "vkCreateCommandPool (compute)");
        createPool(*queueFamilies_.transfer, &transferCommandPool_,
                   "vkCreateCommandPool (transfer)");
    }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred) const {
        std::optional<uint32_t> fallback;
        for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) == 0) {
                continue;
            }
            const VkMemoryPropertyFlags flags =
                memoryProperties_.memoryTypes[i].propertyFlags;
            if ((flags & required) != required) {
                continue;
            }
            if ((flags & preferred) == preferred) {
                return i;
            }
            if (!fallback) {
                fallback = i;
            }
        }
        if (fallback) {
            return *fallback;
        }
        throw std::runtime_error("No compatible Vulkan memory type was found");
    }

    // VkBuffer requirements determine legal allocation size and memory types.
    // Allocate and bind natively; offset zero satisfies the returned alignment.
    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags required,
                        VkMemoryPropertyFlags preferred) {
        Buffer buffer;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        checkVk(vkCreateBuffer(device_, &info, nullptr, &buffer.handle),
                "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
        uint32_t memoryType = 0;
        try {
            memoryType =
                findMemoryType(requirements.memoryTypeBits, required, preferred);
        } catch (...) {
            vkDestroyBuffer(device_, buffer.handle, nullptr);
            throw;
        }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType;
        const VkResult allocateResult =
            vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory);
        if (allocateResult != VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer.handle, nullptr);
            checkVk(allocateResult, "vkAllocateMemory (buffer)");
        }
        buffer.allocationSize = requirements.size;
        buffer.coherent = (memoryProperties_.memoryTypes[memoryType].propertyFlags &
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        try {
            checkVk(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0),
                    "vkBindBufferMemory");
        } catch (...) {
            vkDestroyBuffer(device_, buffer.handle, nullptr);
            vkFreeMemory(device_, buffer.memory, nullptr);
            throw;
        }
        return buffer;
    }

    void flushBuffer(const Buffer& buffer) const {
        if (buffer.coherent) {
            return;
        }
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer.memory;
        range.size = VK_WHOLE_SIZE;
        checkVk(vkFlushMappedMemoryRanges(device_, 1, &range),
                "vkFlushMappedMemoryRanges");
    }

    void destroyBuffer(Buffer* buffer) {
        if (buffer->mapped != nullptr) {
            vkUnmapMemory(device_, buffer->memory);
            buffer->mapped = nullptr;
        }
        if (buffer->handle != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer->handle, nullptr);
            buffer->handle = VK_NULL_HANDLE;
        }
        if (buffer->memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buffer->memory, nullptr);
            buffer->memory = VK_NULL_HANDLE;
        }
    }

    void beginOneTimeCommands(VkCommandPool pool, VkCommandBuffer* commandBuffer) {
        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        checkVk(vkAllocateCommandBuffers(device_, &allocation, commandBuffer),
                "vkAllocateCommandBuffers (one-time)");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(*commandBuffer, &begin), "vkBeginCommandBuffer");
    }

    // Submit upload work with a fence instead of queueWaitIdle; the host waits only
    // for the copied staging data to become reusable.
    void submitOneTimeCommands(VkCommandPool pool, VkQueue queue,
                               VkCommandBuffer commandBuffer) {
        checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        VkCommandBufferSubmitInfo commandInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &commandInfo;
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        checkVk(vkCreateFence(device_, &fenceInfo, nullptr, &fence),
                "vkCreateFence (upload)");
        const VkResult submitResult = vkQueueSubmit2(queue, 1, &submit, fence);
        if (submitResult != VK_SUCCESS) {
            vkDestroyFence(device_, fence, nullptr);
            throw std::runtime_error("vkQueueSubmit2 (upload) failed (VkResult " +
                                     std::to_string(submitResult) + ")");
        }
        checkVk(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
                "vkWaitForFences (upload)");
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, pool, 1, &commandBuffer);
    }

    void createGeometryBuffers() {
        const std::array<Vertex, 18> vertices = {{
            {{0.0f, 0.9f, 0.0f}, {1.0f, 0.25f, 0.18f}, {0.5f, 1.0f}},
            {{-0.7f, -0.7f, -0.7f}, {1.0f, 0.25f, 0.18f}, {0.0f, 0.0f}},
            {{0.7f, -0.7f, -0.7f}, {1.0f, 0.25f, 0.18f}, {1.0f, 0.0f}},
            {{0.0f, 0.9f, 0.0f}, {0.2f, 0.8f, 0.95f}, {0.5f, 1.0f}},
            {{0.7f, -0.7f, -0.7f}, {0.2f, 0.8f, 0.95f}, {0.0f, 0.0f}},
            {{0.7f, -0.7f, 0.7f}, {0.2f, 0.8f, 0.95f}, {1.0f, 0.0f}},
            {{0.0f, 0.9f, 0.0f}, {0.95f, 0.75f, 0.2f}, {0.5f, 1.0f}},
            {{0.7f, -0.7f, 0.7f}, {0.95f, 0.75f, 0.2f}, {0.0f, 0.0f}},
            {{-0.7f, -0.7f, 0.7f}, {0.95f, 0.75f, 0.2f}, {1.0f, 0.0f}},
            {{0.0f, 0.9f, 0.0f}, {0.55f, 0.35f, 0.95f}, {0.5f, 1.0f}},
            {{-0.7f, -0.7f, 0.7f}, {0.55f, 0.35f, 0.95f}, {0.0f, 0.0f}},
            {{-0.7f, -0.7f, -0.7f}, {0.55f, 0.35f, 0.95f}, {1.0f, 0.0f}},
            {{-0.7f, -0.7f, -0.7f}, {0.2f, 0.8f, 0.4f}, {0.0f, 0.0f}},
            {{0.7f, -0.7f, 0.7f}, {0.2f, 0.8f, 0.4f}, {1.0f, 1.0f}},
            {{0.7f, -0.7f, -0.7f}, {0.2f, 0.8f, 0.4f}, {1.0f, 0.0f}},
            {{-0.7f, -0.7f, -0.7f}, {0.2f, 0.8f, 0.4f}, {0.0f, 0.0f}},
            {{-0.7f, -0.7f, 0.7f}, {0.2f, 0.8f, 0.4f}, {0.0f, 1.0f}},
            {{0.7f, -0.7f, 0.7f}, {0.2f, 0.8f, 0.4f}, {1.0f, 1.0f}}
        }};
        constexpr std::array<uint16_t, 18> indices = {
            0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 17};
        const VkDeviceSize vertexBytes = sizeof(vertices);
        const VkDeviceSize indexBytes = sizeof(indices);
        const VkDeviceSize stagingBytes = vertexBytes + indexBytes;
        Buffer staging = createBuffer(
            stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        try {
            void* mapped = nullptr;
            checkVk(vkMapMemory(device_, staging.memory, 0,
                                staging.allocationSize, 0, &mapped),
                    "vkMapMemory (geometry staging)");
            staging.mapped = mapped;
            std::memcpy(mapped, vertices.data(), sizeof(vertices));
            std::memcpy(static_cast<char*>(mapped) + vertexBytes,
                        indices.data(), sizeof(indices));
            flushBuffer(staging);
            vkUnmapMemory(device_, staging.memory);
            staging.mapped = nullptr;

            vertexBuffer_ = createBuffer(
                vertexBytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            indexBuffer_ = createBuffer(
                indexBytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            beginOneTimeCommands(transferCommandPool_, &commandBuffer);
            const std::array<VkBufferCopy, 2> copies = {{
                {0, 0, vertexBytes},
                {vertexBytes, 0, indexBytes}
            }};
            vkCmdCopyBuffer(commandBuffer, staging.handle,
                            vertexBuffer_.handle, 1, &copies[0]);
            vkCmdCopyBuffer(commandBuffer, staging.handle,
                            indexBuffer_.handle, 1, &copies[1]);
            std::array<VkBufferMemoryBarrier2, 2> barriers{};
            barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            barriers[0].buffer = vertexBuffer_.handle;
            barriers[0].size = VK_WHOLE_SIZE;
            barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_2_INDEX_READ_BIT;
            barriers[1].buffer = indexBuffer_.handle;
            barriers[1].size = VK_WHOLE_SIZE;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.bufferMemoryBarrierCount =
                static_cast<uint32_t>(barriers.size());
            dependency.pBufferMemoryBarriers = barriers.data();
            vkCmdPipelineBarrier2(commandBuffer, &dependency);
            submitOneTimeCommands(transferCommandPool_, transferQueue_, commandBuffer);
        } catch (...) {
            destroyBuffer(&staging);
            throw;
        }
        destroyBuffer(&staging);
    }

    VkFormat findDepthFormat() const {
        constexpr std::array<VkFormat, 3> candidates = {
            VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT};
        for (VkFormat format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);
            if ((properties.optimalTilingFeatures &
                 VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth attachment format was found");
    }

    ImageAllocation createImage(uint32_t width, uint32_t height, VkFormat format,
                                VkImageUsageFlags usage, VkImageTiling tiling,
                                VkMemoryPropertyFlags properties) {
        ImageAllocation allocation;
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        checkVk(vkCreateImage(device_, &imageInfo, nullptr, &allocation.image),
                "vkCreateImage");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, allocation.image, &requirements);
        VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocateInfo.allocationSize = requirements.size;
        try {
            allocateInfo.memoryTypeIndex =
                findMemoryType(requirements.memoryTypeBits, properties, properties);
        } catch (...) {
            vkDestroyImage(device_, allocation.image, nullptr);
            throw;
        }
        const VkResult allocateResult =
            vkAllocateMemory(device_, &allocateInfo, nullptr, &allocation.memory);
        if (allocateResult != VK_SUCCESS) {
            vkDestroyImage(device_, allocation.image, nullptr);
            checkVk(allocateResult, "vkAllocateMemory (image)");
        }
        try {
            checkVk(vkBindImageMemory(device_, allocation.image, allocation.memory, 0),
                    "vkBindImageMemory");
        } catch (...) {
            vkDestroyImage(device_, allocation.image, nullptr);
            vkFreeMemory(device_, allocation.memory, nullptr);
            throw;
        }
        return allocation;
    }

    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspect) {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange.aspectMask = aspect;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        checkVk(vkCreateImageView(device_, &info, nullptr, &view), "vkCreateImageView");
        return view;
    }

    // Stage a single white texel. Sampling it is visually neutral but exercises a
    // real optimal-tiled image, view, sampler, image barriers, and combined descriptor.
    void createTexture() {
        const std::array<uint8_t, 4> texel = {255, 255, 255, 255};
        Buffer staging = createBuffer(
            texel.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* mapped = nullptr;
        try {
            checkVk(vkMapMemory(device_, staging.memory, 0, staging.allocationSize,
                                0, &mapped),
                    "vkMapMemory (texture staging)");
            staging.mapped = mapped;
            std::memcpy(mapped, texel.data(), texel.size());
            flushBuffer(staging);
            vkUnmapMemory(device_, staging.memory);
            staging.mapped = nullptr;
        } catch (...) {
            destroyBuffer(&staging);
            throw;
        }
        try {
            texture_ = createImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT,
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            beginOneTimeCommands(transferCommandPool_, &commandBuffer);
            VkImageMemoryBarrier2 toTransfer{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.image = texture_.image;
            toTransfer.subresourceRange =
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo toTransferDependency{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            toTransferDependency.imageMemoryBarrierCount = 1;
            toTransferDependency.pImageMemoryBarriers = &toTransfer;
            vkCmdPipelineBarrier2(commandBuffer, &toTransferDependency);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {1, 1, 1};
            vkCmdCopyBufferToImage(commandBuffer, staging.handle, texture_.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            VkImageMemoryBarrier2 toSample{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            toSample.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            toSample.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            toSample.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            toSample.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            toSample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toSample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toSample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSample.image = texture_.image;
            toSample.subresourceRange =
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo toSampleDependency{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            toSampleDependency.imageMemoryBarrierCount = 1;
            toSampleDependency.pImageMemoryBarriers = &toSample;
            vkCmdPipelineBarrier2(commandBuffer, &toSampleDependency);
            submitOneTimeCommands(transferCommandPool_, transferQueue_, commandBuffer);
        } catch (...) {
            destroyBuffer(&staging);
            throw;
        }
        destroyBuffer(&staging);
        textureView_ = createImageView(texture_.image, VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_ASPECT_COLOR_BIT);
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.maxLod = 0.0f;
        checkVk(vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler_),
                "vkCreateSampler");
    }

    // Allocate camera and compute output independently per frame in flight; this
    // prevents CPU updates or compute writes from racing a prior graphics read.
    void createFrameBuffers() {
        const std::array<uint32_t, 2> computeGraphicsFamilies = {
            *queueFamilies_.compute, *queueFamilies_.graphics};
        const bool sharedCompute = queueFamilies_.compute == queueFamilies_.graphics;
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            cameraBuffers_[i] = createBuffer(
                sizeof(CameraUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            checkVk(vkMapMemory(device_, cameraBuffers_[i].memory, 0,
                                cameraBuffers_[i].allocationSize, 0,
                                &cameraBuffers_[i].mapped),
                    "vkMapMemory (camera uniform)");
            VkBufferCreateInfo outputInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            outputInfo.size = sizeof(glm::vec4);
            outputInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            if (sharedCompute) {
                outputInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            } else {
                outputInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
                outputInfo.queueFamilyIndexCount =
                    static_cast<uint32_t>(computeGraphicsFamilies.size());
                outputInfo.pQueueFamilyIndices = computeGraphicsFamilies.data();
            }
            Buffer& output = computeOutputBuffers_[i];
            checkVk(vkCreateBuffer(device_, &outputInfo, nullptr, &output.handle),
                    "vkCreateBuffer (compute output)");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device_, output.handle, &requirements);
            VkMemoryAllocateInfo allocateInfo{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocateInfo.allocationSize = requirements.size;
            allocateInfo.memoryTypeIndex = findMemoryType(
                requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            checkVk(vkAllocateMemory(device_, &allocateInfo, nullptr, &output.memory),
                    "vkAllocateMemory (compute output)");
            output.allocationSize = requirements.size;
            checkVk(vkBindBufferMemory(device_, output.handle, output.memory, 0),
                    "vkBindBufferMemory (compute output)");
        }
    }

    // Layouts are the binding contracts consumed by graphics/compute pipelines;
    // binding numbers and stage visibility must match the GLSL set/binding declarations.
    void createDescriptorLayouts() {
        std::array<VkDescriptorSetLayoutBinding, 3> graphicsBindings{};
        graphicsBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                               VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        graphicsBindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                               VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        graphicsBindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                               VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo graphicsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        graphicsInfo.bindingCount = static_cast<uint32_t>(graphicsBindings.size());
        graphicsInfo.pBindings = graphicsBindings.data();
        checkVk(vkCreateDescriptorSetLayout(device_, &graphicsInfo, nullptr,
                                            &graphicsSetLayout_),
                "vkCreateDescriptorSetLayout (graphics)");
        VkDescriptorSetLayoutBinding computeBinding{};
        computeBinding.binding = 0;
        computeBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        computeBinding.descriptorCount = 1;
        computeBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo computeInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        computeInfo.bindingCount = 1;
        computeInfo.pBindings = &computeBinding;
        checkVk(vkCreateDescriptorSetLayout(device_, &computeInfo, nullptr,
                                            &computeSetLayout_),
                "vkCreateDescriptorSetLayout (compute)");
        VkPipelineLayoutCreateInfo graphicsLayoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        graphicsLayoutInfo.setLayoutCount = 1;
        graphicsLayoutInfo.pSetLayouts = &graphicsSetLayout_;
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRange.size = sizeof(glm::vec4);
        graphicsLayoutInfo.pushConstantRangeCount = 1;
        graphicsLayoutInfo.pPushConstantRanges = &pushRange;
        checkVk(vkCreatePipelineLayout(device_, &graphicsLayoutInfo, nullptr,
                                       &graphicsPipelineLayout_),
                "vkCreatePipelineLayout (graphics)");
        VkPipelineLayoutCreateInfo computeLayoutInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        computeLayoutInfo.setLayoutCount = 1;
        computeLayoutInfo.pSetLayouts = &computeSetLayout_;
        checkVk(vkCreatePipelineLayout(device_, &computeLayoutInfo, nullptr,
                                       &computePipelineLayout_),
                "vkCreatePipelineLayout (compute)");
    }

    // Pools allocate frame-owned descriptor sets. vkUpdateDescriptorSets writes
    // references to buffers/images; shaders consume those references during dispatch/draw.
    void createDescriptorSets() {
        const std::array<VkDescriptorPoolSize, 3> graphicsSizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFramesInFlight}
        }};
        VkDescriptorPoolCreateInfo graphicsPoolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        graphicsPoolInfo.maxSets = kFramesInFlight;
        graphicsPoolInfo.poolSizeCount = static_cast<uint32_t>(graphicsSizes.size());
        graphicsPoolInfo.pPoolSizes = graphicsSizes.data();
        checkVk(vkCreateDescriptorPool(device_, &graphicsPoolInfo, nullptr,
                                       &graphicsDescriptorPool_),
                "vkCreateDescriptorPool (graphics)");
        const std::array<VkDescriptorSetLayout, kFramesInFlight> graphicsLayouts = {
            graphicsSetLayout_, graphicsSetLayout_};
        VkDescriptorSetAllocateInfo graphicsAllocate{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        graphicsAllocate.descriptorPool = graphicsDescriptorPool_;
        graphicsAllocate.descriptorSetCount = kFramesInFlight;
        graphicsAllocate.pSetLayouts = graphicsLayouts.data();
        checkVk(vkAllocateDescriptorSets(device_, &graphicsAllocate,
                                         graphicsDescriptorSets_.data()),
                "vkAllocateDescriptorSets (graphics)");
        const VkDescriptorPoolSize computeSize{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFramesInFlight};
        VkDescriptorPoolCreateInfo computePoolInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        computePoolInfo.maxSets = kFramesInFlight;
        computePoolInfo.poolSizeCount = 1;
        computePoolInfo.pPoolSizes = &computeSize;
        checkVk(vkCreateDescriptorPool(device_, &computePoolInfo, nullptr,
                                       &computeDescriptorPool_),
                "vkCreateDescriptorPool (compute)");
        const std::array<VkDescriptorSetLayout, kFramesInFlight> computeLayouts = {
            computeSetLayout_, computeSetLayout_};
        VkDescriptorSetAllocateInfo computeAllocate{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        computeAllocate.descriptorPool = computeDescriptorPool_;
        computeAllocate.descriptorSetCount = kFramesInFlight;
        computeAllocate.pSetLayouts = computeLayouts.data();
        checkVk(vkAllocateDescriptorSets(device_, &computeAllocate,
                                         computeDescriptorSets_.data()),
                "vkAllocateDescriptorSets (compute)");
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            VkDescriptorBufferInfo uniformInfo{
                cameraBuffers_[i].handle, 0, sizeof(CameraUniform)};
            VkDescriptorImageInfo imageInfo{
                textureSampler_, textureView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo graphicsStorageInfo{
                computeOutputBuffers_[i].handle, 0, sizeof(glm::vec4)};
            VkDescriptorBufferInfo computeStorageInfo{
                computeOutputBuffers_[i].handle, 0, sizeof(glm::vec4)};
            std::array<VkWriteDescriptorSet, 3> writes{};
            writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         graphicsDescriptorSets_[i], 0, 0, 1,
                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr,
                         &uniformInfo, nullptr};
            writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         graphicsDescriptorSets_[i], 1, 0, 1,
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &imageInfo,
                         nullptr, nullptr};
            writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         graphicsDescriptorSets_[i], 2, 0, 1,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
                         &graphicsStorageInfo, nullptr};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
            VkWriteDescriptorSet computeWrite{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            computeWrite.dstSet = computeDescriptorSets_[i];
            computeWrite.dstBinding = 0;
            computeWrite.descriptorCount = 1;
            computeWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            computeWrite.pBufferInfo = &computeStorageInfo;
            vkUpdateDescriptorSets(device_, 1, &computeWrite, 0, nullptr);
        }
    }

    VkSurfaceFormatKHR chooseSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& formats) const {
        if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED) {
            return {VK_FORMAT_B8G8R8A8_SRGB, formats.front().colorSpace};
        }
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats.front();
    }

    VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(
        const VkSurfaceCapabilitiesKHR& capabilities) const {
        constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> preferences = {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
        for (VkCompositeAlphaFlagBitsKHR alpha : preferences) {
            if ((capabilities.supportedCompositeAlpha & alpha) != 0) {
                return alpha;
            }
        }
        throw std::runtime_error("Surface has no supported composite alpha mode");
    }

    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        }
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        VkExtent2D extent{static_cast<uint32_t>(std::max(width, 0)),
                          static_cast<uint32_t>(std::max(height, 0))};
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
        return extent;
    }

    // A swapchain image is presentation-owned; its view/depth attachment/semaphore
    // are indexed by acquired image, not by the CPU frame currently in flight.
    void createSwapchain() {
        const SwapchainSupport support = querySwapchainSupport(physicalDevice_);
        const VkSurfaceFormatKHR format = chooseSurfaceFormat(support.formats);
        const VkExtent2D extent = chooseExtent(support.capabilities);
        if (extent.width == 0 || extent.height == 0) {
            throw std::runtime_error("Cannot create a swapchain for a zero-sized framebuffer");
        }
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 &&
            imageCount > support.capabilities.maxImageCount) {
            imageCount = support.capabilities.maxImageCount;
        }
        VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        info.surface = surface_;
        info.minImageCount = imageCount;
        info.imageFormat = format.format;
        info.imageColorSpace = format.colorSpace;
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        const std::array<uint32_t, 2> sharingFamilies = {
            *queueFamilies_.graphics, *queueFamilies_.present};
        if (queueFamilies_.graphics != queueFamilies_.present) {
            info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            info.queueFamilyIndexCount = static_cast<uint32_t>(sharingFamilies.size());
            info.pQueueFamilyIndices = sharingFamilies.data();
        } else {
            info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        info.preTransform = support.capabilities.currentTransform;
        info.compositeAlpha = chooseCompositeAlpha(support.capabilities);
        info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        info.clipped = VK_TRUE;
        checkVk(vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_),
                "vkCreateSwapchainKHR");
        checkVk(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr),
                "vkGetSwapchainImagesKHR");
        swapchainImages_.resize(imageCount);
        checkVk(vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount,
                                        swapchainImages_.data()),
                "vkGetSwapchainImagesKHR");
        swapchainImageFormat_ = format.format;
        swapchainExtent_ = extent;
        swapchainImageViews_.reserve(swapchainImages_.size());
        for (VkImage image : swapchainImages_) {
            swapchainImageViews_.push_back(
                createImageView(image, swapchainImageFormat_, VK_IMAGE_ASPECT_COLOR_BIT));
        }
        presentReadySemaphores_.resize(swapchainImages_.size(), VK_NULL_HANDLE);
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (VkSemaphore& semaphore : presentReadySemaphores_) {
            checkVk(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &semaphore),
                    "vkCreateSemaphore (per swapchain image)");
        }
    }

    void createDepthResources() {
        depthFormat_ = findDepthFormat();
        depthImages_.resize(swapchainImages_.size());
        depthImageViews_.resize(swapchainImages_.size(), VK_NULL_HANDLE);
        for (size_t i = 0; i < depthImages_.size(); ++i) {
            depthImages_[i] = createImage(
                swapchainExtent_.width, swapchainExtent_.height, depthFormat_,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_TILING_OPTIMAL,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            depthImageViews_[i] = createImageView(
                depthImages_[i].image, depthFormat_, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
    }

    // The CPU describes attachment formats, layouts, clears, and subpass use;
    // the render-pass pipeline and framebuffer later consume this compatibility contract.
    void createRenderPass() {
        VkAttachmentDescription color{};
        color.format = swapchainImageFormat_;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription depth{};
        depth.format = depthFormat_;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthReference{
            1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        subpass.pDepthStencilAttachment = &depthReference;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = dependency.srcStageMask;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        const std::array<VkAttachmentDescription, 2> attachments = {color, depth};
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dependency;
        checkVk(vkCreateRenderPass(device_, &info, nullptr, &renderPass_),
                "vkCreateRenderPass");
    }

    std::vector<char> readShader(const std::string& name) const {
        const std::string path = std::string(SHADER_DIR) + "/" + name;
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open shader: " + path);
        }
        const auto size = file.tellg();
        if (size <= 0 || size % 4 != 0) {
            throw std::runtime_error("Shader has an invalid SPIR-V size: " + path);
        }
        std::vector<char> code(static_cast<size_t>(size));
        file.seekg(0);
        file.read(code.data(), size);
        if (!file) {
            throw std::runtime_error("Could not read shader: " + path);
        }
        return code;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) const {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = code.size();
        info.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module = VK_NULL_HANDLE;
        checkVk(vkCreateShaderModule(device_, &info, nullptr, &module),
                "vkCreateShaderModule");
        return module;
    }

    // Pipeline creation compiles fixed-function and shader state in the driver;
    // the returned pipeline is bound by a secondary graphics command buffer.
    VkPipeline createGraphicsPipeline(VkShaderModule vertexModule,
                                     VkShaderModule fragmentModule,
                                     VkRenderPass compatibleRenderPass) {
        const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
             VK_SHADER_STAGE_VERTEX_BIT, vertexModule, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
             VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main", nullptr}
        }};
        const VkVertexInputBindingDescription binding{
            0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array<VkVertexInputAttributeDescription, 3> attributes = {{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)}
        }};
        VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewportState{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterizer{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo multisampling{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depthStencil{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.maxDepthBounds = 1.0f;
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                         VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT |
                                         VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blending{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount = 1;
        blending.pAttachments = &blendAttachment;
        const std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        VkPipelineRenderingCreateInfo renderingInfo{
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &swapchainImageFormat_;
        renderingInfo.depthAttachmentFormat = depthFormat_;
        VkGraphicsPipelineCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &blending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = graphicsPipelineLayout_;
        pipelineInfo.renderPass = compatibleRenderPass;
        pipelineInfo.subpass = 0;
        if (compatibleRenderPass == VK_NULL_HANDLE) {
            pipelineInfo.pNext = &renderingInfo;
        }
        VkPipeline pipeline = VK_NULL_HANDLE;
        checkVk(vkCreateGraphicsPipelines(device_, pipelineCache_, 1, &pipelineInfo,
                                          nullptr, &pipeline),
                "vkCreateGraphicsPipelines");
        return pipeline;
    }

    // Both selectable pipelines use identical shaders, layouts, and pipeline cache.
    void createGraphicsPipelines() {
        const auto vertexCode = readShader("triangle.vert.spv");
        const auto fragmentCode = readShader("triangle.frag.spv");
        const VkShaderModule vertexModule = createShaderModule(vertexCode);
        VkShaderModule fragmentModule = VK_NULL_HANDLE;
        try {
            fragmentModule = createShaderModule(fragmentCode);
            graphicsPipeline_ =
                createGraphicsPipeline(vertexModule, fragmentModule, renderPass_);
            dynamicRenderingPipeline_ =
                createGraphicsPipeline(vertexModule, fragmentModule, VK_NULL_HANDLE);
        } catch (...) {
            if (fragmentModule != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, fragmentModule, nullptr);
            }
            vkDestroyShaderModule(device_, vertexModule, nullptr);
            throw;
        }
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, vertexModule, nullptr);
    }

    // Create the compute shader pipeline from its SPIR-V module and compute layout;
    // recorded dispatch commands bind this pipeline before writing the frame storage buffer.
    void createComputePipeline() {
        const VkShaderModule module = createShaderModule(readShader("compute.comp.spv"));
        VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = stage;
        info.layout = computePipelineLayout_;
        const VkResult result = vkCreateComputePipelines(
            device_, pipelineCache_, 1, &info, nullptr, &computePipeline_);
        vkDestroyShaderModule(device_, module, nullptr);
        checkVk(result, "vkCreateComputePipelines");
    }

    void createPipelineCache() {
        VkPipelineCacheCreateInfo info{
            VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        checkVk(vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_),
                "vkCreatePipelineCache");
    }

    // Framebuffers bind swapchain and depth views to the render pass; render-pass
    // commands select one framebuffer using the acquired swapchain image index.
    void createFramebuffers() {
        framebuffers_.reserve(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            const std::array<VkImageView, 2> attachments = {
                swapchainImageViews_[i], depthImageViews_[i]};
            VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            info.renderPass = renderPass_;
            info.attachmentCount = static_cast<uint32_t>(attachments.size());
            info.pAttachments = attachments.data();
            info.width = swapchainExtent_.width;
            info.height = swapchainExtent_.height;
            info.layers = 1;
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            checkVk(vkCreateFramebuffer(device_, &info, nullptr, &framebuffer),
                    "vkCreateFramebuffer");
            framebuffers_.push_back(framebuffer);
        }
    }

    void allocateFrameCommandBuffers() {
        primaryCommandBuffers_.resize(swapchainImages_.size());
        secondaryCommandBuffers_.resize(swapchainImages_.size());
        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = graphicsCommandPool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount =
            static_cast<uint32_t>(primaryCommandBuffers_.size());
        checkVk(vkAllocateCommandBuffers(device_, &allocation,
                                        primaryCommandBuffers_.data()),
                "vkAllocateCommandBuffers (graphics primary)");
        allocation.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        allocation.commandBufferCount =
            static_cast<uint32_t>(secondaryCommandBuffers_.size());
        checkVk(vkAllocateCommandBuffers(device_, &allocation,
                                        secondaryCommandBuffers_.data()),
                "vkAllocateCommandBuffers (graphics secondary)");
        imagesInFlight_.assign(swapchainImages_.size(), VK_NULL_HANDLE);
    }

    void createComputeCommandBuffers() {
        VkCommandBufferAllocateInfo allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = computeCommandPool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = kFramesInFlight;
        checkVk(vkAllocateCommandBuffers(device_, &allocation,
                                        computeCommandBuffers_.data()),
                "vkAllocateCommandBuffers (compute)");
    }

    void createSwapchainResources() {
        createSwapchain();
        createDepthResources();
        createRenderPass();
        createGraphicsPipelines();
        createFramebuffers();
        allocateFrameCommandBuffers();
    }

    // The secondary records the indexed draw and inherited render context; its
    // primary records render scope, barriers, and executes the secondary.
    void recordComputeCommandBuffer(uint32_t frame) {
        VkCommandBuffer commandBuffer = computeCommandBuffers_[frame];
        checkVk(vkResetCommandBuffer(commandBuffer, 0),
                "vkResetCommandBuffer (compute)");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(commandBuffer, &begin),
                "vkBeginCommandBuffer (compute)");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computePipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout_, 0, 1,
                                &computeDescriptorSets_[frame], 0, nullptr);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        checkVk(vkEndCommandBuffer(commandBuffer),
                "vkEndCommandBuffer (compute)");
    }

    void recordSecondaryCommandBuffer(uint32_t imageIndex, uint32_t frame) {
        VkCommandBuffer secondary = secondaryCommandBuffers_[imageIndex];
        checkVk(vkResetCommandBuffer(secondary, 0),
                "vkResetCommandBuffer (secondary)");
        VkCommandBufferInheritanceRenderingInfo renderingInheritance{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO};
        renderingInheritance.colorAttachmentCount = 1;
        renderingInheritance.pColorAttachmentFormats = &swapchainImageFormat_;
        renderingInheritance.depthAttachmentFormat = depthFormat_;
        renderingInheritance.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkCommandBufferInheritanceInfo inheritance{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        if (!useDynamicRendering_) {
            inheritance.renderPass = renderPass_;
            inheritance.subpass = 0;
            inheritance.framebuffer = framebuffers_[imageIndex];
        } else {
            inheritance.pNext = &renderingInheritance;
        }
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
                      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        begin.pInheritanceInfo = &inheritance;
        checkVk(vkBeginCommandBuffer(secondary, &begin),
                "vkBeginCommandBuffer (secondary)");
        vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          useDynamicRendering_ ? dynamicRenderingPipeline_
                                               : graphicsPipeline_);
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchainExtent_.width);
        viewport.height = static_cast<float>(swapchainExtent_.height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(secondary, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = swapchainExtent_;
        vkCmdSetScissor(secondary, 0, 1, &scissor);
        const VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(secondary, 0, 1, &vertexBuffer_.handle, &vertexOffset);
        vkCmdBindIndexBuffer(secondary, indexBuffer_.handle, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                graphicsPipelineLayout_, 0, 1,
                                &graphicsDescriptorSets_[frame], 0, nullptr);
        const glm::vec4 tint(1.0f);
        vkCmdPushConstants(secondary, graphicsPipelineLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(tint), &tint);
        vkCmdDrawIndexed(secondary, 18, 1, 0, 0, 0);
        checkVk(vkEndCommandBuffer(secondary), "vkEndCommandBuffer (secondary)");
    }

    // Record barriers and the selected render path in a primary buffer; queue
    // submission executes it after compute and acquire semaphore dependencies.
    void recordPrimaryCommandBuffer(uint32_t imageIndex, uint32_t frame) {
        VkCommandBuffer primary = primaryCommandBuffers_[imageIndex];
        checkVk(vkResetCommandBuffer(primary, 0),
                "vkResetCommandBuffer (primary)");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        checkVk(vkBeginCommandBuffer(primary, &begin),
                "vkBeginCommandBuffer (primary)");
        VkBufferMemoryBarrier2 computeToVertex{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        computeToVertex.srcStageMask =
            queueFamilies_.compute == queueFamilies_.graphics
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_NONE;
        computeToVertex.srcAccessMask =
            queueFamilies_.compute == queueFamilies_.graphics
                ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                : VK_ACCESS_2_NONE;
        computeToVertex.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        computeToVertex.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        computeToVertex.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        computeToVertex.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        computeToVertex.buffer = computeOutputBuffers_[frame].handle;
        computeToVertex.size = VK_WHOLE_SIZE;
        VkDependencyInfo computeDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        computeDependency.bufferMemoryBarrierCount = 1;
        computeDependency.pBufferMemoryBarriers = &computeToVertex;
        vkCmdPipelineBarrier2(primary, &computeDependency);

        if (useDynamicRendering_) {
            VkImageMemoryBarrier2 colorBarrier{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            colorBarrier.dstStageMask =
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            colorBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            colorBarrier.image = swapchainImages_[imageIndex];
            colorBarrier.subresourceRange =
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageMemoryBarrier2 depthBarrier{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            depthBarrier.dstStageMask =
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            depthBarrier.dstAccessMask =
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depthBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthBarrier.newLayout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.image = depthImages_[imageIndex].image;
            depthBarrier.subresourceRange =
                {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            const std::array<VkImageMemoryBarrier2, 2> barriers = {
                colorBarrier, depthBarrier};
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount =
                static_cast<uint32_t>(barriers.size());
            dependency.pImageMemoryBarriers = barriers.data();
            vkCmdPipelineBarrier2(primary, &dependency);
            VkClearValue colorClear{};
            colorClear.color = {{0.035f, 0.045f, 0.075f, 1.0f}};
            VkClearValue depthClear{};
            depthClear.depthStencil = {1.0f, 0};
            VkRenderingAttachmentInfo colorAttachment{
                VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colorAttachment.imageView = swapchainImageViews_[imageIndex];
            colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.clearValue = colorClear;
            VkRenderingAttachmentInfo depthAttachment{
                VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            depthAttachment.imageView = depthImageViews_[imageIndex];
            depthAttachment.imageLayout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.clearValue = depthClear;
            VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            rendering.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
            rendering.renderArea.extent = swapchainExtent_;
            rendering.layerCount = 1;
            rendering.colorAttachmentCount = 1;
            rendering.pColorAttachments = &colorAttachment;
            rendering.pDepthAttachment = &depthAttachment;
            vkCmdBeginRendering(primary, &rendering);
        } else {
            VkRenderPassBeginInfo renderPassBegin{
                VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            renderPassBegin.renderPass = renderPass_;
            renderPassBegin.framebuffer = framebuffers_[imageIndex];
            renderPassBegin.renderArea.extent = swapchainExtent_;
            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = {{0.035f, 0.045f, 0.075f, 1.0f}};
            clearValues[1].depthStencil = {1.0f, 0};
            renderPassBegin.clearValueCount =
                static_cast<uint32_t>(clearValues.size());
            renderPassBegin.pClearValues = clearValues.data();
            vkCmdBeginRenderPass(primary, &renderPassBegin,
                                 VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        }
        const VkCommandBuffer secondary = secondaryCommandBuffers_[imageIndex];
        vkCmdExecuteCommands(primary, 1, &secondary);
        if (useDynamicRendering_) {
            vkCmdEndRendering(primary);
            VkImageMemoryBarrier2 presentBarrier{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            presentBarrier.srcStageMask =
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            presentBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            presentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            presentBarrier.image = swapchainImages_[imageIndex];
            presentBarrier.subresourceRange =
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &presentBarrier;
            vkCmdPipelineBarrier2(primary, &dependency);
        } else {
            vkCmdEndRenderPass(primary);
        }
        checkVk(vkEndCommandBuffer(primary), "vkEndCommandBuffer (primary)");
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            checkVk(vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                                      &imageAvailableSemaphores_[i]),
                    "vkCreateSemaphore (image available)");
            checkVk(vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]),
                    "vkCreateFence (frame)");
        }
        VkSemaphoreTypeCreateInfo typeInfo{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo timelineInfo{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        timelineInfo.pNext = &typeInfo;
        checkVk(vkCreateSemaphore(device_, &timelineInfo, nullptr, &computeTimeline_),
                "vkCreateSemaphore (compute timeline)");
    }

    // A binary acquire semaphore gates rendering; a timeline value links the
    // per-frame compute dispatch to the vertex shader; image-specific binaries gate present.
    void drawFrame() {
        checkVk(vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_],
                                VK_TRUE, UINT64_MAX),
                "vkWaitForFences (frame)");
        uint32_t imageIndex = 0;
        const VkResult acquireResult = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            framebufferResized_ = false;
            recreateSwapchain();
            return;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            checkVk(acquireResult, "vkAcquireNextImageKHR");
        }
        if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
            checkVk(vkWaitForFences(device_, 1, &imagesInFlight_[imageIndex],
                                    VK_TRUE, UINT64_MAX),
                    "vkWaitForFences (swapchain image)");
        }
        const CameraUniform camera{viewProjection()};
        std::memcpy(cameraBuffers_[currentFrame_].mapped, &camera, sizeof(camera));
        flushBuffer(cameraBuffers_[currentFrame_]);
        const uint32_t frame = static_cast<uint32_t>(currentFrame_);
        recordComputeCommandBuffer(frame);
        recordSecondaryCommandBuffer(imageIndex, frame);
        recordPrimaryCommandBuffer(imageIndex, frame);

        ++computeTimelineValue_;
        VkCommandBufferSubmitInfo computeCommand{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        computeCommand.commandBuffer = computeCommandBuffers_[currentFrame_];
        VkSemaphoreSubmitInfo computeSignal{
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        computeSignal.semaphore = computeTimeline_;
        computeSignal.value = computeTimelineValue_;
        computeSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 computeSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        computeSubmit.commandBufferInfoCount = 1;
        computeSubmit.pCommandBufferInfos = &computeCommand;
        computeSubmit.signalSemaphoreInfoCount = 1;
        computeSubmit.pSignalSemaphoreInfos = &computeSignal;
        checkVk(vkQueueSubmit2(computeQueue_, 1, &computeSubmit, VK_NULL_HANDLE),
                "vkQueueSubmit2 (compute)");

        checkVk(vkResetFences(device_, 1, &inFlightFences_[currentFrame_]),
                "vkResetFences (frame)");
        imagesInFlight_[imageIndex] = inFlightFences_[currentFrame_];
        VkCommandBufferSubmitInfo graphicsCommand{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        graphicsCommand.commandBuffer = primaryCommandBuffers_[imageIndex];
        std::array<VkSemaphoreSubmitInfo, 2> waits{};
        waits[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waits[0].semaphore = imageAvailableSemaphores_[currentFrame_];
        waits[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        waits[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waits[1].semaphore = computeTimeline_;
        waits[1].value = computeTimelineValue_;
        waits[1].stageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        VkSemaphoreSubmitInfo finishedSignal{
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        finishedSignal.semaphore = presentReadySemaphores_[imageIndex];
        finishedSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 graphicsSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        graphicsSubmit.waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size());
        graphicsSubmit.pWaitSemaphoreInfos = waits.data();
        graphicsSubmit.commandBufferInfoCount = 1;
        graphicsSubmit.pCommandBufferInfos = &graphicsCommand;
        graphicsSubmit.signalSemaphoreInfoCount = 1;
        graphicsSubmit.pSignalSemaphoreInfos = &finishedSignal;
        checkVk(vkQueueSubmit2(graphicsQueue_, 1, &graphicsSubmit,
                               inFlightFences_[currentFrame_]),
                "vkQueueSubmit2 (graphics)");

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &presentReadySemaphores_[imageIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_;
        presentInfo.pImageIndices = &imageIndex;
        const VkResult presentResult = vkQueuePresentKHR(presentQueue_, &presentInfo);
        if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR &&
            presentResult != VK_ERROR_OUT_OF_DATE_KHR) {
            checkVk(presentResult, "vkQueuePresentKHR");
        }
        if (acquireResult == VK_SUBOPTIMAL_KHR ||
            presentResult == VK_SUBOPTIMAL_KHR ||
            presentResult == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized_) {
            framebufferResized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % kFramesInFlight;
    }

    void freeSwapchainCommandBuffers() {
        if (!primaryCommandBuffers_.empty()) {
            vkFreeCommandBuffers(device_, graphicsCommandPool_,
                                 static_cast<uint32_t>(primaryCommandBuffers_.size()),
                                 primaryCommandBuffers_.data());
            primaryCommandBuffers_.clear();
        }
        if (!secondaryCommandBuffers_.empty()) {
            vkFreeCommandBuffers(device_, graphicsCommandPool_,
                                 static_cast<uint32_t>(secondaryCommandBuffers_.size()),
                                 secondaryCommandBuffers_.data());
            secondaryCommandBuffers_.clear();
        }
        imagesInFlight_.clear();
    }

    void cleanupSwapchainResources() {
        if (device_ == VK_NULL_HANDLE) {
            return;
        }
        freeSwapchainCommandBuffers();
        for (VkFramebuffer framebuffer : framebuffers_) {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
        framebuffers_.clear();
        if (graphicsPipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
            graphicsPipeline_ = VK_NULL_HANDLE;
        }
        if (dynamicRenderingPipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, dynamicRenderingPipeline_, nullptr);
            dynamicRenderingPipeline_ = VK_NULL_HANDLE;
        }
        for (VkImageView view : depthImageViews_) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, view, nullptr);
            }
        }
        depthImageViews_.clear();
        for (ImageAllocation& allocation : depthImages_) {
            if (allocation.image != VK_NULL_HANDLE) {
                vkDestroyImage(device_, allocation.image, nullptr);
            }
            if (allocation.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, allocation.memory, nullptr);
            }
        }
        depthImages_.clear();
        if (renderPass_ != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device_, renderPass_, nullptr);
            renderPass_ = VK_NULL_HANDLE;
        }
        for (VkImageView view : swapchainImageViews_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        swapchainImageViews_.clear();
        for (VkSemaphore semaphore : presentReadySemaphores_) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore, nullptr);
            }
        }
        presentReadySemaphores_.clear();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
        swapchainImages_.clear();
    }

    void recreateSwapchain() {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            if (glfwWindowShouldClose(window_) == GLFW_TRUE) {
                return;
            }
            glfwWaitEvents();
            glfwGetFramebufferSize(window_, &width, &height);
        }
        checkVk(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle (swapchain recreation)");
        cleanupSwapchainResources();
        createSwapchainResources();
    }

    void mainLoop() {
        while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
            glfwPollEvents();
            drawFrame();
        }
    }

    void initVulkan() {
        createInstance();
        checkVk(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
                "glfwCreateWindowSurface");
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPools();
        createPipelineCache();
        createDescriptorLayouts();
        createGeometryBuffers();
        createTexture();
        createFrameBuffers();
        createDescriptorSets();
        createComputePipeline();
        createComputeCommandBuffers();
        createSyncObjects();
        createSwapchainResources();
    }

    void cleanupVulkan() {
        if (device_ != VK_NULL_HANDLE) {
            const VkResult idleResult = vkDeviceWaitIdle(device_);
            if (idleResult != VK_SUCCESS) {
                std::cerr << "vkDeviceWaitIdle during cleanup failed (VkResult "
                          << idleResult << ")\n";
            }
            cleanupSwapchainResources();
            for (VkFence fence : inFlightFences_) {
                if (fence != VK_NULL_HANDLE) {
                    vkDestroyFence(device_, fence, nullptr);
                }
            }
            for (VkSemaphore semaphore : imageAvailableSemaphores_) {
                if (semaphore != VK_NULL_HANDLE) {
                    vkDestroySemaphore(device_, semaphore, nullptr);
                }
            }
            if (computeTimeline_ != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, computeTimeline_, nullptr);
            }
            if (computePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, computePipeline_, nullptr);
            }
            if (graphicsDescriptorPool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device_, graphicsDescriptorPool_, nullptr);
            }
            if (computeDescriptorPool_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device_, computeDescriptorPool_, nullptr);
            }
            if (textureSampler_ != VK_NULL_HANDLE) {
                vkDestroySampler(device_, textureSampler_, nullptr);
            }
            if (textureView_ != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, textureView_, nullptr);
            }
            if (texture_.image != VK_NULL_HANDLE) {
                vkDestroyImage(device_, texture_.image, nullptr);
            }
            if (texture_.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, texture_.memory, nullptr);
            }
            for (Buffer& buffer : cameraBuffers_) {
                destroyBuffer(&buffer);
            }
            for (Buffer& buffer : computeOutputBuffers_) {
                destroyBuffer(&buffer);
            }
            destroyBuffer(&vertexBuffer_);
            destroyBuffer(&indexBuffer_);
            if (graphicsPipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, graphicsPipelineLayout_, nullptr);
            }
            if (computePipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, computePipelineLayout_, nullptr);
            }
            if (graphicsSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, graphicsSetLayout_, nullptr);
            }
            if (computeSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, computeSetLayout_, nullptr);
            }
            if (pipelineCache_ != VK_NULL_HANDLE) {
                vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
            }
            if (computeCommandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, computeCommandPool_, nullptr);
            }
            if (transferCommandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, transferCommandPool_, nullptr);
            }
            if (graphicsCommandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, graphicsCommandPool_, nullptr);
            }
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (debugMessenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
            auto destroyMessenger =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance_,
                                          "vkDestroyDebugUtilsMessengerEXT"));
            if (destroyMessenger != nullptr) {
                destroyMessenger(instance_, debugMessenger_, nullptr);
            }
            debugMessenger_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        bool useDynamicRendering = false;
        if (argc == 2 && std::string(argv[1]) == "--dynamic-rendering") {
            useDynamicRendering = true;
        } else if (argc > 1) {
            throw std::runtime_error(
                "Usage: vulkan_triangle [--dynamic-rendering]");
        }
        VulkanPyramid app;
        app.setDynamicRendering(useDynamicRendering);
        app.run();
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
