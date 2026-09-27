package main

import "base:runtime"
import "core:fmt"
import "core:image"
import "core:image/png"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "vendor:glfw"
import vk "vendor:vulkan"
import "vendor:wasm/WebGL"

WIDTH :: 800
HEIGHT :: 800
MAX_FRAMES_IN_FLIGHT :: 2
STAGING_BUFFER_MEMORY :: 2048 * 2048 * 4

vertices := [?]Vertex {
	Vertex{{-0.5, -0.5, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0}},
	Vertex{{0.5, -0.5, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0}},
	Vertex{{0.5, 0.5, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0}},
	Vertex{{-0.5, 0.5, 0.0}, {1.0, 1.0, 1.0}, {0.0, 1.0}},
	Vertex{{-0.5, -0.5, -0.5}, {1.0, 0.0, 0.0}, {0.0, 0.0}},
	Vertex{{0.5, -0.5, -0.5}, {0.0, 1.0, 0.0}, {1.0, 0.0}},
	Vertex{{0.5, 0.5, -0.5}, {0.0, 0.0, 1.0}, {1.0, 1.0}},
	Vertex{{-0.5, 0.5, -0.5}, {1.0, 1.0, 1.0}, {0.0, 1.0}},
}
indices := [?]u16{0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4}
vs_buffer_offsets := []vk.DeviceSize{0}

UniformBufferObject :: struct {
	model:      matrix[4, 4]f32,
	view:       matrix[4, 4]f32,
	projection: matrix[4, 4]f32,
}

// All Vulkan/GLFW handles and per-frame state live in one struct so they can be
// shared across the helper procs without long parameter lists.
vk_state := struct {
	surface:                    vk.SurfaceKHR,
	window:                     glfw.WindowHandle,
	instance:                   vk.Instance,
	queue_family_index:         u32,
	queue:                      vk.Queue,
	logical_device:             vk.Device,
	physical_device:            vk.PhysicalDevice,
	physical_device_properties: vk.PhysicalDeviceProperties,
	swapchain:                  vk.SwapchainKHR,
	swapchain_images:           []vk.Image,
	image_views:                []vk.ImageView,
	pipeline_layout:            vk.PipelineLayout,
	graphics_pipeline:          vk.Pipeline,
	command_pool:               vk.CommandPool,
	command_buffers:            [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer,
	present_complete_sems:      [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
	render_finished_sems:       []vk.Semaphore,
	in_flight_fences:           [MAX_FRAMES_IN_FLIGHT]vk.Fence,
	swapchain_extent:           vk.Extent2D,
	viewport:                   vk.Viewport,
	scissor:                    vk.Rect2D,
	surface_format:             vk.SurfaceFormatKHR,
	present_mode:               vk.PresentModeKHR,
	image_index:                u32,
	frame_index:                u32,
	vs_buffer:                  vk.Buffer,
	vs_buffer_memory:           vk.DeviceMemory,
	index_buffer:               vk.Buffer,
	index_buffer_memory:        vk.DeviceMemory,
	uniform_buffers:            [MAX_FRAMES_IN_FLIGHT]vk.Buffer,
	uniform_buffer_memories:    [MAX_FRAMES_IN_FLIGHT]vk.DeviceMemory,
	uniform_buffers_mapped:     [MAX_FRAMES_IN_FLIGHT]rawptr,
	ubo:                        UniformBufferObject,
	descriptor_pool:            vk.DescriptorPool,
	descriptor_sets:            [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
	texture_image:              vk.Image,
	texture_memory:             vk.DeviceMemory,
	texture_image_view:         vk.ImageView,
	texture_sampler:            vk.Sampler,
	depth_image:                vk.Image,
	depth_memory:               vk.DeviceMemory,
	depth_image_view:           vk.ImageView,
	depth_format:               vk.Format,
}{}

Vertex :: struct {
	pos:   [3]f32,
	color: [3]f32,
	uv:    [2]f32,
}

IDENTITY :: matrix[4, 4]f32{
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
}

matrix_transform :: proc(v: [3]f32) -> (transform: matrix[4, 4]f32) {
	transform[0] = [4]f32{1, 0, 0, 0}
	transform[1] = [4]f32{0, 1, 0, 0}
	transform[2] = [4]f32{0, 0, 1, 0}
	transform[3] = [4]f32{v[0], v[1], v[2], 1}
	return transform
}

matrix_scale :: proc(v: [3]f32) -> (scale: matrix[4, 4]f32) {
	scale[0] = [4]f32{v[0], 0, 0, 0}
	scale[1] = [4]f32{0, v[1], 0, 0}
	scale[2] = [4]f32{0, 0, v[2], 0}
	scale[3] = [4]f32{0, 0, 0, 1}
	return scale
}

matrix_rotate :: proc(angle_radians: f32, v: [3]f32) -> (rot: matrix[4, 4]f32) {
	c := math.cos(angle_radians)
	s := math.sin(angle_radians)

	a := linalg.normalize(v)
	t := a * (1 - c)

	rot[0, 0] = c + t[0] * a[0]
	rot[1, 0] = 0 + t[0] * a[1] + s * a[2]
	rot[2, 0] = 0 + t[0] * a[2] - s * a[1]
	rot[3, 0] = 0

	rot[0, 1] = 0 + t[1] * a[0] - s * a[2]
	rot[1, 1] = c + t[1] * a[1]
	rot[2, 1] = 0 + t[1] * a[2] + s * a[0]
	rot[3, 1] = 0

	rot[0, 2] = 0 + t[2] * a[0] + s * a[1]
	rot[1, 2] = 0 + t[2] * a[1] - s * a[0]
	rot[2, 2] = c + t[2] * a[2]
	rot[3, 2] = 0

	rot[0, 3] = 0
	rot[1, 3] = 0
	rot[2, 3] = 0
	rot[3, 3] = 1

	return rot
}


get_binding_description :: proc() -> vk.VertexInputBindingDescription {
	binding_description := vk.VertexInputBindingDescription {
		binding   = 0,
		stride    = size_of(Vertex),
		inputRate = .VERTEX,
	}
	return binding_description
}

get_attribute_descriptions :: proc() -> [3]vk.VertexInputAttributeDescription {
	return {
		vk.VertexInputAttributeDescription {
			location = 0,
			binding = 0,
			format = .R32G32B32_SFLOAT,
			offset = u32(offset_of(Vertex, pos)),
		},
		vk.VertexInputAttributeDescription {
			location = 1,
			binding = 0,
			format = .R32G32B32_SFLOAT,
			offset = u32(offset_of(Vertex, color)),
		},
		vk.VertexInputAttributeDescription {
			location = 2,
			binding = 0,
			format = .R32G32_SFLOAT,
			offset = u32(offset_of(Vertex, uv)),
		},
	}
}


clamp :: proc(value: $T, min: T, max: T) -> T {
	final_value := value > max ? max : value
	return final_value < min ? min : final_value
}


record_command_buffer :: proc() {
	command_buffer := vk_state.command_buffers[vk_state.frame_index]
	command_buffer_begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {},
	}
	result := vk.BeginCommandBuffer(command_buffer, &command_buffer_begin_info)
	assert(result == .SUCCESS)

	// Undefined -> ColorAttachmentOptimal, so we can safely render into it.

	barriers := [?]vk.ImageMemoryBarrier2 {
		vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			srcStageMask = {.TOP_OF_PIPE},
			dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
			srcAccessMask = {},
			dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
			oldLayout = .UNDEFINED,
			newLayout = .COLOR_ATTACHMENT_OPTIMAL,
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			image = vk_state.swapchain_images[vk_state.image_index],
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		},
		vk.ImageMemoryBarrier2 {
			sType = .IMAGE_MEMORY_BARRIER_2,
			dstStageMask = {.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS},
			dstAccessMask = {.DEPTH_STENCIL_ATTACHMENT_WRITE},
			srcStageMask = {.EARLY_FRAGMENT_TESTS, .LATE_FRAGMENT_TESTS},
			srcAccessMask = {.DEPTH_STENCIL_ATTACHMENT_WRITE},
			oldLayout = .UNDEFINED,
			newLayout = .DEPTH_ATTACHMENT_OPTIMAL,
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			image = vk_state.depth_image,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.DEPTH},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		},
	}

	dependency_info := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 2,
		pImageMemoryBarriers    = raw_data(barriers[:]),
	}
	vk.CmdPipelineBarrier2(vk_state.command_buffers[vk_state.frame_index], &dependency_info)

	clear_color: vk.ClearValue = {
		color = vk.ClearColorValue{float32 = [4]f32{0.0, 0.0, 0.0, 1.0}},
	}
	clear_depth: vk.ClearValue = {
		depthStencil = vk.ClearDepthStencilValue{depth = 1.0, stencil = 0},
	}

	color_attachment_info := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = vk_state.image_views[vk_state.image_index],
		imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR, // CLEAR so clearValue actually takes effect
		storeOp     = .STORE,
		clearValue  = clear_color,
	}
	depth_attachment_info := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = vk_state.depth_image_view,
		imageLayout = .DEPTH_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR,
		storeOp     = .STORE,
		clearValue  = clear_depth,
	}

	rendering_info := vk.RenderingInfo {
		sType = .RENDERING_INFO,
		renderArea = vk.Rect2D {
			offset = vk.Offset2D{x = 0, y = 0},
			extent = vk_state.swapchain_extent,
		},
		layerCount = 1,
		colorAttachmentCount = 1,
		pColorAttachments = &color_attachment_info,
		pDepthAttachment = &depth_attachment_info,
	}

	vk.CmdBeginRendering(command_buffer, &rendering_info)
	vk.CmdBindPipeline(command_buffer, .GRAPHICS, vk_state.graphics_pipeline)
	vk.CmdSetViewport(command_buffer, 0, 1, &vk_state.viewport)
	vk.CmdSetScissor(command_buffer, 0, 1, &vk_state.scissor)
	vk.CmdBindVertexBuffers(command_buffer, 0, 1, &vk_state.vs_buffer, raw_data(vs_buffer_offsets))
	vk.CmdBindIndexBuffer(command_buffer, vk_state.index_buffer, 0, .UINT16)
	vk.CmdBindDescriptorSets(
		command_buffer,
		.GRAPHICS,
		vk_state.pipeline_layout,
		0,
		1,
		&vk_state.descriptor_sets[vk_state.frame_index],
		0,
		nil,
	)
	vk.CmdDrawIndexed(command_buffer, len(indices), 1, 0, 0, 0)
	vk.CmdEndRendering(command_buffer)

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		dstStageMask = {.BOTTOM_OF_PIPE},
		srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
		dstAccessMask = {},
		oldLayout = .COLOR_ATTACHMENT_OPTIMAL,
		newLayout = .PRESENT_SRC_KHR,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = vk_state.swapchain_images[vk_state.image_index],
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}
	dependency_info = vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(vk_state.command_buffers[vk_state.frame_index], &dependency_info)


	result = vk.EndCommandBuffer(command_buffer)
	assert(result == .SUCCESS)
}

draw_frame :: proc() {
	result := vk.WaitForFences(
		vk_state.logical_device,
		1,
		&vk_state.in_flight_fences[vk_state.frame_index],
		true,
		max(u64),
	)
	assert(result == .SUCCESS)
	vk.ResetFences(vk_state.logical_device, 1, &vk_state.in_flight_fences[vk_state.frame_index])

	result = vk.AcquireNextImageKHR(
		vk_state.logical_device,
		vk_state.swapchain,
		max(u64),
		vk_state.present_complete_sems[vk_state.frame_index],
		{},
		&vk_state.image_index,
	)
	assert(result == .SUCCESS)

	// Record commands for this frame's image and submit them.
	record_command_buffer()

	time := f32(glfw.GetTime())
	new_ubo: UniformBufferObject
	new_ubo.model = linalg.matrix4_rotate(time * 0.1, [3]f32{0, 0, 1})
	new_ubo.view = linalg.matrix4_look_at([3]f32{2, 2, 2}, [3]f32{0, 0, 0}, [3]f32{0, 0, 1})
	new_ubo.projection = linalg.matrix4_perspective(
		f32(math.to_radians(45.0)),
		f32(vk_state.swapchain_extent.width) / f32(vk_state.swapchain_extent.height),
		0.1,
		10,
	)
	mem.copy(
		vk_state.uniform_buffers_mapped[vk_state.frame_index],
		&new_ubo,
		size_of(UniformBufferObject),
	)

	dst_stage_mask := vk.PipelineStageFlags{.COLOR_ATTACHMENT_OUTPUT}
	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		waitSemaphoreCount   = 1,
		pWaitSemaphores      = &vk_state.present_complete_sems[vk_state.frame_index],
		pWaitDstStageMask    = &dst_stage_mask,
		commandBufferCount   = 1,
		pCommandBuffers      = &vk_state.command_buffers[vk_state.frame_index],
		signalSemaphoreCount = 1,
		pSignalSemaphores    = &vk_state.render_finished_sems[vk_state.image_index],
	}
	vk.QueueSubmit(
		vk_state.queue,
		1,
		&submit_info,
		vk_state.in_flight_fences[vk_state.frame_index],
	)

	// Present the rendered image once rendering (render_finished_sem) is done.
	present_info_khr := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &vk_state.render_finished_sems[vk_state.image_index],
		swapchainCount     = 1,
		pSwapchains        = &vk_state.swapchain,
		pImageIndices      = &vk_state.image_index,
	}
	vk.QueuePresentKHR(vk_state.queue, &present_info_khr)

	vk_state.frame_index = (vk_state.frame_index + 1) % MAX_FRAMES_IN_FLIGHT
}

initialize_glfw :: proc() -> glfw.WindowHandle {
	if !glfw.Init() {
		panic("GLFW: Could not be initialized")
	}
	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, glfw.FALSE)
	glfw_window := glfw.CreateWindow(WIDTH, HEIGHT, "Vulkan: Hello Triangle", nil, nil)
	if glfw_window == nil {
		panic("GLFW: Window could not be created!")
	}

	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))
	return glfw_window
}

check_instance_extensions :: proc() -> ([]cstring, bool) {
	// Vulkan: Get supported extensions
	vulkan_extensions_count: u32
	vk.EnumerateInstanceExtensionProperties(nil, &vulkan_extensions_count, nil)
	extension_properties := make([]vk.ExtensionProperties, vulkan_extensions_count)
	defer delete(extension_properties)
	vk.EnumerateInstanceExtensionProperties(
		nil,
		&vulkan_extensions_count,
		raw_data(extension_properties),
	)

	// Vulkan: Check required window extensions against supported vulkan extensions
	glfw_extensions := glfw.GetRequiredInstanceExtensions()
	for extension in glfw_extensions {
		found := false
		for &property in extension_properties {
			property_name := cstring(&property.extensionName[0])
			if property_name == extension {
				found = true
				break
			}
		}
		if !found {
			return glfw_extensions, false
		}
	}
	return glfw_extensions, true
}

when ODIN_DEBUG {
	check_validation_layers :: proc(validation_layers: []cstring) -> bool {
		layer_count: u32
		result := vk.EnumerateInstanceLayerProperties(&layer_count, nil)
		assert(result == .SUCCESS)

		layer_properties := make([]vk.LayerProperties, layer_count)
		defer delete(layer_properties)
		vk.EnumerateInstanceLayerProperties(&layer_count, raw_data(layer_properties))

		for required_layer in validation_layers {
			found := false
			for &layer in layer_properties {
				layer_name := cstring(&layer.layerName[0])
				if required_layer == layer_name {
					found = true
					break
				}
			}
			if !found {
				return false
			}

		}
		return true
	}
}

check_device_extensions :: proc(
	required_device_extensions: []cstring,
	physical_device: vk.PhysicalDevice,
) -> bool {
	device_extension_count: u32
	vk.EnumerateDeviceExtensionProperties(
		vk_state.physical_device,
		nil,
		&device_extension_count,
		nil,
	)
	available_device_extensions := make([]vk.ExtensionProperties, device_extension_count)
	defer delete(available_device_extensions)
	vk.EnumerateDeviceExtensionProperties(
		physical_device,
		nil,
		&device_extension_count,
		raw_data(available_device_extensions),
	)
	for required_extension in required_device_extensions {
		found := false
		for &available_extension in available_device_extensions {
			extension_name := cstring(&available_extension.extensionName[0])
			if extension_name == required_extension {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

create_image :: proc(
	device: vk.Device,
	physical_device: vk.PhysicalDevice,
	staging_buffer_mapped: rawptr,
) -> (
	vk.Image,
	vk.DeviceMemory,
	u32,
	u32,
) {
	texture, err := png.load_from_file("texture.png")
	if err != nil {
		fmt.println("There has been an error:", err)
		panic("There has been an error")
	}
	defer image.destroy(texture)
	image.alpha_add_if_missing(texture)
	texture_size := texture.width * texture.height * 4
	mem.copy(staging_buffer_mapped, raw_data(texture.pixels.buf), texture_size)
	new_image: vk.Image
	texture_info := vk.ImageCreateInfo {
		sType = .IMAGE_CREATE_INFO,
		imageType = .D2,
		format = .R8G8B8A8_SRGB,
		extent = vk.Extent3D{width = u32(texture.width), height = u32(texture.height), depth = 1},
		mipLevels = 1,
		arrayLayers = 1,
		samples = {._1},
		tiling = .OPTIMAL,
		usage = {.TRANSFER_DST, .SAMPLED},
		sharingMode = .EXCLUSIVE,
	}
	vk.CreateImage(device, &texture_info, nil, &new_image)
	new_image_memory_req: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, new_image, &new_image_memory_req)
	alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = new_image_memory_req.size,
		memoryTypeIndex = choose_memory_type(
			physical_device,
			new_image_memory_req.memoryTypeBits,
			{.DEVICE_LOCAL},
		),
	}
	new_image_memory: vk.DeviceMemory
	vk.AllocateMemory(device, &alloc_info, nil, &new_image_memory)
	vk.BindImageMemory(device, new_image, new_image_memory, 0)
	return new_image, new_image_memory, u32(texture.width), u32(texture.height)
}

create_depth_image :: proc(
	device: vk.Device,
	physical_device: vk.PhysicalDevice,
) -> (
	vk.Image,
	vk.ImageView,
	vk.DeviceMemory,
	vk.Format,
) {
	supported_formats := [?]vk.Format{.D32_SFLOAT, .D32_SFLOAT_S8_UINT, .D24_UNORM_S8_UINT}
	requested_features: vk.FormatFeatureFlags = {.DEPTH_STENCIL_ATTACHMENT}
	depth_image: vk.Image
	depth_image_view: vk.ImageView
	depth_format: vk.Format
	depth_memory: vk.DeviceMemory
	for format in supported_formats {
		format_prop: vk.FormatProperties
		vk.GetPhysicalDeviceFormatProperties(physical_device, format, &format_prop)
		if requested_features <= format_prop.optimalTilingFeatures {
			depth_format = format
			break
		}
	}
	vk.CreateImage(
		device,
		&vk.ImageCreateInfo {
			sType = .IMAGE_CREATE_INFO,
			imageType = .D2,
			format = depth_format,
			extent = {vk_state.swapchain_extent.width, vk_state.swapchain_extent.height, 1},
			mipLevels = 1,
			arrayLayers = 1,
			samples = {._1},
			tiling = .OPTIMAL,
			usage = {.DEPTH_STENCIL_ATTACHMENT},
			sharingMode = .EXCLUSIVE,
		},
		nil,
		&depth_image,
	)

	depth_image_mem_req: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(vk_state.logical_device, depth_image, &depth_image_mem_req)
	vk.AllocateMemory(
		vk_state.logical_device,
		&vk.MemoryAllocateInfo {
			sType = .MEMORY_ALLOCATE_INFO,
			allocationSize = depth_image_mem_req.size,
			memoryTypeIndex = choose_memory_type(
				physical_device,
				depth_image_mem_req.memoryTypeBits,
				{.DEVICE_LOCAL},
			),
		},
		nil,
		&depth_memory,
	)
	vk.BindImageMemory(device, depth_image, depth_memory, vk.DeviceSize(0))

	vk.CreateImageView(
		device,
		&vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = depth_image,
			viewType = .D2,
			format = depth_format,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.DEPTH},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		},
		nil,
		&depth_image_view,
	)
	return depth_image, depth_image_view, depth_memory, depth_format
}

create_instance :: proc(
	instance_extensions: []cstring,
	validation_layers: []cstring,
) -> (
	vk.Instance,
	bool,
) {
	// Vulkan: Create instance
	instance: vk.Instance
	app_info := vk.ApplicationInfo {
		pApplicationName   = "Vulkan: Hello Triangle",
		pEngineName        = "No Engine",
		apiVersion         = vk.API_VERSION_1_4,
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		engineVersion      = vk.MAKE_VERSION(1, 0, 0),
		sType              = .APPLICATION_INFO,
	}
	instance_info := vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &app_info,
		enabledExtensionCount   = u32(len(instance_extensions)),
		ppEnabledExtensionNames = raw_data(instance_extensions),
		enabledLayerCount       = u32(len(validation_layers)),
		ppEnabledLayerNames     = raw_data(validation_layers),
	}
	result := vk.CreateInstance(&instance_info, nil, &instance)
	return instance, result == .SUCCESS
}

create_logical_device :: proc(
	queue_family_index: u32,
	physical_device: vk.PhysicalDevice,
	required_device_extensions: []cstring,
) -> (
	vk.Device,
	bool,
) {
	queue_priority: f32 = 1.0
	vk_queue := vk.DeviceQueueCreateInfo {
		sType            = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = vk_state.queue_family_index,
		queueCount       = 1,
		pQueuePriorities = &queue_priority,
	}

	// Vulkan: Physical device features chain (pNext-linked).
	// Order: features2 -> features11 -> features13 -> features_ext.
	// synchronization2 is required by vk.CmdPipelineBarrier2 used below.
	features_ext := vk.PhysicalDeviceExtendedDynamicStateFeaturesEXT {
		sType                = .PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
		extendedDynamicState = true,
		pNext                = nil,
	}

	features13 := vk.PhysicalDeviceVulkan13Features {
		sType            = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		dynamicRendering = true,
		synchronization2 = true,
		pNext            = &features_ext,
	}

	features11 := vk.PhysicalDeviceVulkan11Features {
		sType                = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		shaderDrawParameters = true,
		pNext                = &features13,
	}

	features2 := vk.PhysicalDeviceFeatures2 {
		sType = .PHYSICAL_DEVICE_FEATURES_2,
		pNext = &features11,
		features = vk.PhysicalDeviceFeatures{samplerAnisotropy = true},
	}

	// Vulkan: Create logical device
	logical_device_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		pNext                   = &features2,
		queueCreateInfoCount    = 1,
		pQueueCreateInfos       = &vk_queue,
		enabledExtensionCount   = u32(len(required_device_extensions)),
		ppEnabledExtensionNames = raw_data(required_device_extensions[:]),
	}

	logical_device: vk.Device
	result := vk.CreateDevice(physical_device, &logical_device_info, nil, &logical_device)
	return logical_device, result == .SUCCESS
}

create_swapchain_with_images :: proc(
	physical_device: vk.PhysicalDevice,
	logical_device: vk.Device,
	surface: vk.SurfaceKHR,
	window: glfw.WindowHandle,
) -> (
	vk.Extent2D,
	vk.SwapchainKHR,
	[]vk.Image,
	[]vk.ImageView,
) {
	surface_capability: vk.SurfaceCapabilitiesKHR
	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_capability)

	chosen_extent: vk.Extent2D
	if surface_capability.currentExtent.width != max(u32) {
		chosen_extent = surface_capability.currentExtent
	} else {
		width, height := glfw.GetFramebufferSize(window)
		chosen_extent.width = clamp(
			u32(width),
			surface_capability.minImageExtent.width,
			surface_capability.maxImageExtent.width,
		)
		chosen_extent.height = clamp(
			u32(height),
			surface_capability.minImageExtent.height,
			surface_capability.maxImageExtent.height,
		)
	}

	// Vulkan: Number of Images in Swap Chain
	min_image_count := surface_capability.minImageCount + 1
	if surface_capability.maxImageCount > 0 && min_image_count > surface_capability.maxImageCount {
		min_image_count = surface_capability.maxImageCount
	}

	// Vulkan: Creating Swap Chain
	swapchain_create_info := vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = surface,
		minImageCount    = min_image_count,
		imageFormat      = vk_state.surface_format.format,
		imageColorSpace  = vk_state.surface_format.colorSpace,
		imageExtent      = chosen_extent,
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT},
		imageSharingMode = .EXCLUSIVE,
		preTransform     = surface_capability.currentTransform,
		compositeAlpha   = {.OPAQUE},
		presentMode      = vk_state.present_mode,
		clipped          = true,
	}
	swapchain: vk.SwapchainKHR
	vk.CreateSwapchainKHR(logical_device, &swapchain_create_info, nil, &swapchain)
	swapchain_image_count: u32
	vk.GetSwapchainImagesKHR(logical_device, swapchain, &swapchain_image_count, nil)

	swapchain_images := make([]vk.Image, swapchain_image_count)
	vk.GetSwapchainImagesKHR(
		logical_device,
		swapchain,
		&swapchain_image_count,
		raw_data(swapchain_images),
	)

	// Vulkan: Create Image Views (one per swapchain image)
	image_views := make([]vk.ImageView, len(swapchain_images))
	image_view_create_info := vk.ImageViewCreateInfo {
		sType = .IMAGE_VIEW_CREATE_INFO,
		components = vk.ComponentMapping {
			r = .IDENTITY,
			g = .IDENTITY,
			b = .IDENTITY,
			a = .IDENTITY,
		},
		viewType = .D2,
		format = vk_state.surface_format.format,
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	for image, index in swapchain_images {
		image_view: vk.ImageView
		image_view_create_info.image = image
		vk.CreateImageView(logical_device, &image_view_create_info, nil, &image_view)
		image_views[index] = image_view
	}

	return chosen_extent, swapchain, swapchain_images, image_views

}

create_buffer :: proc(
	physical_device: vk.PhysicalDevice,
	logical_device: vk.Device,
	buffer_size: vk.DeviceSize,
	usage_flags: vk.BufferUsageFlags,
	properties_flags: vk.MemoryPropertyFlags,
) -> (
	vk.Buffer,
	vk.DeviceMemory,
) {
	buffer_create_info := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = buffer_size,
		usage       = usage_flags,
		sharingMode = .EXCLUSIVE,
	}
	buffer: vk.Buffer
	memory: vk.DeviceMemory
	if result := vk.CreateBuffer(logical_device, &buffer_create_info, nil, &buffer);
	   result != .SUCCESS {
		panic("Vertex buffer couldn't be created")
	}

	vs_buffer_mem_requirement: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(logical_device, buffer, &vs_buffer_mem_requirement)
	vs_buffer_mem_alloc_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = vs_buffer_mem_requirement.size,
		memoryTypeIndex = choose_memory_type(
			physical_device,
			vs_buffer_mem_requirement.memoryTypeBits,
			properties_flags,
		),
	}
	if result := vk.AllocateMemory(logical_device, &vs_buffer_mem_alloc_info, nil, &memory);
	   result != .SUCCESS {
		panic("couldn't allocate memory for the buffer")
	}
	vk.BindBufferMemory(logical_device, buffer, memory, 0)
	return buffer, memory
}


choose_physical_device :: proc(
	instance: vk.Instance,
) -> (
	vk.PhysicalDevice,
	vk.PhysicalDeviceProperties,
) {
	physical_devices_count: u32
	result := vk.EnumeratePhysicalDevices(instance, &physical_devices_count, nil)
	assert(result == .SUCCESS)
	physical_devices := make([]vk.PhysicalDevice, physical_devices_count)
	defer delete(physical_devices)
	vk.EnumeratePhysicalDevices(instance, &physical_devices_count, raw_data(physical_devices))


	physical_device_properties := make([]vk.PhysicalDeviceProperties, physical_devices_count)
	defer delete(physical_device_properties)
	for i in 0 ..< physical_devices_count {
		vk.GetPhysicalDeviceProperties(
			physical_devices[i],
			raw_data(physical_device_properties[i:]),
		)
	}
	chosen_device_index := 0
	for &property in physical_device_properties {
		if property.deviceType == .DISCRETE_GPU {
			break
		}
		chosen_device_index += 1
	}

	return physical_devices[chosen_device_index], physical_device_properties[chosen_device_index]
}

choose_queue_family :: proc(
	physical_device: vk.PhysicalDevice,
	surface: vk.SurfaceKHR,
) -> (
	u32,
	bool,
) {
	queue_family_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(vk_state.physical_device, &queue_family_count, nil)

	queue_families := make([]vk.QueueFamilyProperties, queue_family_count)
	defer delete(queue_families)
	vk.GetPhysicalDeviceQueueFamilyProperties(
		physical_device,
		&queue_family_count,
		raw_data(queue_families),
	)

	supports_graphics := false
	queue_family_index: u32
	for property, index in queue_families {
		surface_supported: b32 = false
		vk.GetPhysicalDeviceSurfaceSupportKHR(
			physical_device,
			queue_family_index,
			surface,
			&surface_supported,
		)
		if .GRAPHICS in property.queueFlags && surface_supported {
			supports_graphics = true
			return u32(index), true
		}
	}

	return 0, false
}

choose_surface_format :: proc(
	physical_device: vk.PhysicalDevice,
	surface: vk.SurfaceKHR,
) -> vk.SurfaceFormatKHR {
	surface_format_count: u32
	vk.GetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, nil)

	surface_formats := make([]vk.SurfaceFormatKHR, surface_format_count)
	defer delete(surface_formats)
	vk.GetPhysicalDeviceSurfaceFormatsKHR(
		physical_device,
		surface,
		&surface_format_count,
		raw_data(surface_formats),
	)

	surface_present_mode_count: u32
	vk.GetPhysicalDeviceSurfacePresentModesKHR(
		physical_device,
		surface,
		&surface_present_mode_count,
		nil,
	)

	chosen_surface_format: vk.SurfaceFormatKHR = surface_formats[0]
	for format in surface_formats {
		if format.format == .B8G8R8A8_SRGB && format.colorSpace == .SRGB_NONLINEAR {
			chosen_surface_format = format
			break
		}
	}

	return chosen_surface_format
}

choose_present_mode :: proc(
	physical_device: vk.PhysicalDevice,
	surface: vk.SurfaceKHR,
) -> vk.PresentModeKHR {
	surface_present_mode_count: u32
	vk.GetPhysicalDeviceSurfacePresentModesKHR(
		physical_device,
		surface,
		&surface_present_mode_count,
		nil,
	)

	surface_present_modes := make([]vk.PresentModeKHR, surface_present_mode_count)
	defer delete(surface_present_modes)

	vk.GetPhysicalDeviceSurfacePresentModesKHR(
		physical_device,
		surface,
		&surface_present_mode_count,
		raw_data(surface_present_modes),
	)

	chosen_present_mode: vk.PresentModeKHR = .FIFO
	for mode in surface_present_modes {
		if mode == .MAILBOX {
			chosen_present_mode = .MAILBOX
		}
	}
	return chosen_present_mode
}

choose_memory_type :: proc(
	physical_device: vk.PhysicalDevice,
	type_filter: u32,
	properties: vk.MemoryPropertyFlags,
) -> u32 {
	memory_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physical_device, &memory_properties)

	for i in 0 ..< memory_properties.memoryTypeCount {
		if (type_filter & (i << 1) != 0) &&
		   (properties <= memory_properties.memoryTypes[i].propertyFlags) {
			return u32(i)
		}
	}
	return 0
}


main :: proc() {
	result: vk.Result

	vk_state.window = initialize_glfw()
	defer
	{
		glfw.Terminate()
		glfw.DestroyWindow(vk_state.window)
	}

	required_extensions, ok := check_instance_extensions()
	if !ok {
		panic("Oh No, a crucial instance extension is missing. Go buy a new computer")
	}


	validation_layers := [?]cstring{"VK_LAYER_KHRONOS_validation"}
	when ODIN_DEBUG {
		if !check_validation_layers(validation_layers[:]) {
			panic("Oh No, you are missing a validation layer. Go buy a new computer")
		}
	}

	instance: vk.Instance
	instance, ok = create_instance(required_extensions, validation_layers[:])
	if !ok {
		panic("vulkan instance couldn't be created, I don't know why.")
	}
	defer vk.DestroyInstance(vk_state.instance, nil)
	vk_state.instance = instance
	vk.load_proc_addresses(vk_state.instance) // load instance/device function pointers

	vk_state.physical_device, vk_state.physical_device_properties = choose_physical_device(
		vk_state.instance,
	)

	result = glfw.CreateWindowSurface(vk_state.instance, vk_state.window, nil, &vk_state.surface)
	defer vk.DestroySurfaceKHR(vk_state.instance, vk_state.surface, nil)
	assert(result == .SUCCESS)

	if queue_family_index, ok := choose_queue_family(vk_state.physical_device, vk_state.surface);
	   !ok {
		panic("No queue family supporting graphics operations")
	}


	required_device_extensions := [?]cstring{vk.KHR_SWAPCHAIN_EXTENSION_NAME}
	if ok := check_device_extensions(required_device_extensions[:], vk_state.physical_device);
	   !ok {
		panic("Required device extensions are not supported")
	}

	device: vk.Device
	if device, ok = create_logical_device(
		vk_state.queue_family_index,
		vk_state.physical_device,
		required_device_extensions[:],
	); !ok {
		panic("Logical device couldn't be created")
	}
	defer vk.DestroyDevice(vk_state.logical_device, nil)
	vk_state.logical_device = device

	// Vulkan: Get queue handle
	vk.GetDeviceQueue(vk_state.logical_device, vk_state.queue_family_index, 0, &vk_state.queue)


	vk_state.surface_format = choose_surface_format(vk_state.physical_device, vk_state.surface)
	chosen_present_mode := choose_present_mode(vk_state.physical_device, vk_state.surface)
	vk_state.swapchain_extent, vk_state.swapchain, vk_state.swapchain_images, vk_state.image_views =
		create_swapchain_with_images(
			vk_state.physical_device,
			vk_state.logical_device,
			vk_state.surface,
			vk_state.window,
		)

	defer vk.DestroySwapchainKHR(vk_state.logical_device, vk_state.swapchain, nil)
	defer delete(vk_state.swapchain_images)
	defer {
		for image_view in vk_state.image_views {
			vk.DestroyImageView(vk_state.logical_device, image_view, nil)
		}
	}


	// Vulkan: Sync objects (fence starts signaled so the first draw_frame doesn't block)
	create_sem_info := vk.SemaphoreCreateInfo {
			sType = .SEMAPHORE_CREATE_INFO,
		}
	draw_fence_create_info := vk.FenceCreateInfo {
			sType = .FENCE_CREATE_INFO,
			flags = {.SIGNALED},
		}

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		vk.CreateSemaphore(
			vk_state.logical_device,
			&create_sem_info,
			nil,
			&vk_state.present_complete_sems[i],
		)
		vk.CreateFence(
			vk_state.logical_device,
			&draw_fence_create_info,
			nil,
			&vk_state.in_flight_fences[i],
		)
	}

	vk_state.render_finished_sems = make([]vk.Semaphore, len(vk_state.swapchain_images))
	defer delete(vk_state.render_finished_sems)
	for i in 0 ..< len(vk_state.swapchain_images) {
		vk.CreateSemaphore(
			vk_state.logical_device,
			&create_sem_info,
			nil,
			&vk_state.render_finished_sems[i],
		)
	}

	defer {

		for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
			vk.DestroySemaphore(vk_state.logical_device, vk_state.present_complete_sems[i], nil)
			vk.DestroyFence(vk_state.logical_device, vk_state.in_flight_fences[i], nil)
		}

		for i in 0 ..< len(vk_state.swapchain_images) {
			vk.DestroySemaphore(vk_state.logical_device, vk_state.render_finished_sems[i], nil)
		}
	}


	// Vulkan: Loading Shader
	data, err := os.read_entire_file("slang.spv", context.allocator)
	if err != nil {
		panic("Shader file could not be loaded!")
	}

	shader_create_info := vk.ShaderModuleCreateInfo {
			sType    = .SHADER_MODULE_CREATE_INFO,
			codeSize = len(data),
			pCode    = cast(^u32)raw_data(data),
		}
	shader_module: vk.ShaderModule
	result = vk.CreateShaderModule(
		vk_state.logical_device,
		&shader_create_info,
		nil,
		&shader_module,
	)
	defer vk.DestroyShaderModule(vk_state.logical_device, shader_module, nil)
	assert(result == .SUCCESS)

	// Vulkan: Create Shader Stages
	vertex_shader_stage_create_info := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.VERTEX},
		module = shader_module,
		pName  = cstring("vertMain"),
	}
	fragment_shader_stage_create_info := vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.FRAGMENT},
		module = shader_module,
		pName  = cstring("fragMain"),
	}
	pipeline_shader_stage_create_info := [?]vk.PipelineShaderStageCreateInfo {
		vertex_shader_stage_create_info,
		fragment_shader_stage_create_info,
	}

	dynamic_states := [?]vk.DynamicState{.VIEWPORT, .SCISSOR}
	dynamic_state_create_info := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = len(dynamic_states),
		pDynamicStates    = raw_data(dynamic_states[:]),
	}
	input_assembly_create_info := vk.PipelineInputAssemblyStateCreateInfo {
		sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology = .TRIANGLE_LIST,
	}
	viewport := vk.Viewport {
		x        = 0,
		y        = 0,
		width    = f32(vk_state.swapchain_extent.width),
		height   = f32(vk_state.swapchain_extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}
	scissor := vk.Rect2D {
		offset = vk.Offset2D{0, 0},
		extent = vk_state.swapchain_extent,
	}
	vk_state.viewport = viewport
	vk_state.scissor = scissor
	viewport_create_info := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
		pViewports    = &viewport,
		pScissors     = &scissor,
	}
	rasterization_create_info := vk.PipelineRasterizationStateCreateInfo {
		sType                   = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable        = false,
		rasterizerDiscardEnable = false,
		polygonMode             = .FILL,
		cullMode                = {.BACK},
		frontFace               = .CLOCKWISE,
		depthBiasEnable         = false,
		lineWidth               = 1.0,
	}
	multisample_create_info := vk.PipelineMultisampleStateCreateInfo {
		sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples = {._1},
		sampleShadingEnable  = false,
	}
	// No blending for a simple opaque triangle; just write all color channels.
	color_blend_attachment_state := vk.PipelineColorBlendAttachmentState {
		blendEnable    = false,
		colorWriteMask = {.R, .G, .B, .A},
	}
	color_blend_create_info := vk.PipelineColorBlendStateCreateInfo {
		sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable   = false,
		logicOp         = .COPY,
		attachmentCount = 1,
		pAttachments    = &color_blend_attachment_state,
	}

	bindings := [?]vk.DescriptorSetLayoutBinding {
		vk.DescriptorSetLayoutBinding {
			binding = 0,
			descriptorType = .UNIFORM_BUFFER,
			descriptorCount = 1,
			stageFlags = {.VERTEX},
		},
		vk.DescriptorSetLayoutBinding {
			binding = 1,
			descriptorType = .COMBINED_IMAGE_SAMPLER,
			descriptorCount = 1,
			stageFlags = {.FRAGMENT},
		},
	}


	index_buffer_size: vk.DeviceSize = len(indices) * size_of(u16)
	vk_state.index_buffer, vk_state.index_buffer_memory = create_buffer(
		vk_state.physical_device,
		vk_state.logical_device,
		index_buffer_size,
		{.TRANSFER_DST, .INDEX_BUFFER},
		{.DEVICE_LOCAL},
	)
	defer vk.FreeMemory(vk_state.logical_device, vk_state.index_buffer_memory, nil)
	defer vk.DestroyBuffer(vk_state.logical_device, vk_state.index_buffer, nil)

	vs_buffer_size: vk.DeviceSize = len(vertices) * size_of(Vertex)
	vs_staging_buffer, vs_staging_buffer_memory := create_buffer(
		vk_state.physical_device,
		vk_state.logical_device,
		STAGING_BUFFER_MEMORY,
		{.TRANSFER_SRC},
		{.HOST_COHERENT, .HOST_VISIBLE},
	)
	defer vk.FreeMemory(vk_state.logical_device, vs_staging_buffer_memory, nil)
	defer vk.DestroyBuffer(vk_state.logical_device, vs_staging_buffer, nil)
	buffer_memory: rawptr
	if result := vk.MapMemory(
		vk_state.logical_device,
		vs_staging_buffer_memory,
		0,
		vs_buffer_size,
		{},
		&buffer_memory,
	); result != .SUCCESS {
		panic("couldn't map memory for buffer")
	}
	defer vk.UnmapMemory(vk_state.logical_device, vs_staging_buffer_memory)


	command_pool_create_info := vk.CommandPoolCreateInfo {
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = vk_state.queue_family_index,
	}
	result = vk.CreateCommandPool(
		vk_state.logical_device,
		&command_pool_create_info,
		nil,
		&vk_state.command_pool,
	)
	defer vk.DestroyCommandPool(vk_state.logical_device, vk_state.command_pool, nil)
	assert(result == .SUCCESS)
	cmd_buffer_alloc_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = vk_state.command_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	cmd_buffer: vk.CommandBuffer
	if result := vk.AllocateCommandBuffers(
		vk_state.logical_device,
		&cmd_buffer_alloc_info,
		&cmd_buffer,
	); result != .SUCCESS {
		panic("Command buffer to copy content to vertex shader buffer couldn't be created")
	}
	defer vk.FreeCommandBuffers(vk_state.logical_device, vk_state.command_pool, 1, &cmd_buffer)

	cmd_begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}

	cmd_buffer_fence: vk.Fence
	vk.CreateFence(
		vk_state.logical_device,
		&vk.FenceCreateInfo{sType = .FENCE_CREATE_INFO, flags = {}},
		nil,
		&cmd_buffer_fence,
	)
	defer vk.DestroyFence(vk_state.logical_device, cmd_buffer_fence, nil)

	mem.copy(buffer_memory, &indices[0], len(indices) * size_of(u16))
	result = vk.BeginCommandBuffer(cmd_buffer, &cmd_begin_info)
	vk.CmdCopyBuffer(
		cmd_buffer,
		vs_staging_buffer,
		vk_state.index_buffer,
		1,
		&vk.BufferCopy{0, 0, index_buffer_size},
	)
	result = vk.EndCommandBuffer(cmd_buffer)


	vk.QueueSubmit(
		vk_state.queue,
		1,
		&vk.SubmitInfo {
			sType = .SUBMIT_INFO,
			commandBufferCount = 1,
			pCommandBuffers = &cmd_buffer,
		},
		cmd_buffer_fence,
	)
	vk.WaitForFences(vk_state.logical_device, 1, &cmd_buffer_fence, true, max(u64))
	result = vk.ResetFences(vk_state.logical_device, 1, &cmd_buffer_fence)

	texture_width, texture_height: u32
	vk_state.texture_image, vk_state.texture_memory, texture_width, texture_height = create_image(
		vk_state.logical_device,
		vk_state.physical_device,
		buffer_memory,
	)
	defer {
		vk.DestroyImage(vk_state.logical_device, vk_state.texture_image, nil)
		vk.FreeMemory(vk_state.logical_device, vk_state.texture_memory, nil)
	}

	vk_state.depth_image, vk_state.depth_image_view, vk_state.depth_memory, vk_state.depth_format =
		create_depth_image(vk_state.logical_device, vk_state.physical_device)
	defer {
		vk.DestroyImage(vk_state.logical_device, vk_state.depth_image, nil)
		vk.DestroyImageView(vk_state.logical_device, vk_state.depth_image_view, nil)
		vk.FreeMemory(vk_state.logical_device, vk_state.depth_memory, nil)
	}

	vk.BeginCommandBuffer(cmd_buffer, &cmd_begin_info)
	vk.CmdPipelineBarrier(
		cmd_buffer,
		{.TOP_OF_PIPE},
		{.TRANSFER},
		{},
		0,
		nil,
		0,
		nil,
		1,
		&vk.ImageMemoryBarrier {
			sType = .IMAGE_MEMORY_BARRIER,
			srcAccessMask = {},
			dstAccessMask = {.TRANSFER_WRITE},
			oldLayout = .UNDEFINED,
			newLayout = .TRANSFER_DST_OPTIMAL,
			image = vk_state.texture_image,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		},
	)
	vk.CmdCopyBufferToImage(
		cmd_buffer,
		vs_staging_buffer,
		vk_state.texture_image,
		.TRANSFER_DST_OPTIMAL,
		1,
		&vk.BufferImageCopy {
			bufferOffset = 0,
			bufferRowLength = 0,
			bufferImageHeight = 0,
			imageSubresource = vk.ImageSubresourceLayers {
				aspectMask = {.COLOR},
				mipLevel = 0,
				baseArrayLayer = 0,
				layerCount = 1,
			},
			imageOffset = {0, 0, 0},
			imageExtent = {texture_width, texture_height, 1},
		},
	)
	vk.CmdPipelineBarrier(
		cmd_buffer,
		{.TRANSFER},
		{.FRAGMENT_SHADER},
		{},
		0,
		nil,
		0,
		nil,
		1,
		&vk.ImageMemoryBarrier {
			sType = .IMAGE_MEMORY_BARRIER,
			srcAccessMask = {.TRANSFER_WRITE},
			dstAccessMask = {.SHADER_READ},
			oldLayout = .TRANSFER_DST_OPTIMAL,
			newLayout = .SHADER_READ_ONLY_OPTIMAL,
			image = vk_state.texture_image,
			dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		},
	)
	vk.EndCommandBuffer(cmd_buffer)
	vk.QueueSubmit(
		vk_state.queue,
		1,
		&vk.SubmitInfo {
			sType = .SUBMIT_INFO,
			commandBufferCount = 1,
			pCommandBuffers = &cmd_buffer,
		},
		cmd_buffer_fence,
	)
	vk.WaitForFences(vk_state.logical_device, 1, &cmd_buffer_fence, true, max(u64))
	vk.ResetFences(vk_state.logical_device, 1, &cmd_buffer_fence)

	vk.CreateImageView(
		vk_state.logical_device,
		&vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = vk_state.texture_image,
			viewType = .D2,
			format = .R8G8B8A8_SRGB,
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		},
		nil,
		&vk_state.texture_image_view,
	)
	vk.CreateSampler(
		vk_state.logical_device,
		&vk.SamplerCreateInfo {
			sType = .SAMPLER_CREATE_INFO,
			magFilter = .LINEAR,
			minFilter = .LINEAR,
			mipmapMode = .LINEAR,
			addressModeU = .REPEAT,
			addressModeV = .REPEAT,
			addressModeW = .REPEAT,
			anisotropyEnable = true,
			maxAnisotropy = vk_state.physical_device_properties.limits.maxSamplerAnisotropy,
			compareEnable = true,
			compareOp = .ALWAYS,
			borderColor = .INT_OPAQUE_BLACK,
			unnormalizedCoordinates = false,
		},
		nil,
		&vk_state.texture_sampler,
	)
	defer {
		vk.DestroySampler(vk_state.logical_device, vk_state.texture_sampler, nil)
		vk.DestroyImageView(vk_state.logical_device, vk_state.texture_image_view, nil)
	}

	layoutInfo := vk.DescriptorSetLayoutCreateInfo {
			sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			bindingCount = 2,
			pBindings    = raw_data(bindings[:]),
		}
	descriptor_set_layout: vk.DescriptorSetLayout
	result = vk.CreateDescriptorSetLayout(
		vk_state.logical_device,
		&layoutInfo,
		nil,
		&descriptor_set_layout,
	)
	defer vk.DestroyDescriptorSetLayout(vk_state.logical_device, descriptor_set_layout, nil)

	pipeline_layout_create_info := vk.PipelineLayoutCreateInfo {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 1,
		pSetLayouts            = &descriptor_set_layout,
		pushConstantRangeCount = 0,
	}
	vk.CreatePipelineLayout(
		vk_state.logical_device,
		&pipeline_layout_create_info,
		nil,
		&vk_state.pipeline_layout,
	)
	defer vk.DestroyPipelineLayout(vk_state.logical_device, vk_state.pipeline_layout, nil)

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		buffer_size: vk.DeviceSize : size_of(UniformBufferObject)
		vk_state.uniform_buffers[i], vk_state.uniform_buffer_memories[i] = create_buffer(
			vk_state.physical_device,
			vk_state.logical_device,
			buffer_size,
			{.UNIFORM_BUFFER},
			{.HOST_VISIBLE, .HOST_COHERENT},
		)
		vk.MapMemory(
			vk_state.logical_device,
			vk_state.uniform_buffer_memories[i],
			0,
			buffer_size,
			{},
			&vk_state.uniform_buffers_mapped[i],
		)
	}
	defer {
		for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
			vk.DestroyBuffer(vk_state.logical_device, vk_state.uniform_buffers[i], nil)
			vk.FreeMemory(vk_state.logical_device, vk_state.uniform_buffer_memories[i], nil)
		}
	}

	pool_sizes := [?]vk.DescriptorPoolSize {
		vk.DescriptorPoolSize{type = .UNIFORM_BUFFER, descriptorCount = MAX_FRAMES_IN_FLIGHT},
		vk.DescriptorPoolSize {
			type = .COMBINED_IMAGE_SAMPLER,
			descriptorCount = MAX_FRAMES_IN_FLIGHT,
		},
	}
	descriptor_pool_info := vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		flags         = {.FREE_DESCRIPTOR_SET},
		maxSets       = MAX_FRAMES_IN_FLIGHT,
		poolSizeCount = 2,
		pPoolSizes    = raw_data(pool_sizes[:]),
	}

	layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{}
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		layouts[i] = descriptor_set_layout
	}
	vk.CreateDescriptorPool(
		vk_state.logical_device,
		&descriptor_pool_info,
		nil,
		&vk_state.descriptor_pool,
	)
	defer vk.DestroyDescriptorPool(vk_state.logical_device, vk_state.descriptor_pool, nil)
	alloc_info := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = vk_state.descriptor_pool,
		descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
		pSetLayouts        = &layouts[0],
	}
	vk.AllocateDescriptorSets(
		vk_state.logical_device,
		&alloc_info,
		raw_data(vk_state.descriptor_sets[:]),
	)
	defer vk.FreeDescriptorSets(
		vk_state.logical_device,
		vk_state.descriptor_pool,
		MAX_FRAMES_IN_FLIGHT,
		&vk_state.descriptor_sets[0],
	)
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		buffer_info := vk.DescriptorBufferInfo {
			buffer = vk_state.uniform_buffers[i],
			offset = 0,
			range  = size_of(UniformBufferObject),
		}
		image_info := vk.DescriptorImageInfo {
			sampler     = vk_state.texture_sampler,
			imageView   = vk_state.texture_image_view,
			imageLayout = .SHADER_READ_ONLY_OPTIMAL,
		}
		descriptor_writes := [?]vk.WriteDescriptorSet {
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = vk_state.descriptor_sets[i],
				dstBinding = 0,
				dstArrayElement = 0,
				descriptorCount = 1,
				descriptorType = .UNIFORM_BUFFER,
				pBufferInfo = &buffer_info,
			},
			{
				sType = .WRITE_DESCRIPTOR_SET,
				dstSet = vk_state.descriptor_sets[i],
				dstBinding = 1,
				dstArrayElement = 0,
				descriptorCount = 1,
				descriptorType = .COMBINED_IMAGE_SAMPLER,
				pImageInfo = &image_info,
			},
		}
		vk.UpdateDescriptorSets(vk_state.logical_device, 2, raw_data(descriptor_writes[:]), 0, nil)
	}

	// Vulkan: Dynamic rendering format for the pipeline (no render pass)
	pipeline_rendering_create_info := vk.PipelineRenderingCreateInfo {
		sType                   = .PIPELINE_RENDERING_CREATE_INFO,
		colorAttachmentCount    = 1,
		pColorAttachmentFormats = &vk_state.surface_format.format,
		depthAttachmentFormat   = vk_state.depth_format,
	}

	vs_binding_description := get_binding_description()
	vs_attribute_descriptions := get_attribute_descriptions()
	vs_input_state_create_info := vk.PipelineVertexInputStateCreateInfo {
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 1,
		pVertexBindingDescriptions      = &vs_binding_description,
		vertexAttributeDescriptionCount = len(vs_attribute_descriptions),
		pVertexAttributeDescriptions    = raw_data(vs_attribute_descriptions[:]),
	}
	depth_create_info := vk.PipelineDepthStencilStateCreateInfo {
		sType            = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		depthTestEnable  = true,
		depthWriteEnable = true,
		depthCompareOp   = .LESS,
	}

	graphics_pipeline_create_info := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		pNext               = &pipeline_rendering_create_info,
		stageCount          = 2,
		pStages             = raw_data(pipeline_shader_stage_create_info[:]),
		pVertexInputState   = &vs_input_state_create_info,
		pInputAssemblyState = &input_assembly_create_info,
		pViewportState      = &viewport_create_info,
		pRasterizationState = &rasterization_create_info,
		pMultisampleState   = &multisample_create_info,
		pColorBlendState    = &color_blend_create_info,
		pDepthStencilState  = &depth_create_info,
		pDynamicState       = &dynamic_state_create_info,
		layout              = vk_state.pipeline_layout,
		renderPass          = {},
	}


	vk_state.vs_buffer, vk_state.vs_buffer_memory = create_buffer(
		vk_state.physical_device,
		vk_state.logical_device,
		vs_buffer_size,
		{.TRANSFER_DST, .VERTEX_BUFFER},
		{.DEVICE_LOCAL},
	)
	defer vk.FreeMemory(vk_state.logical_device, vk_state.vs_buffer_memory, nil)
	defer vk.DestroyBuffer(vk_state.logical_device, vk_state.vs_buffer, nil)


	mem.copy(buffer_memory, raw_data(vertices[:]), int(vs_buffer_size))
	result = vk.BeginCommandBuffer(cmd_buffer, &cmd_begin_info)
	vk.CmdCopyBuffer(
		cmd_buffer,
		vs_staging_buffer,
		vk_state.vs_buffer,
		1,
		&vk.BufferCopy{0, 0, vs_buffer_size},
	)
	result = vk.EndCommandBuffer(cmd_buffer)
	vk.QueueSubmit(
		vk_state.queue,
		1,
		&vk.SubmitInfo {
			sType = .SUBMIT_INFO,
			commandBufferCount = 1,
			pCommandBuffers = &cmd_buffer,
		},
		cmd_buffer_fence,
	)
	vk.WaitForFences(vk_state.logical_device, 1, &cmd_buffer_fence, true, max(u64))
	result = vk.ResetFences(vk_state.logical_device, 1, &cmd_buffer_fence)

	result = vk.CreateGraphicsPipelines(
		vk_state.logical_device,
		0,
		1,
		&graphics_pipeline_create_info,
		nil,
		&vk_state.graphics_pipeline,
	)
	defer vk.DestroyPipeline(vk_state.logical_device, vk_state.graphics_pipeline, nil)
	assert(result == .SUCCESS)

	command_buffer_create_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = vk_state.command_pool,
		level              = .PRIMARY,
		commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	}
	result = vk.AllocateCommandBuffers(
		vk_state.logical_device,
		&command_buffer_create_info,
		raw_data(vk_state.command_buffers[:]),
	)
	assert(result == .SUCCESS)

	// Main Loop
	for !glfw.WindowShouldClose(vk_state.window) {
		glfw.PollEvents()
		draw_frame()
	}

	result = vk.DeviceWaitIdle(vk_state.logical_device)
	assert(result == .SUCCESS)
}
