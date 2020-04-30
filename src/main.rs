// This code roughly follows https://vulkan-tutorial.com/ - Drawing a triangle

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Entry,
};
use std::{
    ffi::{c_void, CStr, CString},
    io::Cursor,
};
use structopt::StructOpt;
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const FRAMES_IN_FLIGHT: usize = 2;
const SHADER_VERT: &[u8] = include_bytes!("triangle.vert.spv");
const SHADER_FRAG: &[u8] = include_bytes!("triangle.frag.spv");

#[derive(Debug, StructOpt)]
struct Opt {
    /// Use validation layers
    #[structopt(short, long)]
    validation_layers: bool,
}

unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    eprintln!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );

    vk::FALSE
}

fn main() {
    let opt = Opt::from_args();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("erupt-examples: triangle")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let entry = Entry::new().unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let application_name = CString::new("Hello Triangle").unwrap();
    let engine_name = CString::new("No Engine").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&application_name)
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let mut instance_extensions = ash_window::enumerate_required_extensions(&window).unwrap();
    if opt.validation_layers {
        instance_extensions.push(DebugUtils::name());
    }

    let mut instance_layers = Vec::new();
    if opt.validation_layers {
        instance_layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
    }

    let device_extensions = vec![Swapchain::name()];

    let mut device_layers = Vec::new();
    if opt.validation_layers {
        device_layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
    }

    let instance_extensions: Vec<_> = instance_extensions.iter().map(|s| s.as_ptr()).collect();
    let instance_layers: Vec<_> = instance_layers.iter().map(|s| s.as_ptr()).collect();
    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    let instance = unsafe { entry.create_instance(&create_info, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
    let debug_utils = if opt.validation_layers {
        let debug_utils = DebugUtils::new(&entry, &instance);

        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        let messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) }.unwrap();

        Some((debug_utils, messenger))
    } else {
        None
    };

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Window_surface
    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }.unwrap();
    let surface_loader = Surface::new(&entry, &instance);

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
    let (physical_device, queue_family, format, present_mode, properties) =
        unsafe { instance.enumerate_physical_devices() }
            .unwrap()
            .into_iter()
            .filter_map(|physical_device| unsafe {
                let queue_family = match instance
                    .get_physical_device_queue_family_properties(physical_device)
                    .into_iter()
                    .enumerate()
                    .position(|(i, properties)| {
                        properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && surface_loader
                                .get_physical_device_surface_support(
                                    physical_device,
                                    i as u32,
                                    surface,
                                )
                                .unwrap()
                                == true
                    }) {
                    Some(queue_family) => queue_family as u32,
                    None => return None,
                };

                let formats = surface_loader
                    .get_physical_device_surface_formats(physical_device, surface)
                    .unwrap();
                let format = match formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .and_then(|_| formats.get(0))
                {
                    Some(surface_format) => surface_format.clone(),
                    None => return None,
                };

                let present_mode = surface_loader
                    .get_physical_device_surface_present_modes(physical_device, surface)
                    .unwrap()
                    .into_iter()
                    .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX)
                    .unwrap_or(vk::PresentModeKHR::FIFO);

                let supported_extensions = instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap();
                if !device_extensions.iter().all(|device_extension| {
                    supported_extensions.iter().any(|properties| {
                        &CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                    })
                }) {
                    return None;
                }

                let properties = instance.get_physical_device_properties(physical_device);
                Some((
                    physical_device,
                    queue_family,
                    format,
                    present_mode,
                    properties,
                ))
            })
            .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            })
            .expect("No suitable physical device found");

    println!("Using physical device: {:?}", unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
    });

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Logical_device_and_queues
    let queue_create_info = vec![vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])
        .build()];
    let features = vk::PhysicalDeviceFeatures::builder();

    let device_extensions: Vec<_> = device_extensions.iter().map(|s| s.as_ptr()).collect();
    let device_layers: Vec<_> = device_layers.iter().map(|s| s.as_ptr()).collect();
    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);

    let device = unsafe { instance.create_device(physical_device, &create_info, None) }.unwrap();

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let swapchain_loader = Swapchain::new(&instance, &device);

    let create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(surface_caps.current_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None) }.unwrap();
    let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views
    let swapchain_image_views: Vec<_> = swapchain_images
        .iter()
        .map(|swapchain_image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            unsafe { device.create_image_view(&create_info, None) }.unwrap()
        })
        .collect();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules
    let entry_point = CString::new("main").unwrap();

    let vert_decoded = util::read_spv(&mut Cursor::new(SHADER_VERT)).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::builder().code(&vert_decoded);
    let shader_vert = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

    let frag_decoded = util::read_spv(&mut Cursor::new(SHADER_FRAG)).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::builder().code(&frag_decoded);
    let shader_frag = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

    let shader_stages = vec![
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_vert)
            .name(&entry_point)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_frag)
            .name(&entry_point)
            .build(),
    ];

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewports = vec![vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(surface_caps.current_extent.width as f32)
        .height(surface_caps.current_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)
        .build()];
    let scissors = vec![vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(surface_caps.current_extent)
        .build()];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_clamp_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachments = vec![vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)
        .build()];
    let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let create_info = vk::PipelineLayoutCreateInfo::builder();
    let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Render_passes
    let attachments = vec![vk::AttachmentDescription::builder()
        .format(format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let color_attachment_refs = vec![vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];
    let subpasses = vec![vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .build()];
    let dependencies = vec![vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build()];

    let create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let render_pass = unsafe { device.create_render_pass(&create_info, None) }.unwrap();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Conclusion
    let create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build();

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    }
    .unwrap()[0];

    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Framebuffers
    let swapchain_framebuffers: Vec<_> = swapchain_image_views
        .iter()
        .map(|image_view| {
            let attachments = vec![*image_view];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(surface_caps.current_extent.width)
                .height(surface_caps.current_extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }.unwrap()
        })
        .collect();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
    let create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family);
    let command_pool = unsafe { device.create_command_pool(&create_info, None) }.unwrap();

    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap();

    for (&command_buffer, &framebuffer) in command_buffers.iter().zip(swapchain_framebuffers.iter())
    {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.unwrap();

        let clear_values = vec![vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: surface_caps.current_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(command_buffer);

            device.end_command_buffer(command_buffer).unwrap();
        }
    }

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation
    let create_info = vk::SemaphoreCreateInfo::builder();
    let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&create_info, None) }.unwrap())
        .collect();
    let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_semaphore(&create_info, None) }.unwrap())
        .collect();

    let create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| unsafe { device.create_fence(&create_info, None) }.unwrap())
        .collect();
    let mut images_in_flight: Vec<_> = swapchain_images.iter().map(|_| vk::Fence::null()).collect();

    let mut frame = 0;
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Poll;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(keycode),
                state,
                ..
            }) => match (keycode, state) {
                (VirtualKeyCode::Escape, ElementState::Released) => {
                    *control_flow = ControlFlow::Exit
                }
                _ => (),
            },
            _ => (),
        },
        Event::MainEventsCleared => {
            unsafe {
                device
                    .wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX)
                    .unwrap();
            }

            let image_index = unsafe {
                swapchain_loader.acquire_next_image(
                    swapchain,
                    u64::MAX,
                    image_available_semaphores[frame],
                    vk::Fence::null(),
                )
            }
            .unwrap()
            .0;

            let image_in_flight = images_in_flight[image_index as usize];
            if image_in_flight != vk::Fence::null() {
                unsafe { device.wait_for_fences(&[image_in_flight], true, u64::MAX) }.unwrap();
            }
            images_in_flight[image_index as usize] = in_flight_fences[frame];

            let wait_semaphores = vec![image_available_semaphores[frame]];
            let command_buffers = vec![command_buffers[image_index as usize]];
            let signal_semaphores = vec![render_finished_semaphores[frame]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            unsafe {
                let in_flight_fence = in_flight_fences[frame];
                device.reset_fences(&[in_flight_fence]).unwrap();
                device
                    .queue_submit(queue, &[submit_info], in_flight_fence)
                    .unwrap()
            }

            let swapchains = vec![swapchain];
            let image_indices = vec![image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe { swapchain_loader.queue_present(queue, &present_info) }.unwrap();

            frame = (frame + 1) % FRAMES_IN_FLIGHT;
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            for &semaphore in image_available_semaphores
                .iter()
                .chain(render_finished_semaphores.iter())
            {
                device.destroy_semaphore(semaphore, None);
            }

            for &fence in &in_flight_fences {
                device.destroy_fence(fence, None);
            }

            device.destroy_command_pool(command_pool, None);

            for &framebuffer in &swapchain_framebuffers {
                device.destroy_framebuffer(framebuffer, None);
            }

            device.destroy_pipeline(pipeline, None);

            device.destroy_render_pass(render_pass, None);

            device.destroy_pipeline_layout(pipeline_layout, None);

            device.destroy_shader_module(shader_vert, None);
            device.destroy_shader_module(shader_frag, None);

            for &image_view in &swapchain_image_views {
                device.destroy_image_view(image_view, None);
            }

            swapchain_loader.destroy_swapchain(swapchain, None);

            device.destroy_device(None);

            surface_loader.destroy_surface(surface, None);

            if let Some((debug_utils, messenger)) = &debug_utils {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }

            instance.destroy_instance(None);
            println!("Exited cleanly");
        },
        _ => (),
    })
}
