use ash::{version::EntryV1_0, vk};
use std::error::Error;
use winit::{
    event::{DeviceEvent, Event, KeyboardInput, StartCause, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop)?;

    let entry = ash::Entry::new()?;
    let surface_extensions = ash_window::enumerate_required_extensions(&window)?;
    let instance_extensions = surface_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();
    let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_version(1, 0, 0));
    let instance_desc = vk::InstanceCreateInfo::builder()
        .application_info(&app_desc)
        .enabled_extension_names(&instance_extensions);

    let instance = unsafe { entry.create_instance(&instance_desc, None)? };

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None)? };
    let surface_fn = ash::extensions::khr::Surface::new(&entry, &instance);
    println!("surface: {:?}", surface);

    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Wait;
        }
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(VirtualKeyCode::Escape),
                ..
            }) => {
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        },
        Event::LoopDestroyed => unsafe {
            surface_fn.destroy_surface(surface, None);
        },
        _ => (),
    });
}
