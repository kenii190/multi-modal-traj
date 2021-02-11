import gx
import numpy as np


#Main menu bar class
class MainMenuBar():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(self):
		self.need_exit = False
		self.need_save = False

	#--------------------------
	# Show
	#--------------------------
	def show(self, window):
		if window.begin_main_menu_bar():
			if window.begin_menu("File"):
				clicked, selected = window.show_menu_item("Quit", False)
				if clicked:
					self.need_exit = True

				clicked, selected = window.show_menu_item("Save", False)
				if clicked:
					self.need_save = True

				window.end_menu()
			window.end_main_menu_bar()


#Player window class
class PlayerWindow():
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, name, c_dim):
		self.name         = name
		self.c_dim        = c_dim
		self.img_tex_id   = 0
		self.frame_num    = 0
		self.data_idx     = 0
		self.max_data_idx = 0
		self.speed        = 1.0
		self.is_playing   = False
		self.need_reset   = False
		self.cs           = np.zeros((1, c_dim), dtype=np.float32)

	#-----------------------
	# Load next
	#-----------------------
	def load_next(self):
		if self.data_idx < self.max_data_idx:
			self.data_idx += 1
			self.need_reset = True

	#-----------------------
	# Load previous
	#-----------------------
	def load_prev(self):
		if self.data_idx > 0:
			self.data_idx -= 1
			self.need_reset = True

	#-----------------------
	# Show the window
	#-----------------------
	def show(self, window):
		window.begin(self.name)
		
		#Image
		window.show_image(self.img_tex_id, 800, 800)

		#Play / Pause / Loop
		if window.show_button("Play"):
			self.is_playing = True

		window.same_line()

		if window.show_button("Pause"):
			self.is_playing = False

		window.same_line()

		if window.show_button("Reset"):
			self.need_reset = True

		#Prev / Next
		if window.show_button("Prev"):
			self.load_prev()

		window.same_line()

		if window.show_button("Next"):
			self.load_next()

		#Frame / Speed / Data idx / cs
		changed, self.speed = window.show_slider_float("Speed", self.speed, 0.0, 2.0)

		changed, self.data_idx = window.show_slider_int("{:d} / {:d}".format(self.data_idx, self.max_data_idx), self.data_idx, 0, self.max_data_idx)
		if changed:
			self.need_reset = True

		for i in range(self.c_dim):
			changed, self.cs[0, i] = window.show_slider_float("z{:d}".format(i), self.cs[0, i], -2.0, 2.0)

		window.show_text("Frame {:d}".format(self.frame_num))

		#Delta time
		delta_time = window.get_delta_time()
		if delta_time < 1e-6:
			fps = 0
		else:
			fps = int(1.0 / delta_time)
		window.show_text("{:d} FPS".format(fps))

		window.end()


#---------------------------
# Do camera movement
#---------------------------
def do_cam_movement(window, camera):
	delta_x, delta_y = window.get_cursor_delta()
	delta_time = window.get_delta_time()
	#camera.process_mouse(delta_x, -delta_y, True)

	if window.get_key_down(gx.GLFW_KEY_W): camera.process_keyboard(gx.FORWARD, delta_time)
	if window.get_key_down(gx.GLFW_KEY_S): camera.process_keyboard(gx.BACKWARD, delta_time)
	if window.get_key_down(gx.GLFW_KEY_A): camera.process_keyboard(gx.LEFT, delta_time)
	if window.get_key_down(gx.GLFW_KEY_D): camera.process_keyboard(gx.RIGHT, delta_time)
	if window.get_key_down(gx.GLFW_KEY_SPACE): camera.process_keyboard(gx.UPWARD, delta_time)
	if window.get_key_down(gx.GLFW_KEY_LEFT_CONTROL): camera.process_keyboard(gx.DOWNWARD, delta_time)

#OpenGL window
#--------------------------------
window = gx.Window()
window.init_glfw("Policy", 1920, 1152)
window.init_glad()
window.init_imgui()
window.set_font("./DroidSans.ttf", 20.0)

gx.gl_enable(gx.GL_DEPTH_TEST)
gx.gl_enable(gx.GL_CULL_FACE)
gx.gl_cull_face(gx.GL_BACK)

#Cameras
#--------------------------------
camera_fix = gx.Camera(gx.vec3(0.0, 36.0, 36.0), gx.vec3(0, 1, 0), -90.0, -30.0, 1.0)
camera = gx.Camera(gx.vec3(130, 0, 10), gx.vec3(0, 1, 0), -90.0, 0.0, window.width/window.height)
camera.set_ortho(-288, 288, -172.8, 172.8, 0.01, 4000.0)
camera.set_speed(48.0)

#Textures
#--------------------------------
tex_albedo = gx.Texture()
tex_albedo.gen_buffer("./assets/textures/tiles/Tiles12_col.jpg", gx.GL_SRGB, gx.GL_BGR)

tex_normal = gx.Texture()
tex_normal.gen_buffer("./assets/textures/tiles/Tiles12_nrm.jpg", gx.GL_SRGB, gx.GL_BGR)

tex_metallic  = gx.Texture()
tex_metallic.gen_buffer(64, 64, gx.vec3(50))

tex_roughness = gx.Texture()
tex_roughness.gen_buffer(64, 64, gx.vec3(70))

tex_ao = gx.Texture()
tex_ao.gen_buffer(64, 64, gx.vec3(10))

#Create framebuffers
#--------------------------------
tex_multisampled1 = gx.Texture()
tex_multisampled1.gen_buffer(800, 800, 4)

tex_multisampled2 = gx.Texture()
tex_multisampled2.gen_buffer(window.width, window.height, 4)

tex_screen1 = gx.Texture()
tex_screen1.gen_buffer(800, 800, 2)

tex_screen2 = gx.Texture()
tex_screen2.gen_buffer(window.width, window.height, 2)

tex_depth = gx.Texture()
tex_depth.gen_buffer(1024, 1024, 0)

framebuffer_multisampled1 = gx.Framebuffer()
framebuffer_multisampled1.generate()
framebuffer_multisampled1.bind_texture(tex_multisampled1, 4)

framebuffer_multisampled2 = gx.Framebuffer()
framebuffer_multisampled2.generate()
framebuffer_multisampled2.bind_texture(tex_multisampled2, 4)

framebuffer_screen1 = gx.Framebuffer()
framebuffer_screen1.generate()
framebuffer_screen1.bind_texture(tex_screen1, 2)

framebuffer_screen2 = gx.Framebuffer()
framebuffer_screen2.generate()
framebuffer_screen2.bind_texture(tex_screen2, 2)

framebuffer_depth = gx.Framebuffer()
framebuffer_depth.generate()
framebuffer_depth.bind_texture(tex_depth, 0)

#Objects
#--------------------------------
#Sphere
v_data, n_data, uv_data, idx_data = gx.load_obj2("./assets/models/sphere.obj")
mesh_sphere = gx.Mesh()
mesh_sphere.gen_buffer()
mesh_sphere.upload_vertices(v_data, n_data, uv_data, idx_data)

obj_sphere = gx.Obj3d(mesh_sphere)
obj_sphere.add_texture(tex_depth)

#Cylinder
v_data, n_data, uv_data, idx_data = gx.load_obj2("./assets/models/cylinder.obj")
mesh_cylinder = gx.Mesh()
mesh_cylinder.gen_buffer()
mesh_cylinder.upload_vertices(v_data, n_data, uv_data, idx_data)

obj_cylinder = gx.Obj3d(mesh_cylinder)
obj_cylinder.add_texture(tex_depth)

#Ground
v_data, n_data, uv_data, idx_data = gx.create_quad()
mesh_quad = gx.Mesh()
mesh_quad.gen_buffer()
mesh_quad.upload_vertices(v_data, n_data, uv_data, idx_data)

obj_ground = gx.Obj3d(mesh_quad)
obj_ground.add_texture(tex_depth)
obj_ground.add_texture(tex_albedo)
obj_ground.add_texture(tex_normal)
obj_ground.add_texture(tex_metallic)
obj_ground.add_texture(tex_roughness)
obj_ground.add_texture(tex_ao)
obj_ground.scale = gx.vec3(512.0)

#Quad 2D
v_data, uv_data, idx_data = gx.create_quad2d()
mesh_2d = gx.Mesh()
mesh_2d.gen_buffer()
mesh_2d.upload_vertices2d(v_data, uv_data, idx_data, gx.GL_STATIC_DRAW)

obj_quad2d = gx.Obj2d(mesh_2d)
obj_quad2d.add_texture(tex_screen2)

#Highlight sphere
obj_highlight = gx.Obj3d(mesh_sphere)
obj_highlight.scale = gx.vec3(1.5)

obj_key = gx.Obj3d(mesh_sphere)
obj_key.scale = gx.vec3(0.8)

#Axis
mesh_axis = gx.Mesh()
mesh_axis.gen_buffer()
mesh_axis.upload_vertices([
	-200.0,    0.0,    0.0,
	 200.0,    0.0,    0.0,
	   0.0, -200.0,    0.0,
	   0.0,  200.0,    0.0,
	   0.0,    0.0, -200.0,
	   0.0,    0.0,  200.0
])
obj_axis = gx.Obj3d(mesh_axis)

#Shaders
#--------------------------------
light_positions = [gx.vec3(-20.0,  30.0, 20.0)]
light_colors = [gx.vec3(3000.0, 2000.0, 1000.0)]
ambient_power = 3.0
light_power = 2.0
near_plane = 0.01
far_plane  = 512.0
light_proj = gx.ortho(-20.0, 20.0, -20.0, 20.0, near_plane, far_plane)
light_view = gx.look_at(light_positions[0], gx.vec3(0.0, 10.0, 0.0), gx.vec3(0.0, 1.0, 0.0))

shader_2d = gx.Shader()
shader_2d.compile("./shaders/2d.vs", "./shaders/2d.fs")

shader_depth = gx.Shader()
shader_depth.compile("./shaders/simpleDepth.vs", "./shaders/simpleDepth.fs")
shader_depth.set_uniform("lightProj", light_proj)
shader_depth.set_uniform("lightView", light_view)

shader_pbr1 = gx.Shader()
shader_pbr1.compile("./shaders/pbrShadow.vs", "./shaders/pbrShadow.fs")
shader_pbr1.set_uniform("view", camera_fix.get_view())
shader_pbr1.set_uniform("proj", camera_fix.get_proj())
shader_pbr1.set_uniform("viewPos", camera_fix.get_pos())
shader_pbr1.set_uniform("lightProj", light_proj)
shader_pbr1.set_uniform("lightView", light_view)
shader_pbr1.set_uniform("shadowMap", 0)
shader_pbr1.set_uniform("albedo", gx.vec3(1.0, 0.0, 0.0))
shader_pbr1.set_uniform("ao", 1.0)
shader_pbr1.set_uniform("metallic", 0.7)
shader_pbr1.set_uniform("roughness", 0.2)
shader_pbr1.set_uniform("ambientPower", 0.1)

shader_pbr2 = gx.Shader()
shader_pbr2.compile("./shaders/pbrShadow.vs", "./shaders/pbrShadow.fs")
shader_pbr2.set_uniform("view", camera_fix.get_view())
shader_pbr2.set_uniform("proj", camera_fix.get_proj())
shader_pbr2.set_uniform("viewPos", camera_fix.get_pos())
shader_pbr2.set_uniform("lightProj", light_proj)
shader_pbr2.set_uniform("lightView", light_view)
shader_pbr2.set_uniform("shadowMap", 0)
shader_pbr2.set_uniform("albedo", gx.vec3(0.0, 0.7, 0.0))
shader_pbr2.set_uniform("ao", 1.0)
shader_pbr2.set_uniform("metallic", 0.7)
shader_pbr2.set_uniform("roughness", 0.2)
shader_pbr2.set_uniform("ambientPower", 0.1)

shader_pbr3 = gx.Shader()
shader_pbr3.compile("./shaders/pbrShadow.vs", "./shaders/pbrShadow.fs")
shader_pbr3.set_uniform("view", camera_fix.get_view())
shader_pbr3.set_uniform("proj", camera_fix.get_proj())
shader_pbr3.set_uniform("viewPos", camera_fix.get_pos())
shader_pbr3.set_uniform("lightProj", light_proj)
shader_pbr3.set_uniform("lightView", light_view)
shader_pbr3.set_uniform("shadowMap", 0)
shader_pbr3.set_uniform("albedo", gx.vec3(1.0, 1.0, 1.0))
shader_pbr3.set_uniform("ao", 1.0)
shader_pbr3.set_uniform("metallic", 0.2)
shader_pbr3.set_uniform("roughness", 0.2)
shader_pbr3.set_uniform("ambientPower", 0.1)

shader_pbrTex = gx.Shader()
shader_pbrTex.compile("./shaders/pbrTexShadow.vs", "./shaders/pbrTexShadow.fs")
shader_pbrTex.set_uniform("view", camera_fix.get_view())
shader_pbrTex.set_uniform("proj", camera_fix.get_proj())
shader_pbrTex.set_uniform("viewPos", camera_fix.get_pos())
shader_pbrTex.set_uniform("lightProj", light_proj)
shader_pbrTex.set_uniform("lightView", light_view)
shader_pbrTex.set_uniform("shadowMap", 0)
shader_pbrTex.set_uniform("albedoMap", 1)
shader_pbrTex.set_uniform("normalMap", 2)
shader_pbrTex.set_uniform("metallicMap", 3)
shader_pbrTex.set_uniform("roughnessMap", 4)
shader_pbrTex.set_uniform("aoMap", 5)
shader_pbrTex.set_uniform("repeat", 8.0)
shader_pbrTex.set_uniform("ambientPower", ambient_power)

shader_axis = gx.Shader()
shader_axis.compile("./shaders/axis.vs", "./shaders/axis.fs")
shader_axis.set_uniform("view", camera.get_view())
shader_axis.set_uniform("proj", camera.get_proj())

shader_instance1 = gx.Shader()
shader_instance1.compile("./shaders/pbrInstancing.vs", "./shaders/pbrInstancing.fs")
shader_instance1.set_uniform("view", camera.get_view())
shader_instance1.set_uniform("proj", camera.get_proj())
shader_instance1.set_uniform("albedo", gx.vec3(1.0, 0.0, 0.0))
shader_instance1.set_uniform("ao", 1.0)
shader_instance1.set_uniform("metallic", 0.2)
shader_instance1.set_uniform("roughness", 0.7)
shader_instance1.set_uniform("ambientColor", gx.vec3(0.5))
shader_instance1.set_uniform("lightPositions[0]", gx.vec3(0.0, 0.0, 0.0))
shader_instance1.set_uniform("lightColors[0]", gx.vec3(10000.0, 10000.0, 10000.0))

shader_instance2 = gx.Shader()
shader_instance2.compile("./shaders/pbrInstancing.vs", "./shaders/pbrInstancing.fs")
shader_instance2.set_uniform("view", camera.get_view())
shader_instance2.set_uniform("proj", camera.get_proj())
shader_instance2.set_uniform("albedo", gx.vec3(0.87, 0.67, 0.49))
shader_instance2.set_uniform("ao", 1.0)
shader_instance2.set_uniform("metallic", 0.2)
shader_instance2.set_uniform("roughness", 0.7)
shader_instance2.set_uniform("ambientColor", gx.vec3(0.5))
shader_instance2.set_uniform("lightPositions[0]", gx.vec3(0.0, 0.0, 0.0))
shader_instance2.set_uniform("lightColors[0]", gx.vec3(10000.0, 10000.0, 10000.0))

shader_highlight = gx.Shader()
shader_highlight.compile("./shaders/pbrLighting.vs", "./shaders/pbrLighting.fs")
shader_highlight.set_uniform("view", camera.get_view())
shader_highlight.set_uniform("proj", camera.get_proj())
shader_highlight.set_uniform("albedo", gx.vec3(0.0, 1.0, 0.0))
shader_highlight.set_uniform("ao", 1.0)
shader_highlight.set_uniform("metallic", 0.2)
shader_highlight.set_uniform("roughness", 0.7)
shader_highlight.set_uniform("ambientColor", gx.vec3(0.5))
shader_highlight.set_uniform("lightPositions[0]", gx.vec3(0.0, 0.0, 0.0))
shader_highlight.set_uniform("lightColors[0]", gx.vec3(10000.0, 10000.0, 10000.0))

for i in range(len(light_positions)):
	light_pos_str   = "lightPositions[" + str(i) + "]"
	light_color_str = "lightColors[" + str(i) + "]"

	shader_pbr1.set_uniform(light_pos_str, light_positions[i])
	shader_pbr1.set_uniform(light_color_str, light_power * light_colors[i])
	shader_pbr2.set_uniform(light_pos_str, light_positions[i])
	shader_pbr2.set_uniform(light_color_str, light_power * light_colors[i])
	shader_pbr3.set_uniform(light_pos_str, light_positions[i])
	shader_pbr3.set_uniform(light_color_str, light_power * light_colors[i])
	shader_pbrTex.set_uniform(light_pos_str, light_positions[i])
	shader_pbrTex.set_uniform(light_color_str, light_power * light_colors[i])
	
#Pose
#--------------------------------
pose = gx.Pose(mesh_sphere, mesh_cylinder, 1.0)
pose.build_hierarchy_by_amass()
pose.set_root_position(gx.vec3(-14.0, 16.0, -8.0))

pose_recon = gx.Pose(mesh_sphere, mesh_cylinder, 1.0)
pose_recon.build_hierarchy_by_amass()
pose_recon.set_root_position(gx.vec3(14.0, 16.0, -8.0))

#Plot
#--------------------------------
plot_scale = 168.0
plot = gx.Plot(mesh_sphere, mesh_cylinder, plot_scale, 0.6)

#GUI
#--------------------------------
main_menu_bar = MainMenuBar()
player_window = PlayerWindow("Player (Reconstruction)", 2)
player_window.img_tex_id = tex_screen1.get_tex_id()
player_window.load = True
progress_time = 0.0
key_vals = [
	gx.GLFW_KEY_1, gx.GLFW_KEY_2, gx.GLFW_KEY_3, gx.GLFW_KEY_4, gx.GLFW_KEY_5,
	gx.GLFW_KEY_6, gx.GLFW_KEY_7, gx.GLFW_KEY_8, gx.GLFW_KEY_9, gx.GLFW_KEY_0
]
key_states = [False] * 10
key_left_state = False
key_right_state = False

#---------------------------
# Render to tex (player)
#---------------------------
def render_player(motion, motion_recon, z_points):
	global progress_time

	plot.set_points(z_points)
	pose.apply_amass(motion)
	pose.contact_ground(0.0)
	pose_recon.apply_amass(motion_recon)
	pose_recon.contact_ground(0.0)
	obj_highlight.position = plot_scale * z_points[-1]

	if player_window.is_playing:
		progress_time += 120.0 * window.get_delta_time() * player_window.speed

		if progress_time >= 1:
			progress_time = 0.0
			player_window.frame_num += 1
			return True

	return False

#---------------------------
# Render to screen
#---------------------------
def render_screen():
	global key_vals
	global key_states
	global key_left_state
	global key_right_state

	if window.get_mouse_button_down(1):
		do_cam_movement(window, camera)
		shader_axis.set_uniform("view", camera.get_view())
		shader_instance1.set_uniform("view", camera.get_view())
		shader_instance2.set_uniform("view", camera.get_view())
		shader_highlight.set_uniform("view", camera.get_view())

	#Press 0~9
	for i in range(10):
		if key_states[i] == False and window.get_key_down(key_vals[i]):
			key_states[i] = True

		if key_states[i] == True and not window.get_key_down(key_vals[i]):
			key_states[i] = False

	#Press left key
	if key_left_state == False and window.get_key_down(gx.GLFW_KEY_LEFT):
		key_left_state = True

	if key_left_state == True and not window.get_key_down(gx.GLFW_KEY_LEFT):
		key_left_state = False

	#Press right key
	if key_right_state == False and window.get_key_down(gx.GLFW_KEY_RIGHT):
		key_right_state = True

	if key_right_state == True and not window.get_key_down(gx.GLFW_KEY_RIGHT):
		key_right_state = False

	#1. Render to depth framebuffer
	gx.gl_bind_framebuffer(gx.GL_FRAMEBUFFER, framebuffer_depth.get_fbo())
	gx.gl_enable(gx.GL_DEPTH_TEST)
	gx.gl_viewport(0, 0, 1024, 1024)
	gx.gl_clear(gx.GL_DEPTH_BUFFER_BIT)
	gx.gl_cull_face(gx.GL_FRONT)
	pose.draw(shader_depth, shader_depth)
	pose_recon.draw(shader_depth, shader_depth)
	gx.gl_cull_face(gx.GL_BACK)
	obj_ground.draw(shader_depth)

	#2. Render to multisampled framebuffer
	gx.gl_bind_framebuffer(gx.GL_FRAMEBUFFER, framebuffer_multisampled1.get_fbo())
	gx.gl_enable(gx.GL_DEPTH_TEST)
	gx.gl_viewport(0, 0, 800, 800)
	gx.gl_clear_color(0.1, 0.1, 0.1, 1.0)
	gx.gl_clear(gx.GL_COLOR_BUFFER_BIT | gx.GL_DEPTH_BUFFER_BIT)
	pose.draw(shader_pbr1, shader_pbr3)
	pose_recon.draw(shader_pbr2, shader_pbr3)
	obj_ground.draw(shader_pbrTex)

	gx.gl_bind_framebuffer(gx.GL_FRAMEBUFFER, framebuffer_multisampled2.get_fbo())
	gx.gl_enable(gx.GL_DEPTH_TEST)
	gx.gl_viewport(0, 0, window.width, window.height)
	gx.gl_clear_color(1.0, 1.0, 1.0, 1.0)
	gx.gl_clear(gx.GL_COLOR_BUFFER_BIT | gx.GL_DEPTH_BUFFER_BIT)
	obj_highlight.draw(shader_highlight)
	plot.draw(shader_instance1, shader_instance2)
	obj_axis.draw(shader_axis, mode=gx.GL_LINES)

	#3. Blit multisampled frambuffer to normal color framebuffer
	gx.gl_bind_framebuffer(gx.GL_READ_FRAMEBUFFER, framebuffer_multisampled1.get_fbo())
	gx.gl_bind_framebuffer(gx.GL_DRAW_FRAMEBUFFER, framebuffer_screen1.get_fbo())
	gx.gl_blit_framebuffer(0, 0, 800, 800, 0, 0, 800, 800, gx.GL_COLOR_BUFFER_BIT, gx.GL_NEAREST)

	gx.gl_bind_framebuffer(gx.GL_READ_FRAMEBUFFER, framebuffer_multisampled2.get_fbo())
	gx.gl_bind_framebuffer(gx.GL_DRAW_FRAMEBUFFER, framebuffer_screen2.get_fbo())
	gx.gl_blit_framebuffer(0, 0, window.width, window.height, 0, 0, window.width, window.height, gx.GL_COLOR_BUFFER_BIT, gx.GL_NEAREST)

	#4. Render to screen
	gx.gl_bind_framebuffer(gx.GL_FRAMEBUFFER, 0)
	gx.gl_disable(gx.GL_DEPTH_TEST)
	gx.gl_viewport(0, 0, window.width, window.height)
	gx.gl_clear_color(0.0, 0.0, 0.0, 1.0)
	gx.gl_clear(gx.GL_COLOR_BUFFER_BIT | gx.GL_DEPTH_BUFFER_BIT)
	obj_quad2d.draw(shader_2d)
	main_menu_bar.show(window)
	player_window.show(window)

#---------------------------
# Terminate window
#---------------------------
def terminate():
	tex_albedo.destroy()
	tex_normal.destroy()
	tex_metallic.destroy()
	tex_roughness.destroy()
	tex_ao.destroy()

	tex_multisampled1.destroy()
	tex_multisampled2.destroy()
	tex_screen1.destroy()
	tex_screen2.destroy()
	tex_depth.destroy()
	framebuffer_multisampled1.destroy()
	framebuffer_multisampled2.destroy()
	framebuffer_screen1.destroy()
	framebuffer_screen2.destroy()
	framebuffer_depth.destroy()

	mesh_sphere.destroy()
	mesh_cylinder.destroy()
	mesh_quad.destroy()
	mesh_2d.destroy()
	mesh_axis.destroy()

	shader_2d.destroy()
	shader_depth.destroy()
	shader_pbr1.destroy()
	shader_pbr2.destroy()
	shader_pbr3.destroy()
	shader_pbrTex.destroy()
	shader_axis.destroy()
	shader_instance1.destroy()
	shader_instance2.destroy()
	shader_highlight.destroy()

	window.terminate()