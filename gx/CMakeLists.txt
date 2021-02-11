cmake_minimum_required(VERSION 3.5.1)
project(gx)
add_definitions(-std=c++14)
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/glfw)
include_directories(lib/glfw/include)

#OpenGL
find_package(OpenGL REQUIRED)

#OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

#glad_lib
include_directories(lib/glad/include)
add_library(glad_lib lib/glad/src/glad.c)
set_target_properties(glad_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

#glm
include_directories(lib/glm)

#tinyobjloader_lib
include_directories(lib/tinyobjloader)
add_library(tinyobjloader_lib lib/tinyobjloader/tiny_obj_loader.cc)
set_target_properties(tinyobjloader_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

#imgui_lib
include_directories(lib/imgui)
add_library(imgui_lib 
	../lib/imgui/imgui.cpp 
	../lib/imgui/imgui_demo.cpp 
	../lib/imgui/imgui_draw.cpp 
	../lib/imgui/imgui_impl_glfw.cpp 
	../lib/imgui/imgui_impl_opengl3.cpp 
	../lib/imgui/imgui_widgets.cpp 
)
set_target_properties(imgui_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

#render_lib
include_directories(src/rendering)
add_library(render_lib STATIC 
	src/rendering/camera.cpp 
	src/rendering/obj3d.cpp 
	src/rendering/obj2d.cpp 
	src/rendering/shader.cpp 
	src/rendering/mesh.cpp 
	src/rendering/texture.cpp 
	src/rendering/framebuffer.cpp
	src/rendering/pose.cpp
	src/rendering/plot.cpp
)
set_target_properties(render_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(render_lib tinyobjloader_lib)

#pybind11
add_subdirectory(pybind11)
pybind11_add_module(gx src/main.cpp)
target_link_libraries(gx PRIVATE glad_lib render_lib imgui_lib glfw ${OpenCV_LIBS})