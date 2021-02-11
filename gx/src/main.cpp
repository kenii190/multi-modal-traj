#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "camera.h"
#include "shader.h"
#include "texture.h"
#include "mesh.h"
#include "obj3d.h"
#include "obj2d.h"
#include "pose.h"
#include "plot.h"
#include "framebuffer.h"
#include "objloader.hpp"
#include "primitive.hpp"
#include "window.hpp"

namespace py = pybind11;
using namespace std;


py::tuple createQuad()
{
    vector<float> vData, nData, uvData;
    vector<unsigned int> idxData;

    Primitive::createQuad(vData, nData, uvData, idxData);
    return py::make_tuple(vData, nData, uvData, idxData);
}


py::tuple createQuad2d()
{
    vector<float> vData, uvData;
    vector<unsigned int> idxData;

    Primitive::createQuad2d(vData, uvData, idxData);
    return py::make_tuple(vData, uvData, idxData);
}


py::tuple createCube()
{
    vector<float> vData, nData, uvData;
    vector<unsigned int> idxData;

    Primitive::createCube(vData, nData, uvData, idxData);
    return py::make_tuple(vData, nData, uvData, idxData);
}


py::tuple loadObj(const char *filename)
{
    vector<float> vData, nData, uvData;
    vector<unsigned int> idxData;

    ObjLoader::loadObj(filename, vData, nData, uvData, idxData);
    return py::make_tuple(vData, nData, uvData, idxData);
}


py::tuple loadObj2(const char *filename)
{
    vector<float> vData, nData, uvData;
    vector<unsigned int> idxData;

    ObjLoader::loadObj2(filename, vData, nData, uvData, idxData);
    return py::make_tuple(vData, nData, uvData, idxData);
}


void glViewport_(GLint x, GLint y, GLsizei width, GLsizei height)
{
    glViewport(x, y, width, height);
}


void glClearColor_(GLclampf r, GLclampf g, GLclampf b, GLclampf a)
{
    glClearColor(r, g, b, a);
}


void glEnable_(GLenum cap)
{
    glEnable(cap);
}


void glDisable_(GLenum cap)
{
    glDisable(cap);
}


void glCullFace_(GLenum mode)
{
    glCullFace(mode);
}


void glClear_(GLbitfield mask)
{
    glClear(mask);
}


void glBindFramebuffer_(GLenum target, GLuint framebuffer)
{
    glBindFramebuffer(target, framebuffer);
}


void glBlitFramebuffer_(
    GLint srcX0,
    GLint srcY0,
    GLint srcX1,
    GLint srcY1,
    GLint dstX0,
    GLint dstY0,
    GLint dstX1,
    GLint dstY1,
    GLbitfield mask,
    GLenum filter
)
{
    glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}


glm::mat4 ortho_(
    float const &left,
    float const &right,
    float const &bottom,
    float const &top,
    float const &zNear,
    float const &zFar
)
{
    return glm::ortho(left, right, bottom, top, zNear, zFar);
}


glm::mat4 lookAt_(
    glm::vec3 const &eye,
    glm::vec3 const &center,
    glm::vec3 const &up
)
{
    return glm::lookAt(eye, center, up);
}


PYBIND11_MODULE(gx, m) 
{
    m.doc() = R"pbdoc(
        Test
    )pbdoc";

    //GLFW attributes
    m.attr("GLFW_KEY_0") = GLFW_KEY_0;
    m.attr("GLFW_KEY_1") = GLFW_KEY_1;
    m.attr("GLFW_KEY_2") = GLFW_KEY_2;
    m.attr("GLFW_KEY_3") = GLFW_KEY_3;
    m.attr("GLFW_KEY_4") = GLFW_KEY_4;
    m.attr("GLFW_KEY_5") = GLFW_KEY_5;
    m.attr("GLFW_KEY_6") = GLFW_KEY_6;
    m.attr("GLFW_KEY_7") = GLFW_KEY_7;
    m.attr("GLFW_KEY_8") = GLFW_KEY_8;
    m.attr("GLFW_KEY_9") = GLFW_KEY_9;
    m.attr("GLFW_KEY_W") = GLFW_KEY_W;
    m.attr("GLFW_KEY_A") = GLFW_KEY_A;
    m.attr("GLFW_KEY_S") = GLFW_KEY_S;
    m.attr("GLFW_KEY_D") = GLFW_KEY_D;
    m.attr("GLFW_KEY_Q") = GLFW_KEY_Q;
    m.attr("GLFW_KEY_E") = GLFW_KEY_E;
    m.attr("GLFW_KEY_R") = GLFW_KEY_R;
    m.attr("GLFW_KEY_LEFT") = GLFW_KEY_LEFT;
    m.attr("GLFW_KEY_RIGHT") = GLFW_KEY_RIGHT;
    m.attr("GLFW_KEY_UP") = GLFW_KEY_UP;
    m.attr("GLFW_KEY_DOWN") = GLFW_KEY_DOWN;
    m.attr("GLFW_KEY_SPACE") = GLFW_KEY_SPACE;
    m.attr("GLFW_KEY_LEFT_CONTROL") = GLFW_KEY_LEFT_CONTROL;

    //GL attributes
    m.attr("GL_DEPTH_TEST") = GL_DEPTH_TEST;
    m.attr("GL_CULL_FACE") = GL_CULL_FACE;
    m.attr("GL_FRONT") = GL_FRONT;
    m.attr("GL_BACK") = GL_BACK;
    m.attr("GL_NEAREST") = GL_NEAREST;
    m.attr("GL_TRIANGLES") = GL_TRIANGLES;
    m.attr("GL_POINTS") = GL_POINTS;
    m.attr("GL_LINES") = GL_LINES;
    m.attr("GL_LINE_STRIP") = GL_LINE_STRIP;
    m.attr("GL_TRIANGLE_STRIP") = GL_TRIANGLE_STRIP;
    m.attr("GL_STATIC_DRAW") = GL_STATIC_DRAW;
    m.attr("GL_DYNAMIC_DRAW") = GL_DYNAMIC_DRAW;
    m.attr("GL_COLOR_BUFFER_BIT") = GL_COLOR_BUFFER_BIT;
    m.attr("GL_DEPTH_BUFFER_BIT") = GL_DEPTH_BUFFER_BIT;
    m.attr("GL_FRAMEBUFFER_SRGB") = GL_FRAMEBUFFER_SRGB;
    m.attr("GL_FRAMEBUFFER") = GL_FRAMEBUFFER;
    m.attr("GL_READ_FRAMEBUFFER") = GL_READ_FRAMEBUFFER;
    m.attr("GL_DRAW_FRAMEBUFFER") = GL_DRAW_FRAMEBUFFER;
    m.attr("GL_SRGB") = GL_SRGB;
    m.attr("GL_RGB") = GL_RGB;
    m.attr("GL_BGR") = GL_BGR;
    
    //Functions
    m.def("create_quad", &createQuad);
    m.def("create_quad2d", &createQuad2d);
    m.def("create_cube", &createCube);
    m.def("load_obj", &loadObj);
    m.def("load_obj2", &loadObj2);
    m.def("gl_viewport", &glViewport_);
    m.def("gl_clear_color", &glClearColor_);
    m.def("gl_enable", &glEnable_);
    m.def("gl_disable", &glDisable_);
    m.def("gl_cull_face", &glCullFace_);
    m.def("gl_clear", &glClear_);
    m.def("gl_bind_framebuffer", &glBindFramebuffer_);
    m.def("gl_blit_framebuffer", &glBlitFramebuffer_);
    m.def("ortho", &ortho_);
    m.def("look_at", &lookAt_);

    //CameraMove
    py::enum_<CameraMove>(m, "CameraMove")
        .value("FORWARD", CameraMove::FORWARD)
        .value("BACKWARD", CameraMove::BACKWARD)
        .value("LEFT", CameraMove::LEFT)
        .value("RIGHT", CameraMove::RIGHT)
        .value("UPWARD", CameraMove::UPWARD)
        .value("DOWNWARD", CameraMove::DOWNWARD)
        .export_values();

    //Window
    py::class_<Window>(m, "Window")
        .def(py::init<>())
        .def("init_glfw", &Window::initGLFW)
        .def("init_glad", &Window::initGLAD)
        .def("init_imgui", &Window::initImgui)
        .def("set_font", &Window::setFont)
        .def("should_close", &Window::shouldClose)
        .def("start_frame", &Window::startFrame)
        .def("end_frame", &Window::endFrame)
        .def("terminate", &Window::terminate)
        .def("get_delta_time", &Window::getDeltaTime)
        .def("get_cursor_delta", &Window::getCursorDelta)
        .def("get_mouse_button_down", &Window::getMouseButtonDown)
        .def("get_key_down", &Window::getKeyDown)
        .def("begin", &Window::begin)
        .def("begin_main_menu_bar", &Window::beginMainMenuBar)
        .def("begin_child", &Window::beginChild)
        .def("begin_menu", &Window::beginMenu)
        .def("end", &Window::end)
        .def("end_main_menu_bar", &Window::endMainMenuBar)
        .def("end_child", &Window::endChild)
        .def("end_menu", &Window::endMenu)
        .def("show_image", &Window::showImage)
        .def("show_slider_float", &Window::showSliderFloat)
        .def("show_slider_int", &Window::showSliderInt)
        .def("show_menu_item", &Window::showMenuItem)
        .def("show_text", &Window::showText, "", py::arg("str")=nullptr, py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f)
        .def("show_button", &Window::showButton)
        .def("show_checkbox", &Window::showCheckbox)
        .def("show_combo", &Window::showCombo)
        .def("show_region_pad", &Window::showRegionPad)
        .def("same_line", &Window::sameLine)
        .def_property("width", &Window::getWidth, &Window::setWidth)
        .def_property("height", &Window::getHeight, &Window::setHeight);

    //Camera
    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def(py::init<glm::vec3, glm::vec3, float, float, float>())
        .def("get_view", &Camera::getView)
        .def("get_proj", &Camera::getProj)
        .def("get_pos", &Camera::getPos)
        .def("get_front", &Camera::getFront)
        .def("process_mouse", &Camera::processMouse)
        .def("process_keyboard", &Camera::processKeyboard)
        .def("set_speed", &Camera::setSpeed)
        .def("set_ortho", &Camera::setOrtho);

    //Mesh
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<>())
        .def("gen_buffer", &Mesh::genBuffer)
        .def("destroy", &Mesh::destroy)
        .def(
            "upload_vertices", 
            (void (Mesh::*)(vector<float>&, const GLenum)) &Mesh::uploadVertices, 
            "", 
            py::arg("v")=vector<float>(), 
            py::arg("usage")=GL_STATIC_DRAW
        )
        .def(
            "upload_vertices", 
            (void (Mesh::*)(vector<float>&, vector<float>&, vector<float>&, const GLenum)) &Mesh::uploadVertices, 
            "", 
            py::arg("v")=vector<float>(), 
            py::arg("n")=vector<float>(), 
            py::arg("uv")=vector<float>(), 
            py::arg("usage")=GL_STATIC_DRAW
        )
        .def(
            "upload_vertices", 
            (void (Mesh::*)(vector<float>&, vector<unsigned int>&, const GLenum)) &Mesh::uploadVertices,
            "", 
            py::arg("v")=vector<float>(), 
            py::arg("idx")=vector<unsigned int>(), 
            py::arg("usage")=GL_STATIC_DRAW
        )
        .def(
            "upload_vertices", 
            (void (Mesh::*)(vector<float>&, vector<float>&, vector<unsigned int>&, const GLenum)) &Mesh::uploadVertices,
            "", 
            py::arg("v")=vector<float>(), 
            py::arg("n")=vector<float>(), 
            py::arg("idx")=vector<unsigned int>(), 
            py::arg("usage")=GL_STATIC_DRAW
        )
        .def(
            "upload_vertices", 
            (void (Mesh::*)(vector<float>&, vector<float>&, vector<float>&, vector<unsigned int>&, const GLenum)) &Mesh::uploadVertices,
            "", 
            py::arg("v")=vector<float>(), 
            py::arg("n")=vector<float>(), 
            py::arg("uv")=vector<float>(), 
            py::arg("idx")=vector<unsigned int>(), 
            py::arg("usage")=GL_STATIC_DRAW
        )
        .def("upload_vertices2d", &Mesh::uploadVertices2d)
        .def("draw", (void (Mesh::*)()) &Mesh::draw)
        .def("draw", (void (Mesh::*)(GLenum mode)) &Mesh::draw)
        .def("draw_instanced", &Mesh::drawInstanced)
        .def("bind_vao", &Mesh::bindVAO);

    //Shader
    py::class_<Shader>(m, "Shader")
        .def(py::init<>())
        .def("destroy", &Shader::destroy)
        .def(
            "compile", 
            &Shader::compile,
            "", 
            py::arg("vertexPath")=nullptr, 
            py::arg("fragmentPath")=nullptr,
            py::arg("geomPath")=nullptr
        )
        .def("use", &Shader::use)
        .def("set_uniform", (void (Shader::*)(const char*, const int)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const glm::mat3&)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const glm::mat4&)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const glm::vec3&)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const glm::vec4&)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const float)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const unsigned int)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const int)) &Shader::setUniform)
        .def("set_uniform", (void (Shader::*)(const char*, const bool)) &Shader::setUniform);

    //Texture
    py::class_<Texture>(m, "Texture")
        .def(py::init<>())
        .def("gen_buffer", (void (Texture::*)(const unsigned int, const unsigned int, const glm::vec3&)) &Texture::genBuffer)
        .def("gen_buffer", (void (Texture::*)(const unsigned int, const unsigned int, const unsigned int)) &Texture::genBuffer)
        .def("gen_buffer", (void (Texture::*)(const char*, const int, const GLenum)) &Texture::genBuffer)
        .def("gen_buffer", (void (Texture::*)(const string*, const int, const GLenum)) &Texture::genBuffer)
        .def("destroy", &Texture::destroy)
        .def("load_from_file", &Texture::loadFromFile)
        .def("load_cubemap_from_file", &Texture::loadCubemapFromFile)
        .def("clear", &Texture::clear)
        .def("draw_circle", &Texture::drawCircle)
        .def("draw_line", &Texture::drawLine)
        .def("upload", &Texture::upload)
        .def("get_tex_id", &Texture::getTexID)
        .def("get_width", &Texture::getWidth)
        .def("get_height", &Texture::getHeight)
        .def("get_tex_type", &Texture::getTexType)
        .def("bind", &Texture::bind);

    //Framebuffer
    py::class_<Framebuffer>(m, "Framebuffer")
        .def(py::init<>())
        .def("generate", &Framebuffer::generate)
        .def("destroy", &Framebuffer::destroy)
        .def("bind_texture", &Framebuffer::bindTexture)
        .def("get_fbo", &Framebuffer::getFBO);

    //Obj3d
    py::class_<Obj3d>(m, "Obj3d")
        .def(py::init<>())
        .def(py::init<Mesh*>())
        .def("set_mesh", &Obj3d::setMesh)
        .def("add_texture", &Obj3d::addTexture)
        .def("update_model", &Obj3d::updateModel)
        .def("draw", &Obj3d::draw, "", py::arg("s")=(Shader*)nullptr, py::arg("mode")=GL_TRIANGLES)
        .def("draw_without_model", &Obj3d::drawWithoutModel)
        .def("draw_instanced", &Obj3d::drawInstanced)
        .def_property("position", &Obj3d::getPosition, &Obj3d::setPosition)
        .def_property("rotation", (glm::quat (Obj3d::*)()) &Obj3d::getRotation, (void (Obj3d::*)(const glm::quat&)) &Obj3d::setRotation)
        .def_property("scale", &Obj3d::getScale, &Obj3d::setScale);

    //Obj2d
    py::class_<Obj2d>(m, "Obj2d")
        .def(py::init<>())
        .def(py::init<Mesh*>())
        .def("set_mesh", &Obj2d::setMesh)
        .def("add_texture", &Obj2d::addTexture)
        .def("update_model", &Obj2d::updateModel)
        .def("draw", &Obj2d::draw, "", py::arg("s")=(Shader*)nullptr, py::arg("mode")=GL_TRIANGLES)
        .def_property("position", &Obj2d::getPosition, &Obj2d::setPosition)
        .def_property("rotation", &Obj2d::getRotation, &Obj2d::setRotation)
        .def_property("scale", &Obj2d::getScale, &Obj2d::setScale);

    //Pose
    py::class_<Pose>(m, "Pose")
        .def(py::init<Mesh*, Mesh*, float>())
        .def("load_from_asf_file", &Pose::loadFromAsfFile)
        .def("build_hierarchy_by_amass", &Pose::buildHierarchyByAMASS)
        .def("apply", &Pose::apply)
        .def("apply_amass", &Pose::applyAMASS)
        .def("set_root_position", &Pose::setRootPosition)
        .def("set_root_rotation", &Pose::setRootRotation)
        .def("contact_ground", &Pose::contactGround)
        .def("draw", &Pose::draw);

    //Plot
    py::class_<Plot>(m, "Plot")
        .def(py::init<Mesh*, Mesh*, float, float>())
        .def("set_points", &Plot::setPoints)
        .def("get_points", &Plot::getPoints)
        .def("draw", &Plot::draw);

    //vec2
    py::class_<glm::vec2>(m, "vec2")
        .def(py::init<const float, const float>())
        .def(py::init<const float>());

    //vec3
    py::class_<glm::vec3>(m, "vec3")
        .def(py::init<const float, const float, const float>())
        .def(py::init<const float>())
        .def(float() * py::self)
        .def_readwrite("x", &vec3::x)
        .def_readwrite("y", &vec3::y)
        .def_readwrite("z", &vec3::z);

    //vec4
    py::class_<glm::vec4>(m, "vec4")
        .def(py::init<const float, const float, const float, const float>())
        .def(py::init<const float>());

    //mat3
    py::class_<glm::mat3>(m, "mat3")
        .def(py::init<const float>());

    //mat4
    py::class_<glm::mat4>(m, "mat4")
        .def(py::init<const float>());

    //quat
    py::class_<glm::quat>(m, "quat")
        .def(py::init<const float, const float, const float, const float>());

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
