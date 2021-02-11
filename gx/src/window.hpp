#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <GLFW/glfw3.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

namespace py = pybind11;
using namespace std;


static bool keys[1024] = {false};
static float deltaX = 0, deltaY = 0;
static double lastX = 0, lastY = 0;


/*---------------------------
GLFW Key callback
---------------------------*/
static void keyCallback(GLFWwindow* window, int key, int scanCode, int action, int mode)
{
	if(key >= 0 && key < 1024)
	{
		if(action == GLFW_PRESS) keys[key] = true;
		else if(action == GLFW_RELEASE) keys[key] = false;
	}
}


/*---------------------------
GLFW cursor callback
---------------------------*/
static void cursorCallback(GLFWwindow *win, double xpos, double ypos)
{
	deltaX = (float)(xpos - lastX);
	deltaY = (float)(ypos - lastY);
	lastX  = xpos;
	lastY  = ypos;
}


class Window
{
public:
	/*----------------------
	Constructor
	----------------------*/
	Window()
	{
		window    = nullptr;
		io        = nullptr;
		lastTime  = 0;
		deltaTime = 0;
		fbWidth   = 0;
		fbHeight  = 0;
		mouseState[0] = false;
		mouseState[1] = false;
	}


	/*----------------------
	Init GLAD
	----------------------*/
	void initGLAD()
	{
	    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        	std::cout << "Failed to initialize GLAD" << std::endl;
	}


	/*----------------------
	Init GLFW
	----------------------*/
	py::tuple initGLFW(const string &name, const unsigned int width=1366, const unsigned int height=960)
	{
	    glfwInit();
	    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	    window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
	    if(window == nullptr)
	    {
	        cout << "Failed to create GLFW window" << endl;
	        glfwTerminate();
	        return py::make_tuple(fbWidth, fbHeight);
	    }

	    glfwMakeContextCurrent(window);
	    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
		glfwSetKeyCallback(window, keyCallback);
		glfwSetCursorPosCallback(window, cursorCallback);
		glfwSwapInterval(1);
		lastTime = glfwGetTime();

		return py::make_tuple(fbWidth, fbHeight);
	}


	void initImgui()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 330");
		io = &ImGui::GetIO();
	}


	/*----------------------
	Should window close?
	----------------------*/
	bool shouldClose()
	{
	    return glfwWindowShouldClose(window) || keys[GLFW_KEY_ESCAPE];
	}


	void setFont(const char *filename, const float size)
	{
		ImFont *font = io->Fonts->AddFontFromFileTTF(
			filename, 
			size, 
			NULL, 
			io->Fonts->GetGlyphRangesChineseFull()
		);
		IM_ASSERT(font != NULL);
	}


	bool begin(const char *name)
	{
		return ImGui::Begin(name);
	}


	bool beginMainMenuBar()
	{
		return ImGui::BeginMainMenuBar();
	}


	bool beginChild(const char *name, const int widht, const int height)
	{
		return ImGui::BeginChild(name, ImVec2(widht, height), true);
	}


	bool beginMenu(const char *name)
	{
		return ImGui::BeginMenu(name);
	}


	void end()
	{
		ImGui::End();
	}


	void endMainMenuBar()
	{
		ImGui::EndMainMenuBar();
	}


	void endChild()
	{
		ImGui::EndChild();
	}


	void endMenu()
	{
		ImGui::EndMenu();
	}


	void showImage(const unsigned int tid, const unsigned int width, const unsigned int height)
	{
		ImGui::Image(
			(ImTextureID)tid, 
			ImVec2(width, height), 
			ImVec2(0, 1), 
			ImVec2(1, 0), 
			ImVec4(1, 1, 1, 1), 
			ImVec4(0, 1, 0, 1)
		);
	}


	py::tuple showSliderFloat(const char *name, float val, const float min, const float max)
	{
		bool changed = ImGui::SliderFloat(name, &val, min, max, "%.2f");
		return py::make_tuple(changed, val);
	}


	py::tuple showSliderInt(const char *name, int val, const int min, const int max)
	{
		bool changed = ImGui::SliderInt(name, &val, min, max);
		return py::make_tuple(changed, val);
	}


	py::tuple showMenuItem(const char *name, bool selected)
	{
		bool clicked = ImGui::MenuItem(name, NULL, &selected);
		return py::make_tuple(clicked, selected);
	}


	void showText(const char *str, float r=1.0f, float g=1.0f, float b=1.0f)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(r, g, b, 1.0f));
		ImGui::Text(str);
		ImGui::PopStyleColor(1);
	}


	bool showButton(const char *name)
	{
		return ImGui::Button(name);
	}


	bool showCheckbox(const char *name, bool selected)
	{
		ImGui::Checkbox(name, &selected);
		return selected;
	}


	py::tuple showCombo(const char *name, int idx, vector<string> &items)
	{
		const int size = items.size();
		const char* charItems[64];

		for(int i = 0; i < size; ++i)
			charItems[i] = items[i].c_str();

		bool changed = ImGui::Combo(name, &idx, charItems, size);
		return py::make_tuple(changed, idx);
	}


	py::tuple showRegionPad(const int texID, int width, int height, float minVal, float maxVal)
	{
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::Image(
        	(ImTextureID)texID, 
        	ImVec2(width, height), 
        	ImVec2(0, 1), 
			ImVec2(1, 0), 
			ImVec4(1, 1, 1, 1), 
			ImVec4(0, 1, 0, 1)
        );
        int interval = maxVal - minVal;
        float xVal   = float(io->MousePos.x - pos.x) / width * interval + minVal;
        float yVal   = -(float(io->MousePos.y - pos.y) / height * interval + minVal);
        
        if(ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::Text("(%.2f, %.2f)", xVal, yVal);
            ImGui::EndTooltip();
        }

        if(ImGui::IsItemClicked())
        	return py::make_tuple(true, xVal, yVal);

        return py::make_tuple(false, xVal, yVal);
    }


	void sameLine()
	{
		ImGui::SameLine();
	}


	/*----------------------
	Start the frame
	----------------------*/
	void startFrame()
	{
		glfwPollEvents();
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}


	/*----------------------
	End the frame
	----------------------*/
	void endFrame()
	{
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);

	    float curTime = glfwGetTime();
	    deltaTime     = curTime - lastTime;
	    lastTime      = curTime;
	}


	/*----------------------
	Terminate window
	----------------------*/
	void terminate()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		glfwDestroyWindow(window);
		glfwTerminate();
	}


	/*----------------------
	Get delta time
	----------------------*/
	float getDeltaTime()
	{
		return deltaTime;
	}


	/*----------------------
	Get cursor pos delta
	----------------------*/
	py::tuple getCursorDelta()
	{
		float x = deltaX;
		float y = deltaY;
		deltaX = 0;
		deltaY = 0;

		return py::make_tuple(x, y);
	}


	/*----------------------
	Get mouse button down
	----------------------*/
	bool getMouseButtonDown(const unsigned int idx)
	{
		//Left mouse button
		if(idx == 0)
		{
			if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
				mouseState[0] = true;
			else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
				mouseState[0] = false;
		}
		//Right mouse button
		else if(idx == 1)
		{
			if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
				mouseState[1] = true;
			else if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
				mouseState[1] = false;
		}

		return mouseState[idx];
	}


	/*----------------------
	Get key down
	----------------------*/
	bool getKeyDown(const unsigned int idx)
	{
		return keys[idx];
	}


	/*----------------------
	Access width
	----------------------*/
	int getWidth()
	{
		return fbWidth;
	}


	void setWidth(int width)
	{
		fbWidth = width;
		glViewport(0, 0, fbWidth, fbHeight);
	}


	/*----------------------
	Access height
	----------------------*/
	int getHeight()
	{
		return fbHeight;
	}


	void setHeight(int height)
	{
		fbHeight = height;
		glViewport(0, 0, fbWidth, fbHeight);
	}


private:
	GLFWwindow *window;
	ImGuiIO *io;
	int fbWidth, fbHeight;
	float lastTime, deltaTime;
	bool mouseState[2];
};

#endif