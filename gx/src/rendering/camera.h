#ifndef CAMERA_H
#define CAMERA_H

#include "glad/glad.h"
#include <glm/glm.hpp>

using namespace glm;


enum CameraMove
{
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UPWARD,
	DOWNWARD
};


class Camera
{
public:
	Camera();
	Camera(vec3 pos, vec3 worldUp, GLfloat yaw, GLfloat pitch, float ratio=1.0f);
	mat4 getView();
	mat4 getProj();
	void setProj(const mat4 &p);
	vec3 getPos();
	vec3 getFront();
	void setOrtho(float left, float right, float bottom, float top, float near, float far);
	void setSpeed(float speed);
	void processKeyboard(CameraMove dir, GLfloat deltaTime);
	void processMouse(GLfloat offsetX, GLfloat offsetY, GLboolean constrainPitch);

private:
	vec3 pos;
	vec3 front;
	vec3 up;
	vec3 right;
	vec3 worldUp;
	GLfloat yaw;
	GLfloat pitch;
	GLfloat moveSpeed;
	GLfloat mouseSense;
	GLfloat scrollSense;
	mat4 proj;

	void updateCameraVector();
};

#endif