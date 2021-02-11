#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;


/*-----------------------------
Constructor
-----------------------------*/
Camera::Camera()
{
	this->pos = vec3(0.0f, 0.0f, 0.0f);
	this->front = vec3(0.0f, 0.0f, -1.0f);
	this->worldUp = vec3(0.0f, 1.0f, 0.0f);
	this->yaw = -90.0f;
	this->pitch = 0.0f;
	this->moveSpeed = 1.0f;
	this->mouseSense = 0.1f;
	this->scrollSense = 0.1f;
	this->proj = perspective(45.0f, 1024.0f / 768.0f, 0.01f, 2000.0f);

	updateCameraVector();
}


Camera::Camera(vec3 pos, vec3 worldUp, GLfloat yaw, GLfloat pitch, float ratio)
{
	this->pos = pos;
	this->front = vec3(0.0f, 0.0f, -1.0f);
	this->worldUp = worldUp;
	this->yaw = yaw;
	this->pitch = pitch;
	this->moveSpeed = 1.0f;
	this->mouseSense = 0.1f;
	this->scrollSense = 0.1f;
	this->proj = perspective(45.0f, ratio, 0.01f, 2000.0f);

	updateCameraVector();
}


/*-----------------------------
Get the view matrix
-----------------------------*/
mat4 Camera::getView()
{
	return lookAt(pos, pos + front, up);
}


/*-----------------------------
Access the projection matrix
-----------------------------*/
mat4 Camera::getProj()
{
	return proj;
}


void Camera::setProj(const mat4 &p)
{
	proj = p;
}


/*-----------------------------
Get camera position
-----------------------------*/
vec3 Camera::getPos()
{
	return pos;
}


/*-----------------------------
Get front vector
-----------------------------*/
vec3 Camera::getFront()
{
	return front;
}


/*-----------------------------
Set camera speed
-----------------------------*/
void Camera::setSpeed(float speed)
{
	moveSpeed = speed;
}


/*-----------------------------
Set orthogonal projection
-----------------------------*/
void Camera::setOrtho(float left, float right, float bottom, float top, float near, float far)
{
	this->proj = ortho(left, right, bottom, top, near, far);
}


/*-----------------------------
Process keyboard input
-----------------------------*/
void Camera::processKeyboard(CameraMove dir, GLfloat deltaTime)
{
	GLfloat velocity = moveSpeed * deltaTime;

	switch(dir)
	{
		case FORWARD:  pos += front * velocity; break;
		case BACKWARD: pos -= front * velocity; break;
		case LEFT:     pos -= right * velocity; break;
		case RIGHT:    pos += right * velocity; break;
		case UPWARD:   pos += up * velocity; break;
		case DOWNWARD: pos -= up * velocity; break;
	}
}


/*-----------------------------
Process mouse input
-----------------------------*/
void Camera::processMouse(GLfloat offsetX, GLfloat offsetY, GLboolean constrainPitch)
{
	yaw += offsetX * mouseSense;
	pitch += offsetY * mouseSense;

	if(constrainPitch)
	{
		if(pitch > 89.0f) pitch = 89.0f;
		else if(pitch < -89.0f) pitch = -89.0f;
	}

	updateCameraVector();
}


/*-----------------------------
Update camera vector
-----------------------------*/
void Camera::updateCameraVector()
{
	vec3 front_unscaled;
	GLfloat cosPitch = cos(radians(pitch));

	front_unscaled.x = cosPitch * cos(radians(yaw));
	front_unscaled.y = sin(radians(pitch));
	front_unscaled.z = cosPitch * sin(radians(yaw));

	front = normalize(front_unscaled);
	right = normalize(cross(front, worldUp));
	up = normalize(cross(right, front));
}