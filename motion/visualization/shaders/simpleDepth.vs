#version 330 core
layout (location = 0) in vec3 position;

uniform mat4 lightProj;
uniform mat4 lightView;
uniform mat4 model;


void main()
{
	gl_Position = lightProj * lightView * model * vec4(position, 1.0);
}