#version 330 core
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;

out vec2 uv;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;


void main()
{
	uv = aUV;
	gl_Position = proj * view * model * vec4(aPosition, 1.0);
}