#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;

out vec2 outUV;

uniform mat3 model;


void main()
{
	vec3 pos    = model * vec3(position, 1.0);
	gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
	outUV       = uv;
}