#version 330 core

out vec4 color;

uniform vec3 tint;


void main()
{
	color = vec4(tint, 1.0);
}