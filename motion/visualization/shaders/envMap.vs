#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 n;
out vec3 pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;


void main()
{
	n   = mat3(transpose(inverse(model))) * normal;
	pos = vec3(model * vec4(position, 1.0));
	gl_Position = proj * view * model * vec4(position, 1.0);
}