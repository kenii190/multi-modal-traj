#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in mat4 instanceMat;

out vec2 TexCoords;

uniform mat4 proj;
uniform mat4 view;


void main()
{
	gl_Position = proj * view * instanceMat * vec4(aPos, 1.0);
	TexCoords   = aTexCoords;
}