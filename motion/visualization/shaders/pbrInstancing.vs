#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in mat4 instanceMat;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;

uniform mat4 proj;
uniform mat4 view;


void main()
{
	TexCoords = aTexCoords;
	WorldPos  = vec3(instanceMat * vec4(aPos, 1.0));
	Normal    = mat3(instanceMat) * aNormal;

	gl_Position = proj * view * vec4(WorldPos, 1.0);
}