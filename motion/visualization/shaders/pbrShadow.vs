#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 texCoords;
out vec3 normal;
out vec3 posWorldSpace;
out vec4 posLightSpace;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightProj;
uniform mat4 lightView;


void main()
{
	texCoords     = aTexCoords;
	normal        = mat3(model) * aNormal;
	posWorldSpace = vec3(model * vec4(aPos, 1.0));
	posLightSpace = lightProj * lightView * model * vec4(aPos, 1.0);

	gl_Position = proj * view * vec4(posWorldSpace, 1.0);
}