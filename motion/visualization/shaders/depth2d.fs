#version 330 core

out vec4 color;

in vec2 outUV;

uniform sampler2D depthMap;
uniform float nearPlane;
uniform float farPlane;


float linearizeDepth(float depth)
{
	//Back to NDC
	float z = depth * 2.0 - 1.0;
	return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}


void main()
{
	float depthVal = texture(depthMap, outUV).r;
	color = vec4(vec3(depthVal), 1.0);
}