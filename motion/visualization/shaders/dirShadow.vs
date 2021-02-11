#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

out VS_OUT{
	vec3 posViewSpace;
	vec3 nViewSpace;
	vec2 uv;
	vec4 posLightSpace;
}vs_out;

flat out vec3 lightPosViewSpace;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightProj;
uniform mat4 lightView;
uniform vec3 lightPos;


void main()
{
	vs_out.posViewSpace  = vec3(view * model * vec4(position, 1.0));
	vs_out.nViewSpace    = vec3(view * model * vec4(normal, 0.0));
	vs_out.uv            = uv;
	vs_out.posLightSpace = lightProj * lightView * model * vec4(position, 1.0);
	lightPosViewSpace    = vec3(view * vec4(lightPos, 1.0));

	gl_Position = proj * view * model * vec4(position, 1.0);
}