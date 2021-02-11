#version 330 core

struct Light{
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float power;
};

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform Light light;

flat out vec3 lightPos_c;
out vec3 position_c;
out vec3 normal_c;
out vec2 uv_c;


void main()
{
	gl_Position = proj * view * model * vec4(position, 1.0);
	
	//Convert the world space to the camera space
	position_c = (view * model * vec4(position, 1.0)).xyz;
	normal_c   = normalize((view * model * vec4(normal, 0)).xyz);
	lightPos_c = (view * vec4(light.position, 1)).xyz;
	uv_c       = uv;
}