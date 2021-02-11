#version 330

struct DirLight{
	vec3 dir;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float power;
};

struct PointLight{
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float power;
};

struct SpotLight{
	vec3 position;
	vec3 dir;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float power;
	float cutOff;
	float outerCutOff;
};

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;

#define NR_POINT_LIGHTS 4
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform DirLight dirLight;
uniform SpotLight spotLight;
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

flat out vec3 pointLightPos_c[NR_POINT_LIGHTS];
flat out vec3 dirLightDir_c;
flat out vec3 spotLightPos_c;
flat out vec3 spotLightDir_c;

out VS_OUT{
	vec3 pos;
	vec3 n;
	vec2 uv;
}vs_out;


void main()
{
	//vertex position
	gl_Position = proj * view * model * vec4(position, 1.0);
	
	//Convert the world space to the camera space
	vs_out.pos = (view * model * vec4(position, 1.0)).xyz;
	vs_out.n   = normalize((view * model * vec4(normal, 0)).xyz);
	vs_out.uv  = uv;

	//Convert light pos/dir to camera space
	for(int i = 0; i < NR_POINT_LIGHTS; ++i)
		pointLightPos_c[i] = (view * vec4(pointLights[i].position, 1)).xyz;

	dirLightDir_c  = normalize((view * vec4(dirLight.dir, 0)).xyz);
	spotLightPos_c = (view * vec4(spotLight.position, 1)).xyz;
	spotLightDir_c = normalize((view * vec4(spotLight.dir, 0)).xyz);
}