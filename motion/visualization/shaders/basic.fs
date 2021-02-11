#version 330 core

struct Light{
	vec3 position;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float power;
};

layout(location=0) out vec4 color;

uniform Light light;

flat in vec3 lightPos_c;
in vec3 position_c;
in vec3 normal_c;
in vec2 uv_c;


void main()
{
	//Convert the world space to the camera space
	vec3 eyeDir_c = vec3(0, 0, 0) - position_c;
	vec3 lightDir_c = normalize(lightPos_c + eyeDir_c);
	eyeDir_c = normalize(eyeDir_c);

	//Distance from light to vertex
	float distance = length(lightPos_c - position_c);
	
	//Cosine between normal & lightDir
	float cosTheta = clamp(dot(normal_c, lightDir_c), 0, 1);
	
	//Cosine between eyeDir & relectDir, use halfway vector h = lightDir_c + eyeDir_c
	float cosPhi = clamp(dot(normal_c, normalize(lightDir_c + eyeDir_c)), 0, 1);
	
	//Compute Phong shading model
	vec3 ambient  = vec3(1.0) * light.ambient;
	vec3 diffuse  = vec3(1.0) * light.diffuse * cosTheta / (distance * distance);
	vec3 specular = vec3(1.0) * light.specular * pow(cosPhi, 8) / (distance * distance);

	color = vec4(ambient + (diffuse + specular) * light.power, 1.0);
}