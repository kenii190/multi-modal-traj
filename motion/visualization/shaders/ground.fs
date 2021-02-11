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

layout(location=0) out vec4 color;

#define NR_POINT_LIGHTS 4
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform DirLight dirLight;
uniform SpotLight spotLight;
uniform sampler2D tex;

flat in vec3 pointLightPos_c[NR_POINT_LIGHTS];
flat in vec3 dirLightDir_c;
flat in vec3 spotLightPos_c;
flat in vec3 spotLightDir_c;
in vec3 position_g;
in vec3 position_c;
in vec3 normal_c;
in vec2 uv_c;


vec3 calcDirLight(DirLight light, vec3 lightDir, vec3 normal, vec3 viewVec, vec3 texColor);
vec3 calcPointLight(PointLight light, vec3 lightPos, vec3 normal, vec3 viewVec, vec3 texColor);
vec3 calcSpotLight(SpotLight light, vec3 lightPos, vec3 spotDir, vec3 normal, vec3 viewVec, vec3 texColor);


/*---------------------
Main
---------------------*/
void main()
{
	vec3 viewVec  = -position_c;
	vec3 texColor = texture(tex, uv_c*16).rgb;

	//1. Directional Light
	vec3 result = calcDirLight(dirLight, dirLightDir_c, normal_c, viewVec, texColor);

	//2. Point Light
	for(int i = 0; i < NR_POINT_LIGHTS; ++i)
		result += calcPointLight(pointLights[i], pointLightPos_c[i], normal_c, viewVec, texColor);

	//3. Spot Light
	result += calcSpotLight(spotLight, spotLightPos_c, spotLightDir_c, normal_c, viewVec, texColor);

	color = vec4(result, 1.0);
}


/*---------------------
Directional light
---------------------*/
vec3 calcDirLight(DirLight light, vec3 lightDir, vec3 normal, vec3 viewVec, vec3 texColor)
{
	lightDir  = -lightDir;
	float cosTheta = max(dot(normal, lightDir), 0.0);
	float cosPhi   = max(dot(normal, normalize(lightDir + normalize(viewVec))), 0.0);

	vec3 ambient  = texColor * light.ambient;
	vec3 diffuse  = texColor * light.diffuse * cosTheta;
	vec3 specular = light.specular * pow(cosPhi, 8.0);

	return ambient + (diffuse + specular)*light.power;
}


/*---------------------
Point light
---------------------*/
vec3 calcPointLight(PointLight light, vec3 lightPos, vec3 normal, vec3 viewVec, vec3 texColor)
{
	vec3 lightDir     = normalize(lightPos + viewVec);
	vec3 viewDir      = normalize(viewVec);
	float dist        = length(lightPos);
	float cosTheta    = max(dot(normal, lightDir), 0.0);
	float cosPhi      = max(dot(normal, normalize(lightDir + viewDir)), 0.0);
	float attenuation = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);

	vec3 ambient  = texColor * light.ambient;
	vec3 diffuse  = texColor * light.diffuse * cosTheta;
	vec3 specular = light.specular * pow(cosPhi, 8.0);

	return (ambient + (diffuse + specular)*light.power) * attenuation;
}


/*---------------------
Spot light
---------------------*/
vec3 calcSpotLight(SpotLight light, vec3 lightPos, vec3 spotDir, vec3 normal, vec3 viewVec, vec3 texColor)
{
	vec3 lightDir     = normalize(lightPos + viewVec);
	vec3 viewDir      = normalize(viewVec);
	float dist        = length(lightPos);
	float cosTheta    = max(dot(normal, lightDir), 0.0);
	float cosPhi      = max(dot(normal, normalize(lightDir + viewDir)), 0.0);
	float attenuation = 1.0 / (1.0 + 0.09*dist + 0.032*dist*dist);
	float theta       = dot(lightDir, -spotDir);
	float epsilon     = light.cutOff - light.outerCutOff;
	float intensity   = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

	vec3 ambient  = texColor * light.ambient;
	vec3 diffuse  = texColor * light.diffuse * cosTheta;
	vec3 specular = light.specular * pow(cosPhi, 8.0);

	return (ambient + (diffuse + specular)*light.power*intensity) * attenuation;
}