#version 330 core

out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;
uniform vec3 ambientColor;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
uniform vec3 viewPos;

const float PI = 3.14159265359;


float distributionGGX(vec3 n, vec3 h, float roughness)
{
	float a      = roughness * roughness;
	float a2     = a * a;
	float NdotH  = max(dot(n, h), 0.0);
	float NdotH2 = NdotH * NdotH;
	
	float nom    = a2;
	float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / max(denom, 0.001);
}


float geometrySchlickGGX(float NdotV, float roughness)
{
	float r = roughness + 1.0;
	float k = r*r / 8.0;
	
	float nom   = NdotV;
	float denom = NdotV*(1.0 - k) + k;

	return nom / denom;
}


float geometrySmith(vec3 n, vec3 v, vec3 l, float roughness)
{
	float NdotV = max(dot(n, v), 0.0);
	float NdotL = max(dot(n, l), 0.0);

	return geometrySchlickGGX(NdotL, roughness) * geometrySchlickGGX(NdotV, roughness);
}


vec3 fresnelSchlick(float cosTheta, vec3 f0)
{
	return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}


void main()
{
	vec3 n = normalize(Normal);
	vec3 v = normalize(viewPos - WorldPos);
	float NdotV = max(dot(n, v), 0.0);

	vec3 f0 = vec3(0.04);
	f0 = mix(f0, albedo, metallic);

	vec3 Lo = vec3(0.0);

	for(int i = 0; i < 4; ++i)
	{
		vec3 l = normalize(lightPositions[i] - WorldPos);
		vec3 h = normalize(v + l);
		float NdotL = max(dot(n, l), 0.0);

		//Per-light radiance
		float dist        = length(lightPositions[i] - WorldPos);
		float attenuation = 1.0 / (dist * dist);
		vec3 radiance     = lightColors[i] * attenuation;

		//Cook-Torrance BRDF
		float N = distributionGGX(n, h, roughness);
		float G = geometrySmith(n, v, l, roughness);
		vec3 F  = fresnelSchlick(clamp(dot(h, v), 0.0, 1.0), f0);

		vec3 nominator    = N * G * F;
		float denominator = 4 * NdotV * NdotL;
		vec3 specular     = nominator / max(denominator, 0.001);

		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - metallic;

		//Reflectance equation
		Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	}

	//Ambient
	vec3 ambient = ambientColor * albedo * ao;
	vec3 color   = ambient + Lo;

	//HDR tonemapping
	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/2.2));

	FragColor = vec4(color, 1.0);
}