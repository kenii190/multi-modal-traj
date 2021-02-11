#version 330 core

out vec4 color;

in VS_OUT{
	vec3 FragPos;
	vec3 Normal;
	vec2 TexCoords;
}fs_in;

uniform samplerCube depthMap;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float farPlane;

vec3 gridSamplingDisk[20] = vec3[](
	vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
	vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
	vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
	vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
	vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);


float calcShadow(vec3 fragPos)
{
	vec3 fragToLight   = fragPos - lightPos;
	float currentDepth = length(fragToLight);
	float shadow       = 0.0;
	float bias         = 0.05;
	float viewDistance = length(viewPos - fragPos);
	float diskRadius   = (1.0 + (viewDistance / farPlane)) / 25.0;

	for(int i = 0; i < 20; ++i)
	{
		float closestDepth = texture(depthMap, fragToLight + gridSamplingDisk[i] * diskRadius).r;
		closestDepth *= farPlane;

		if(currentDepth - bias > closestDepth)
		shadow += 1.0;
	}
	shadow /= 20.0;

	return shadow;
}


void main()
{
	vec3 normal   = normalize(fs_in.Normal);
	vec3 lightDir = normalize(lightPos - fs_in.FragPos);
	vec3 viewDir  = normalize(viewPos - fs_in.FragPos);

	//Ambient
	vec3 ambient = 0.3 * vec3(1.0);

	//Diffuse
	vec3 diffuse = max(dot(lightDir, normal), 0.0) * vec3(1.0);

	//Specular
	vec3 halfwayDir = normalize(lightDir + viewDir);
	vec3 specular   = pow(max(dot(normal, halfwayDir), 0.0), 64.0) * vec3(1.0);

	//Shadow
	float shadow = calcShadow(fs_in.FragPos);

	color = vec4(ambient + (1.0 - shadow) * (diffuse + specular), 1.0) * 0.5;
}