#version 330 core

out vec4 color;

in VS_OUT{
	vec3 posViewSpace;
	vec3 nViewSpace;
	vec2 uv;
	vec4 posLightSpace;
}fs_in;

flat in vec3 lightPosViewSpace;

uniform sampler2D shadowMap;


float calcShadow(vec4 posLightSpace, vec3 normal, vec3 lightDir)
{
	vec3 projCoords = posLightSpace.xyz / posLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;

	float currentDepth = projCoords.z;

	//Keep the shadow at 0.0 when outside the farPlane region
	if(currentDepth > 1.0)
		return 0;

	//Add bias
	float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);

	//PCF
	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(shadowMap, 0);

	for(int x = -2; x <= 2; ++x)
		for(int y = -2; y <= 2; ++y)
		{
			float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
			shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
		}

	return shadow / 25.0;
}


void main()
{
	vec3 viewDir  = normalize(-fs_in.posViewSpace);
	vec3 lightDir = normalize(lightPosViewSpace - fs_in.posViewSpace);
	vec3 normal   = normalize(fs_in.nViewSpace);

	//Ambient
	vec3 ambient = 0.3 * vec3(1.0);

	//Diffuse
	vec3 diffuse  = max(dot(lightDir, normal), 0.0) * vec3(1.0);

	//Specular
	vec3 halfwayDir = normalize(lightDir + viewDir);
	vec3 specular   = pow(max(dot(normal, halfwayDir), 0.0), 64.0) * vec3(1.0);

	//Shadow
	float shadow = calcShadow(fs_in.posLightSpace, normal, lightDir);

	color = vec4((ambient + (1.0 - shadow) * (diffuse + specular)), 1.0); 
}