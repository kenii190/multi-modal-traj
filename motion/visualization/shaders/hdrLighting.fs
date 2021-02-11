#version 330 core

out vec4 color;

in VS_OUT{
	vec3 FragPos;
	vec3 Normal;
	vec2 TexCoords;
}fs_in;

struct Light{
	vec3 Position;
	vec3 Color;
};

uniform Light lights[4];
uniform sampler2D diffuseTex;
uniform vec3 viewPos;


void main()
{
	vec3 texColor = texture(diffuseTex, fs_in.TexCoords).rgb;
	vec3 normal   = normalize(fs_in.Normal);
	vec3 ambient  = vec3(0.0);
	vec3 lighting = vec3(0.0);

	for(int i = 0; i < 4; ++i)
	{
		//Diffuse
		vec3 lightDir = normalize(lights[i].Position - fs_in.FragPos);
		vec3 diffuse  = max(dot(lightDir, normal), 0.0) * lights[i].Color * texColor;
		
		//Attenuation
		float dist = length(fs_in.FragPos - lights[i].Position);
		lighting  += diffuse / (dist * dist);
	}

	color = vec4(ambient + lighting, 1.0);
}