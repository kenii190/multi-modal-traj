#version 330 core

out vec4 color;

in vec3 n;
in vec3 pos;

uniform vec3 cameraPos;
uniform samplerCube skybox;


void main()
{
	vec3 I = normalize(pos - cameraPos);
	vec3 R = refract(I, normalize(n), 1.00/1.52);

	color = vec4(texture(skybox, R).rgb, 1.0);
}