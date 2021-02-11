#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out VS_OUT{
	vec3 normal;
}vs_out;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;


void main()
{
	gl_Position = proj * view * model * vec4(aPos, 1.0);
	mat3 normalMat = mat3(transpose(view * model));
	vs_out.normal = normalize(vec3(proj * vec4(normalMat * aNormal, 0.0)));
}