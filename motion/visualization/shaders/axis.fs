#version 330

layout(location = 0) out vec4 color;

in vec3 pos;


void main()
{
	/*
	if(pos.x != 0)
		color.rgb = vec3(0.5, 0.0, 0.0);
	else if(pos.y != 0)
		color.rgb = vec3(0.0, 0.5, 0.0);
	else
		color.rgb = vec3(0.0, 0.0, 0.8);
	*/
	color.rgb = vec3(0.0, 0.0, 0.0);
}