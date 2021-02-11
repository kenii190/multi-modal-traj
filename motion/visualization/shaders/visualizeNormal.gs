#version 330 core
layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in VS_OUT{
	vec3 normal;
}gs_in[];


void generateLine(int idx)
{
	gl_Position = gl_in[idx].gl_Position;
	EmitVertex();
	gl_Position = gl_in[idx].gl_Position + vec4(gs_in[idx].normal, 0.0) * 0.1;
	EmitVertex();
	EndPrimitive();
}


void main()
{
	generateLine(0);
	generateLine(1);
	generateLine(2);
}