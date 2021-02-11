#ifndef SHADER_H
#define SHADER_H

#include "glad/glad.h"
#include <glm/glm.hpp>

using namespace glm;


struct Material{
	vec3 diffuse;
	vec3 specular;
	float shininess;

	Material() : diffuse(vec3(1.0f)), specular(vec3(1.0f)), shininess(10.0f){}
	
	Material(vec3 d, vec3 s, float sh)
	{
		diffuse   = d;
		specular  = s;
		shininess = sh;
	}
};


struct Light{
	vec3 position;
	vec3 direction;
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
	float power;

	Light() : position(vec3(0.0)), direction(vec3(1.0, 0.0, 0.0)), ambient(vec3(1.0)), diffuse(vec3(1.0)), specular(vec3(1.0)), power(1.0){}

	Light(vec3 p, vec3 dir, vec3 a, vec3 d, vec3 s, float pow)
	{
		position  = p;
		direction = dir;
		ambient   = a;
		diffuse   = d;
		specular  = s;
		power     = pow;
	}
};


class Shader
{
public:
	Shader();
	void destroy();
	bool compile(const GLchar *vertexPath, const GLchar *fragmentPath, const GLchar *geomPath=nullptr);
	void use();
	void setUniform(const char* name, const mat4 &mat);
	void setUniform(const char* name, const mat3 &mat);
	void setUniform(const char* name, const vec3 &v);
	void setUniform(const char* name, const vec4 &v);
	void setUniform(const char* name, const GLfloat value);
	void setUniform(const char* name, const GLuint value);
	void setUniform(const char* name, const GLint value);
	void setUniform(const char* name, const bool value);
	void setLightUniform(const char* name, const Light light);

private:
	GLuint program;
};

#endif