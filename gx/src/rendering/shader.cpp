#include "shader.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;


/*-----------------------------
Check compile errors
-----------------------------*/
static void checkCompileErrors(GLuint shader, string type)
{
	GLint success;
	GLchar infoLog[1024];

	if(type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if(!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			cout << "ERROR: Shader compilation error of type: " << type << endl << infoLog << endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if(!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			cout << "ERROR: Program linking error of type: " << type << endl << infoLog << endl;
		}
	}
}


/*-----------------------------
Constructor
-----------------------------*/
Shader::Shader()
{
	program = 0;
}


/*-----------------------------
Destroy program
-----------------------------*/
void Shader::destroy()
{
	glDeleteProgram(program);
}


/*-----------------------------
Compile shaders
-----------------------------*/
bool Shader::compile(const GLchar *vertexPath, const GLchar *fragmentPath, const GLchar *geomPath)
{
	//1. Read shader files
	string vertexCode, fragmentCode, geomCode;
	ifstream vShaderFile, fShaderFile, gShaderFile;

	vShaderFile.exceptions(ifstream::failbit | ifstream::badbit);
	fShaderFile.exceptions(ifstream::failbit | ifstream::badbit);
	gShaderFile.exceptions(ifstream::failbit | ifstream::badbit);

	try
	{
		stringstream vShaderStream, fShaderStream;

		vShaderFile.open(vertexPath);
		fShaderFile.open(fragmentPath);
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		vShaderFile.close();
		fShaderFile.close();
		vertexCode   = vShaderStream.str();
		fragmentCode = fShaderStream.str();

		if(geomPath != nullptr)
		{
			stringstream gShaderStream;

			gShaderFile.open(geomPath);
			gShaderStream << gShaderFile.rdbuf();
			gShaderFile.close();
			geomCode = gShaderStream.str();
		}
	}
	catch(ifstream::failure e)
	{
		cout << "ERROR: Failed to read the shader file" << endl;
		return false;
	}

	//2. compile shaders
	unsigned int vertex, fragment, geom;

	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");

	if(geomPath != nullptr)
	{
		const char* glShaderCode = geomCode.c_str();
		geom = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geom, 1, &glShaderCode, NULL);
		glCompileShader(geom);
		checkCompileErrors(geom, "GEOMETRY");
	}

	//3. Link shaders
	program = glCreateProgram();
	glAttachShader(program, vertex);
	glAttachShader(program, fragment);

	if(geomPath != nullptr)
		glAttachShader(program, geom);

	glLinkProgram(program);
	checkCompileErrors(program, "PROGRAM");

	//4. Delete shaders
	glDeleteShader(vertex);
	glDeleteShader(fragment);

	if(geomPath != nullptr)
		glDeleteShader(geom);
}


/*-----------------------------
Use this program
-----------------------------*/
void Shader::use()
{
	glUseProgram(program);
}


/*-----------------------------
Set uniform variables of the 
shader
-----------------------------*/
void Shader::setUniform(const char* name, const mat4 &mat)
{
	glUseProgram(program);
	glUniformMatrix4fv(glGetUniformLocation(program, name), 1, GL_FALSE, glm::value_ptr(mat));
}


void Shader::setUniform(const char* name, const mat3 &mat)
{
	glUseProgram(program);
	glUniformMatrix3fv(glGetUniformLocation(program, name), 1, GL_FALSE, glm::value_ptr(mat));
}


void Shader::setUniform(const char* name, const glm::vec3 &v)
{
	glUseProgram(program);
	glUniform3fv(glGetUniformLocation(program, name), 1, glm::value_ptr(v));
}


void Shader::setUniform(const char* name, const glm::vec4 &v)
{
	glUseProgram(program);
	glUniform4fv(glGetUniformLocation(program, name), 1, glm::value_ptr(v));
}


void Shader::setUniform(const char* name, const GLfloat value)
{
	glUseProgram(program);
	glUniform1f(glGetUniformLocation(program, name), value);
}


void Shader::setUniform(const char* name, const GLuint value)
{
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, name), value);
}


void Shader::setUniform(const char* name, const GLint value)
{
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, name), value);
}


void Shader::setUniform(const char* name, const bool value)
{
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, name), value);
}


/*-----------------------------
Set light uniform variables of 
the shader
-----------------------------*/
void Shader::setLightUniform(const char* name, const Light light)
{
	char varName[64];

	glUseProgram(program);

	sprintf(varName, "%s.position", name);
	glUniform3fv(glGetUniformLocation(program, varName), 1, glm::value_ptr(light.position));

	sprintf(varName, "%s.dir", name);
	glUniform3fv(glGetUniformLocation(program, varName), 1, glm::value_ptr(light.direction));

	sprintf(varName, "%s.ambient", name);
	glUniform3fv(glGetUniformLocation(program, varName), 1, glm::value_ptr(light.ambient));

	sprintf(varName, "%s.diffuse", name);
	glUniform3fv(glGetUniformLocation(program, varName), 1, glm::value_ptr(light.diffuse));

	sprintf(varName, "%s.specular", name);
	glUniform3fv(glGetUniformLocation(program, varName), 1, glm::value_ptr(light.specular));

	sprintf(varName, "%s.power", name);
	glUniform1f(glGetUniformLocation(program, varName), light.power);
}