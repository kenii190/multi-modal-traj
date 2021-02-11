#include "mesh.h"
#include <iostream>

using namespace std;
using namespace glm;


/*-----------------------------
Constructor
-----------------------------*/
Mesh::Mesh()
{
	VAO         = 0;
	VBO[0]      = 0;
	VBO[1]      = 0;
	VBO[2]      = 0;
	EBO         = 0;
	numVertices = 0;
	numIndices  = 0;
	vertices    = nullptr;
	normals     = nullptr;
	uvs         = nullptr;
	indices     = nullptr;
	InstanceBuffer = -1;
}


/*-----------------------------
Generate buffers
-----------------------------*/
void Mesh::genBuffer()
{
	glGenVertexArrays(1, &VAO);
	glGenBuffers(3, VBO);
	glGenBuffers(1, &EBO);
}


/*-----------------------------
Destroy buffers
-----------------------------*/
void Mesh::destroy()
{
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(3, VBO);
	glDeleteBuffers(1, &EBO);

	if(InstanceBuffer > 0)
		glDeleteBuffers(1, &InstanceBuffer);
}


/*-----------------------------
Upload vertices
-----------------------------*/
void Mesh::uploadVertices(vector<GLfloat> &v, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	numVertices = v.size()/3;

	glBindVertexArray(VAO);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}


void Mesh::uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	normals     = (GLfloat*)&(n[0]);
	numVertices = v.size()/3;

	glBindVertexArray(VAO);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);

	//VBO 1: normals
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, normals, usage);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_TRUE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);
}


void Mesh::uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, vector<GLfloat> &uv, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	normals     = (GLfloat*)&(n[0]);
	uvs         = (GLfloat*)&(uv[0]);
	numVertices = v.size()/3;

	glBindVertexArray(VAO);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);

	//VBO 1: normals
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, normals, usage);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_TRUE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(1);

	//VBO 2: uvs
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat)*numVertices, uvs, usage);
	glVertexAttribPointer(
		2,
		2,
		GL_FLOAT,
		GL_FALSE,
		2*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
}


void Mesh::uploadVertices(vector<GLfloat> &v, vector<GLuint> &idx, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	indices     = (GLuint*)&(idx[0]);
	numIndices  = idx.size();
	numVertices = v.size()/3;

	glBindVertexArray(VAO);

	//EBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numIndices, indices, usage);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}


void Mesh::uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, vector<GLuint> &idx, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	normals     = (GLfloat*)&(n[0]);
	indices     = (GLuint*)&(idx[0]);
	numIndices  = idx.size();
	numVertices = v.size()/3;

	glBindVertexArray(VAO);

	//EBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numIndices, indices, usage);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);

	//VBO 1: normals
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, normals, usage);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_TRUE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);
}


void Mesh::uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, vector<GLfloat> &uv, vector<GLuint> &idx, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	normals     = (GLfloat*)&(n[0]);
	uvs         = (GLfloat*)&(uv[0]);
	indices     = (GLuint*)&(idx[0]);
	numIndices  = idx.size();
	numVertices = v.size()/3;

	glBindVertexArray(VAO);

	//EBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numIndices, indices, usage);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);

	//VBO 1: normals
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, 3*sizeof(GLfloat)*numVertices, normals, usage);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_TRUE,
		3*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(1);

	//VBO 2: uvs
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat)*numVertices, uvs, usage);
	glVertexAttribPointer(
		2,
		2,
		GL_FLOAT,
		GL_FALSE,
		2*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(2);
	glBindVertexArray(0);
}


/*-----------------------------
Upload 2D vertices
-----------------------------*/
void Mesh::uploadVertices2d(vector<GLfloat> &v, vector<GLfloat> &uv, vector<GLuint> &idx, const GLenum usage)
{
	vertices    = (GLfloat*)&(v[0]);
	uvs         = (GLfloat*)&(uv[0]);
	indices     = (GLuint*)&(idx[0]);
	numIndices  = idx.size();
	numVertices = v.size()/2;

	glBindVertexArray(VAO);

	//EBO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numIndices, indices, usage);

	//VBO 0: vertices
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat)*numVertices, vertices, usage);
	glVertexAttribPointer(
		0,
		2,
		GL_FLOAT,
		GL_FALSE,
		2*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(0);

	//VBO 1: uvs
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat)*numVertices, uvs, usage);
	glVertexAttribPointer(
		1,
		2,
		GL_FLOAT,
		GL_FALSE,
		2*sizeof(GLfloat),
		(GLvoid*)0
	);
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);
}


/*-----------------------------
Upload instance matrices
-----------------------------*/
void Mesh::uploadInstanceMats(vector<mat4> &instanceMats)
{
	GLsizei vec4Size = sizeof(vec4);

	glGenBuffers(1, &InstanceBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, InstanceBuffer);
	glBufferData(GL_ARRAY_BUFFER, instanceMats.size() * sizeof(mat4), &instanceMats[0], GL_STATIC_DRAW);

	glBindVertexArray(VAO);
	
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4*vec4Size, (void*)0);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4*vec4Size, (void*)vec4Size);
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4*vec4Size, (void*)(2*vec4Size));
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 4*vec4Size, (void*)(3*vec4Size));

	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);

	glBindVertexArray(0);
}


/*-----------------------------
Draw the mesh
-----------------------------*/
void Mesh::draw()
{
	glBindVertexArray(VAO);

	if(numIndices > 0) glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
	else glDrawArrays(GL_TRIANGLES, 0, numVertices);
}


void Mesh::draw(GLenum mode)
{
	glBindVertexArray(VAO);

	if(numIndices > 0) glDrawElements(mode, numIndices, GL_UNSIGNED_INT, 0);
	else glDrawArrays(mode, 0, numVertices);
}


/*-----------------------------
Draw the mesh with GPU instancing
-----------------------------*/
void Mesh::drawInstanced(const int amount)
{
	glBindVertexArray(VAO);

	if(numIndices > 0) glDrawElementsInstanced(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0, amount);
	else glDrawArraysInstanced(GL_TRIANGLES, 0, numVertices, amount);
}


/*-----------------------------
Bind the vertex array object
-----------------------------*/
void Mesh::bindVAO()
{
	glBindVertexArray(VAO);
}