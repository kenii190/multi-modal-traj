#ifndef MESH_H
#define MESH_H

#include "glad/glad.h"
#include <glm/glm.hpp>
#include <vector>

using namespace std;
using namespace glm;


class Mesh
{
public:
	Mesh();
	void genBuffer();
	void destroy();
	void uploadVertices(vector<GLfloat> &v, const GLenum usage=GL_STATIC_DRAW);
	void uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, const GLenum usage=GL_STATIC_DRAW);
	void uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, vector<GLfloat> &uv, const GLenum usage=GL_STATIC_DRAW);
	void uploadVertices(vector<GLfloat> &v, vector<GLuint> &idx, const GLenum usage=GL_STATIC_DRAW);
	void uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, vector<GLuint> &idx, const GLenum usage=GL_STATIC_DRAW);
	void uploadVertices(vector<GLfloat> &v, vector<GLfloat> &n, vector<GLfloat> &uv, vector<GLuint> &idx, const GLenum usage=GL_STATIC_DRAW);
	void uploadVertices2d(vector<GLfloat> &v, vector<GLfloat> &uv, vector<GLuint> &idx, const GLenum usage=GL_STATIC_DRAW);
	void uploadInstanceMats(vector<mat4> &instanceMats);
	void draw();
	void draw(GLenum mode);
	void drawInstanced(const int amount);
	void bindVAO();

private:
	GLfloat *vertices;
	GLfloat *normals;
	GLfloat *uvs;
	GLuint *indices;
	size_t numVertices;
	size_t numIndices;
	GLuint VAO;
	GLuint VBO[3];
	GLuint EBO;
	GLuint InstanceBuffer;
};

#endif