#ifndef _OBJ2D_H_
#define _OBJ2D_H_

#include <glm/glm.hpp>
#include <vector>
#include "glad/glad.h"
#include "shader.h"
#include "mesh.h"
#include "texture.h"

using namespace std;
using namespace glm;


class Obj2d
{
public:
	Obj2d();
	Obj2d(Mesh *m);
	void setMesh(Mesh *m);
	void addTexture(Texture *t);
	void setPosition(const vec2 &pos);
	vec2 getPosition();
	void setRotation(const float angle);
	float getRotation();
	void setScale(const vec2 &scal);
	vec2 getScale();
	void updateModel(Shader *s);
	void draw(Shader *s, const GLenum mode=GL_TRIANGLES);

private:
	Mesh *mesh;
	vector<Texture*> textures;
	vec2 position;
	vec2 scaling;
	float rotationAngle;
};

#endif