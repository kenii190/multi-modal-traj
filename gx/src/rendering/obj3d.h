#ifndef OBJ3D_H
#define OBJ3D_H

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <vector>
#include "glad/glad.h"
#include "shader.h"
#include "mesh.h"
#include "texture.h"

using namespace std;
using namespace glm;


quat fromToRotation(vec3 start, vec3 dest);
mat4 euler(const vec3 &rot);


class Obj3d
{
public:
	Obj3d();
	Obj3d(Mesh *m, Obj3d *p=nullptr);
	Obj3d(Obj3d *p);
	void setMesh(Mesh *m);
	void addTexture(Texture *t);
	void setParent(Obj3d *p);
	Obj3d* getParent();
	void setPosition(const vec3 &pos);
	vec3 getPosition();
	vec3 getGlobalPosition();
	void setRotation(const quat &rot);
	void setRotation(const vec3 &eulerAngles);
	quat getRotation();
	void setScale(const vec3 &scal);
	vec3 getScale();
	mat4 getModelMat();
	void updateModel();
	void uploadModel(Shader *s);
	void draw(Shader *s, const GLenum mode=GL_TRIANGLES);
	void drawWithoutModel(Shader *s, const GLenum mode=GL_TRIANGLES);
	void drawInstanced(Shader *s, const int amount);

private:
	Obj3d *parent;
	Mesh *mesh;
	vector<Texture*> textures;
	vec3 position;
	quat rotation;
	vec3 scaling;
	mat4 modelMat;
};

#endif