#include "obj3d.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

using namespace glm;


/*-----------------------------
Quaternion of from-to rotation
-----------------------------*/
quat fromToRotation(vec3 start, vec3 dest)
{
	start = normalize(start);
	dest  = normalize(dest);
	float cosTheta = dot(start, dest);
	vec3 rotAxis;

	if(cosTheta < -0.999f)
	{
		rotAxis = cross(vec3(0.0f, 0.0f, 1.0f), start);

		if(glm::length(rotAxis) < 1e-2f)
			rotAxis = cross(vec3(1.0f, 0.0f, 0.0f), start);

		return angleAxis(radians(180.0f), normalize(rotAxis));
	}

	rotAxis    = cross(start, dest);
	float s    = sqrt((1 + cosTheta) * 2);
	float invs = 1.0f / s;

	return quat(s * 0.5f, rotAxis.x * invs, rotAxis.y * invs, rotAxis.z * invs);
}


/*-----------------------------
4x4 matrix of euler angle
-----------------------------*/
mat4 euler(const vec3 &rot)
{
	mat4 mX(1, 0, 0, 0,
			0, cos(rot.x), -sin(rot.x), 0,
			0, sin(rot.x), cos(rot.x), 0,
			0, 0, 0, 1);

	mat4 mY(cos(rot.y), 0, -sin(rot.y), 0,
			0, 1, 0, 0,
			sin(rot.y), 0, cos(rot.y), 0,
			0, 0, 0, 1);

	mat4 mZ(cos(rot.z), sin(rot.z), 0, 0,
			-sin(rot.z), cos(rot.z), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

	return mZ * mY * mX;
}


/*-----------------------------
Constructor
-----------------------------*/
Obj3d::Obj3d()
{
	parent   = nullptr;
	mesh     = nullptr;
	position = vec3(0.0f, 0.0f, 0.0f);
	rotation = quat(1.0f, 0.0f, 0.0f, 0.0f);
	scaling  = vec3(1.0f, 1.0f, 1.0f);

	updateModel();
}


Obj3d::Obj3d(Mesh *m, Obj3d *p)
{
	parent   = p;
	mesh     = m;
	position = vec3(0.0f, 0.0f, 0.0f);
	rotation = quat(1.0f, 0.0f, 0.0f, 0.0f);
	scaling  = vec3(1.0f, 1.0f, 1.0f);

	updateModel();
}


Obj3d::Obj3d(Obj3d *p)
{
	parent   = p;
	mesh     = nullptr;
	position = vec3(0.0f, 0.0f, 0.0f);
	rotation = quat(1.0f, 0.0f, 0.0f, 0.0f);
	scaling  = vec3(1.0f, 1.0f, 1.0f);

	updateModel();
}


/*-----------------------------
Set mesh
-----------------------------*/
void Obj3d::setMesh(Mesh *m)
{
	mesh = m;
}


/*-----------------------------
Add texture
-----------------------------*/
void Obj3d::addTexture(Texture *t)
{
	textures.push_back(t);
}


/*-----------------------------
Access parent
-----------------------------*/
void Obj3d::setParent(Obj3d *p)
{
	parent = p;
}


Obj3d* Obj3d::getParent()
{
	return parent;
}


/*-----------------------------
Access position
-----------------------------*/
void Obj3d::setPosition(const vec3 &pos)
{
	position = pos;
	updateModel();
}


vec3 Obj3d::getPosition()
{
	return position;
}


vec3 Obj3d::getGlobalPosition()
{
	vec4 pos = vec4(position, 1.0);
	Obj3d *p = parent;

	while(p != nullptr)
	{
		pos = p->getModelMat() * pos;
		p   = p->getParent();
	}

	return vec3(pos);
}


/*-----------------------------
Access rotation
-----------------------------*/
void Obj3d::setRotation(const quat &rot)
{
	rotation = rot;
	updateModel();
}


void Obj3d::setRotation(const vec3 &eulerAngles)
{
	quat quatX = angleAxis(eulerAngles.x, vec3(1.0f, 0.0f, 0.0f));
	quat quatY = angleAxis(eulerAngles.y, vec3(0.0f, 1.0f, 0.0f));
	quat quatZ = angleAxis(eulerAngles.z, vec3(0.0f, 0.0f, 1.0f));
	rotation = quatX * quatY * quatZ;
	updateModel();
}


quat Obj3d::getRotation()
{
	return rotation;
}


/*-----------------------------
Access scaling
-----------------------------*/
void Obj3d::setScale(const vec3 &scal)
{
	scaling = scal;
	updateModel();
}


vec3 Obj3d::getScale()
{
	return scaling;
}


/*-----------------------------
Access model matrix
-----------------------------*/
mat4 Obj3d::getModelMat()
{
	return modelMat;
}


/*-----------------------------
Update model matrix
-----------------------------*/
void Obj3d::updateModel()
{
	modelMat = translate(mat4(1.0f), position) * mat4_cast(rotation) * scale(mat4(1.0f), scaling);
}


/*-----------------------------
Upload model matrix
-----------------------------*/
void Obj3d::uploadModel(Shader *s)
{
	mat4 m   = modelMat;
	Obj3d* p = parent;

	while(p != nullptr)
	{
		m = p->getModelMat() * m;
		p = p->getParent();
	}

	s->setUniform("model", m);
}


/*-----------------------------
Draw call
-----------------------------*/
void Obj3d::draw(Shader *s, const GLenum mode)
{
	s->use();
	uploadModel(s);

	for(int i = 0; i < textures.size(); ++i)
		textures[i]->bind(GL_TEXTURE0+i);

	mesh->draw(mode);
}


/*-----------------------------
Draw call without updating model
-----------------------------*/
void Obj3d::drawWithoutModel(Shader *s, const GLenum mode)
{
	s->use();

	for(int i = 0; i < textures.size(); ++i)
		textures[i]->bind(GL_TEXTURE0+i);

	mesh->draw(mode);
}


/*-----------------------------
Draw call with GPU instancing
-----------------------------*/
void Obj3d::drawInstanced(Shader *s, const int amount)
{
	s->use();

	for(int i = 0; i < textures.size(); ++i)
		textures[i]->bind(GL_TEXTURE0+i);

	mesh->drawInstanced(amount);
}