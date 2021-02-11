#include "obj2d.h"

using namespace glm;


/*--------------------------
3x3 matrix tranlation
---------------------------*/
static mat3 translate2d(const vec2 &delta)
{
	return mat3(
		1, 0, 0,
		0, 1, 0,
		delta.x, delta.y, 1
	);
}


/*--------------------------
3x3 matrix rotation
---------------------------*/
static mat3 rotate2d(const float angle)
{
	return mat3(
		cos(angle), -sin(angle), 0,
		sin(angle),  cos(angle), 0,
		0,           0,          1
	);
}


/*--------------------------
3x3 matrix scaling
---------------------------*/
static mat3 scale2d(const vec2 &scal)
{
	return mat3(
		scal.x,      0, 0,
		     0, scal.y, 0,
		     0,      0, 1
	);
}


/*--------------------------
Constructor
---------------------------*/
Obj2d::Obj2d()
{
	mesh     = nullptr;
	position = vec2(0.0f, 0.0f);
	scaling  = vec2(1.0f, 1.0f);
	rotationAngle = 0.0f;
}


Obj2d::Obj2d(Mesh *m)
{
	mesh     = m;
	position = vec2(0.0f, 0.0f);
	scaling  = vec2(1.0f, 1.0f);
	rotationAngle = 0.0f;
}


/*--------------------------
Set mesh
---------------------------*/
void Obj2d::setMesh(Mesh *m)
{
	mesh = m;
}


/*--------------------------
Add texture
---------------------------*/
void Obj2d::addTexture(Texture *t)
{
	textures.push_back(t);
}


/*--------------------------
Access position
---------------------------*/
void Obj2d::setPosition(const vec2 &pos)
{
	position = pos;
}


vec2 Obj2d::getPosition()
{
	return position;
}


/*--------------------------
Access rotation
---------------------------*/
void Obj2d::setRotation(const float angle)
{
	rotationAngle = angle;
}


float Obj2d::getRotation()
{
	return rotationAngle;
}


/*--------------------------
Access scale
---------------------------*/
void Obj2d::setScale(const vec2 &scal)
{
	scaling = scal;
}


vec2 Obj2d::getScale()
{
	return scaling;
}


/*--------------------------
Update model matrix
---------------------------*/
void Obj2d::updateModel(Shader *s)
{
	mat3 model = translate2d(position) * scale2d(scaling) * rotate2d(rotationAngle);
	s->setUniform("model", model);
}


/*--------------------------
Draw call
---------------------------*/
void Obj2d::draw(Shader *s, const GLenum mode)
{
	s->use();
	updateModel(s);

	for(int i = 0; i < textures.size(); ++i)
		textures[i]->bind(GL_TEXTURE0+i);

	mesh->draw(mode);
}