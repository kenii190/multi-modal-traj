#ifndef POSE_H
#define POSE_H

#include <vector>
#include <string>
#include <map>
#include <glm/glm.hpp>
#include "obj3d.h"

using namespace std;
using namespace glm;


struct Bone
{
	vec3 direction;
	vec3 axis;
	vec3 dof;
	float length;

	Bone(): direction(vec3(0.0f)), axis(vec3(0.0f)), dof(vec3(0.0f)), length(1.0f){}
};


struct Joint
{
	Joint *parent;
	vector<Joint*> children;
	Obj3d obj;
	vec3 dof;
	vec3 axisX;
	vec3 axisY;
	vec3 axisZ;
	vec3 direction;
	quat rot;
	float length;

	Joint(Joint *p, Mesh *m)
	{
		parent = p;
		
		if(p != nullptr)
			obj.setParent(&(p->obj));

		obj.setMesh(m);
		dof   = vec3(0.0f);
		axisX = vec3(1.0f, 0.0f, 0.0f);
		axisY = vec3(0.0f, 1.0f, 0.0f);
		axisZ = vec3(0.0f, 0.0f, 1.0f);
		rot   = quat(1.0f, 0.0f, 0.0f, 0.0f);
		direction = vec3(0.0, 1.0, 0.0);
		length = 1.0f;
	}
};


class Pose
{
public:
	Pose(Mesh *meshJoint, Mesh *meshLimb, float limbLen=1.0f);
	void loadFromAsfFile(const char* filename);
	void buildHierarchy(map<string, Bone> &boneMap, vector< vector<string> > &hierarchy);
	void buildHierarchyByAMASS();
	void apply(map<string, vector<float> > &rotMap);
	void applyAMASS(vector<float> posVec);
	void setRootPosition(vec3 &pos);
	void setRootRotation(const vec3 &eulerAngles);
	void contactGround(const float val);
	void drawJoint(Joint *joint, Shader *shaderJoint, Shader *shaderLimb);
	void draw(Shader *shaderJoint, Shader *shaderLimb);

private:
	map<string, Joint*> jointMap;
	Joint *root;
	Obj3d objLimb;
	Mesh *meshJoint;
	Mesh *meshLimb;
	float limbLen;
};

#endif