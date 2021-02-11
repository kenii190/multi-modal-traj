#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <glm/gtx/transform.hpp>
#include "pose.h"

#define ANGLE_TO_RAD 0.01745329251f


/*-----------------------
Split the string to vector
-----------------------*/
static vector<string> split(const string &str, const char c)
{
	vector<string> strs;
	istringstream iss(str);
	string s;

	while(getline(iss, s, c))
		if(s.compare("") != 0)
			strs.push_back(s);

	return strs;
}


/*-----------------------
Constructor
-----------------------*/
Pose::Pose(Mesh *meshJoint, Mesh *meshLimb, float limbLen)
{
	this->objLimb.setMesh(meshLimb);
	this->meshJoint = meshJoint;
	this->meshLimb  = meshLimb;
	this->limbLen   = limbLen;
	this->root      = new Joint(nullptr, meshJoint);

	root->obj.setScale(vec3(0.8f));
}


/*-----------------------
Load skeleton from *.asf file
-----------------------*/
void Pose::loadFromAsfFile(const char* filename)
{
	map<string, Bone> boneMap;
	vector< vector<string> > hierarchy;
	vector<string> lines;
	
	//1. Read the file
	ifstream ifs(filename, ifstream::in);
	string line;

	if(ifs.fail())
	{
		cout << "Failed to open the file: " << filename << endl;
		return;
	}

	while(getline(ifs, line))
		lines.push_back(line);

	ifs.close();

	//2. Parse the data
	int maxIdx = lines.size() - 1;
	int idx = 0;

	while(idx < lines.size())
	{
		//Root data
		if(lines[idx].compare(":root") == 0)
		{
			vector<string> rootOrder  = split(lines[++idx], ' ');
			vector<string> rootAxis   = split(lines[++idx], ' ');
			vector<string> rootPos    = split(lines[++idx], ' ');
			vector<string> rootOrient = split(lines[++idx], ' ');
		}
		//Bone data
		else if(lines[idx].compare(":bonedata") == 0)
		{
			vector<string> items = split(lines[idx+1], ' ');

			while(items.size() == 0 || (items[0].compare("begin") == 0 && idx < maxIdx))
			{
				idx++;
				string name = "";
				Bone bone;

				while(items.size() == 0 || idx < maxIdx-1)
				{
					items = split(lines[++idx], ' ');

					if(items[0].compare("name") == 0)
						name = items[1];
					else if(items[0].compare("direction") == 0)
						bone.direction = vec3(stof(items[1]), stof(items[2]), stof(items[3]));
					else if(items[0].compare("length") == 0)
						bone.length = stof(items[1]) * limbLen;
					else if(items[0].compare("axis") == 0)
						bone.axis = vec3(stof(items[1]) * ANGLE_TO_RAD, stof(items[2]) * ANGLE_TO_RAD, stof(items[3]) * ANGLE_TO_RAD);
					else if(items[0].compare("dof") == 0)
						for(int i = 1; i < items.size(); ++i)
						{
							if(items[i].compare("rx") == 0) bone.dof.x = 1;
							else if(items[i].compare("ry") == 0) bone.dof.y = 1;
							else if(items[i].compare("rz") == 0) bone.dof.z = 1;
						}
					else if(items[0].compare("end") == 0)
						break;
				}
				boneMap[name] = bone;

				if(idx < maxIdx - 1)
					items = split(lines[idx+1], ' ');
			}
		}
		//Hierarchy
		else if(lines[idx].compare(":hierarchy") == 0)
		{
			vector<string> items = split(lines[idx+1], ' ');

			while(items.size() == 0 || (items[0].compare("begin") == 0 && idx < maxIdx))
			{
				idx++;

				while(items.size() == 0 || idx < maxIdx-1)
				{
					items = split(lines[++idx], ' ');

					if(items[0].compare("end") == 0)
						break;
					else
						hierarchy.push_back(items);
				}

				if(idx < maxIdx - 1)
					items = split(lines[idx+1], ' ');
			}
		}
		idx++;
	}

	//3. Build hierarchy
	buildHierarchy(boneMap, hierarchy);
}


/*-----------------------
Build hierarchy
-----------------------*/
void Pose::buildHierarchy(map<string, Bone> &boneMap, vector< vector<string> > &hierarchy)
{
	jointMap["root"] = this->root;
	jointMap["root"]->dof = vec3(1.0f);

	for(int i = 0; i < hierarchy.size(); ++i)
	{
		Joint *parentJoint = jointMap[hierarchy[i][0]];

		for(int j = 1; j < hierarchy[i].size(); ++j)
		{
			Joint *joint = new Joint(parentJoint, meshJoint);
			Bone bone    = boneMap[hierarchy[i][j]];

			vec3 rotAxis = bone.axis;
			mat3 rx = mat3(glm::rotate(mat4(1.0), rotAxis.x, vec3(1.0, 0.0, 0.0)));
			mat3 ry = mat3(glm::rotate(mat4(1.0), rotAxis.y, vec3(0.0, 1.0, 0.0)));
			mat3 rz = mat3(glm::rotate(mat4(1.0), rotAxis.z, vec3(0.0, 0.0, 1.0)));
			mat3 r  = rz * ry * rx;

			joint->axisX     = r * vec3(1.0, 0.0, 0.0);
			joint->axisY     = r * vec3(0.0, 1.0, 0.0);
			joint->axisZ     = r * vec3(0.0, 0.0, 1.0);
			joint->dof       = bone.dof;
			joint->direction = bone.direction;
			joint->length    = bone.length;
			joint->obj.setPosition(joint->direction * joint->length);

			jointMap[hierarchy[i][j]] = joint;
			parentJoint->children.push_back(joint);
		}
	}
}


void Pose::buildHierarchyByAMASS()
{
	jointMap["root"]        = this->root;
	jointMap["root"]->axisX = vec3(1.0, 0.0, 0.0);
	jointMap["root"]->axisY = vec3(0.0, 1.0, 0.0);
	jointMap["root"]->axisZ = vec3(0.0, 0.0, 1.0);
	jointMap["root"]->dof   = vec3(1.0f);

	jointMap["1"]            = new Joint(jointMap["root"], meshJoint);
	jointMap["1"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["1"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["1"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["1"]->direction = vec3(0.7071, -0.7071, 0.0);
	jointMap["1"]->dof       = vec3(1.0f);
	jointMap["1"]->length    = 3.0f;
	jointMap["1"]->obj.setPosition(jointMap["1"]->direction * jointMap["1"]->length);

	jointMap["2"]            = new Joint(jointMap["root"], meshJoint);
	jointMap["2"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["2"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["2"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["2"]->direction = vec3(-0.7071, -0.7071, 0.0);
	jointMap["2"]->dof       = vec3(1.0f);
	jointMap["2"]->length    = 3.0f;
	jointMap["2"]->obj.setPosition(jointMap["2"]->direction * jointMap["2"]->length);

	jointMap["3"]            = new Joint(jointMap["root"], meshJoint);
	jointMap["3"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["3"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["3"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["3"]->direction = vec3(0.0, 1.0, 0.0);
	jointMap["3"]->dof       = vec3(1.0f);
	jointMap["3"]->length    = 3.0f;
	jointMap["3"]->obj.setPosition(jointMap["3"]->direction * jointMap["3"]->length);

	jointMap["4"]            = new Joint(jointMap["1"], meshJoint);
	jointMap["4"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["4"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["4"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["4"]->direction = vec3(0.0, -1.0, 0.0);
	jointMap["4"]->dof       = vec3(1.0f);
	jointMap["4"]->length    = 9.0f;
	jointMap["4"]->obj.setPosition(jointMap["4"]->direction * jointMap["4"]->length);

	jointMap["5"]            = new Joint(jointMap["2"], meshJoint);
	jointMap["5"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["5"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["5"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["5"]->direction = vec3(0.0, -1.0, 0.0);
	jointMap["5"]->dof       = vec3(1.0f);
	jointMap["5"]->length    = 9.0f;
	jointMap["5"]->obj.setPosition(jointMap["5"]->direction * jointMap["5"]->length);

	jointMap["6"]            = new Joint(jointMap["3"], meshJoint);
	jointMap["6"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["6"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["6"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["6"]->direction = vec3(0.0, 1.0, 0.0);
	jointMap["6"]->dof       = vec3(1.0f);
	jointMap["6"]->length    = 3.0f;
	jointMap["6"]->obj.setPosition(jointMap["6"]->direction * jointMap["6"]->length);

	jointMap["7"]            = new Joint(jointMap["4"], meshJoint);
	jointMap["7"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["7"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["7"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["7"]->direction = vec3(0.0, -1.0, 0.0);
	jointMap["7"]->dof       = vec3(1.0f);
	jointMap["7"]->length    = 9.0f;
	jointMap["7"]->obj.setPosition(jointMap["7"]->direction * jointMap["7"]->length);

	jointMap["8"]            = new Joint(jointMap["5"], meshJoint);
	jointMap["8"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["8"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["8"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["8"]->direction = vec3(0.0, -1.0, 0.0);
	jointMap["8"]->dof       = vec3(1.0f);
	jointMap["8"]->length    = 9.0f;
	jointMap["8"]->obj.setPosition(jointMap["8"]->direction * jointMap["8"]->length);

	jointMap["9"]            = new Joint(jointMap["6"], meshJoint);
	jointMap["9"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["9"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["9"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["9"]->direction = vec3(0.0, 1.0, 0.0);
	jointMap["9"]->dof       = vec3(1.0f);
	jointMap["9"]->length    = 1.0f;
	jointMap["9"]->obj.setPosition(jointMap["9"]->direction * jointMap["9"]->length);

	jointMap["10"]            = new Joint(jointMap["7"], meshJoint);
	jointMap["10"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["10"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["10"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["10"]->direction = vec3(0.0, 0.0, 1.0);
	jointMap["10"]->dof       = vec3(1.0f);
	jointMap["10"]->length    = 3.0f;
	jointMap["10"]->obj.setPosition(jointMap["10"]->direction * jointMap["10"]->length);

	jointMap["11"]            = new Joint(jointMap["8"], meshJoint);
	jointMap["11"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["11"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["11"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["11"]->direction = vec3(0.0, 0.0, 1.0);
	jointMap["11"]->dof       = vec3(1.0f);
	jointMap["11"]->length    = 3.0f;
	jointMap["11"]->obj.setPosition(jointMap["11"]->direction * jointMap["11"]->length);

	jointMap["12"]            = new Joint(jointMap["9"], meshJoint);
	jointMap["12"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["12"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["12"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["12"]->direction = vec3(0.0, 1.0, 0.0);
	jointMap["12"]->dof       = vec3(1.0f);
	jointMap["12"]->length    = 4.0f;
	jointMap["12"]->obj.setPosition(jointMap["12"]->direction * jointMap["12"]->length);

	jointMap["13"]            = new Joint(jointMap["12"], meshJoint);
	jointMap["13"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["13"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["13"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["13"]->direction = vec3(0.7071, -0.7071, 0.0);
	jointMap["13"]->dof       = vec3(1.0f);
	jointMap["13"]->length    = 3.0f;
	jointMap["13"]->obj.setPosition(jointMap["13"]->direction * jointMap["13"]->length);

	jointMap["14"]            = new Joint(jointMap["12"], meshJoint);
	jointMap["14"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["14"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["14"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["14"]->direction = vec3(-0.7071, -0.7071, 0.0);
	jointMap["14"]->dof       = vec3(1.0f);
	jointMap["14"]->length    = 3.0f;
	jointMap["14"]->obj.setPosition(jointMap["14"]->direction * jointMap["14"]->length);

	jointMap["15"]            = new Joint(jointMap["12"], meshJoint);
	jointMap["15"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["15"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["15"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["15"]->direction = vec3(0.0, 1.0, 0.0);
	jointMap["15"]->dof       = vec3(1.0f);
	jointMap["15"]->length    = 3.0f;
	jointMap["15"]->obj.setPosition(jointMap["15"]->direction * jointMap["15"]->length);

	jointMap["16"]            = new Joint(jointMap["13"], meshJoint);
	jointMap["16"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["16"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["16"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["16"]->direction = vec3(0.866025, 0.5, 0.0);
	jointMap["16"]->dof       = vec3(1.0f);
	jointMap["16"]->length    = 3.0f;
	jointMap["16"]->obj.setPosition(jointMap["16"]->direction * jointMap["16"]->length);

	jointMap["17"]            = new Joint(jointMap["14"], meshJoint);
	jointMap["17"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["17"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["17"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["17"]->direction = vec3(-0.866025, 0.5, 0.0);
	jointMap["17"]->dof       = vec3(1.0f);
	jointMap["17"]->length    = 3.0f;
	jointMap["17"]->obj.setPosition(jointMap["17"]->direction * jointMap["17"]->length);

	jointMap["18"]            = new Joint(jointMap["16"], meshJoint);
	jointMap["18"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["18"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["18"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["18"]->direction = vec3(1.0, 0.0, 0.0);
	jointMap["18"]->dof       = vec3(1.0f);
	jointMap["18"]->length    = 6.0f;
	jointMap["18"]->obj.setPosition(jointMap["18"]->direction * jointMap["18"]->length);

	jointMap["19"]            = new Joint(jointMap["17"], meshJoint);
	jointMap["19"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["19"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["19"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["19"]->direction = vec3(-1.0, 0.0, 0.0);
	jointMap["19"]->dof       = vec3(1.0f);
	jointMap["19"]->length    = 6.0f;
	jointMap["19"]->obj.setPosition(jointMap["19"]->direction * jointMap["19"]->length);

	jointMap["20"]            = new Joint(jointMap["18"], meshJoint);
	jointMap["20"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["20"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["20"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["20"]->direction = vec3(1.0, 0.0, 0.0);
	jointMap["20"]->dof       = vec3(1.0f);
	jointMap["20"]->length    = 6.0f;
	jointMap["20"]->obj.setPosition(jointMap["20"]->direction * jointMap["20"]->length);

	jointMap["21"]            = new Joint(jointMap["19"], meshJoint);
	jointMap["21"]->axisX     = vec3(1.0, 0.0, 0.0);
	jointMap["21"]->axisY     = vec3(0.0, 1.0, 0.0);
	jointMap["21"]->axisZ     = vec3(0.0, 0.0, 1.0);
	jointMap["21"]->direction = vec3(-1.0, 0.0, 0.0);
	jointMap["21"]->dof       = vec3(1.0f);
	jointMap["21"]->length    = 6.0f;
	jointMap["21"]->obj.setPosition(jointMap["21"]->direction * jointMap["21"]->length);

	//Hierarchy map
	jointMap["root"]->children.push_back(jointMap["1"]);
	jointMap["root"]->children.push_back(jointMap["2"]);
	jointMap["root"]->children.push_back(jointMap["3"]);
	jointMap["1"]->children.push_back(jointMap["4"]);
	jointMap["2"]->children.push_back(jointMap["5"]);
	jointMap["3"]->children.push_back(jointMap["6"]);
	jointMap["4"]->children.push_back(jointMap["7"]);
	jointMap["5"]->children.push_back(jointMap["8"]);
	jointMap["6"]->children.push_back(jointMap["9"]);
	jointMap["7"]->children.push_back(jointMap["10"]);
	jointMap["8"]->children.push_back(jointMap["11"]);
	jointMap["9"]->children.push_back(jointMap["12"]);
	jointMap["12"]->children.push_back(jointMap["13"]);
	jointMap["12"]->children.push_back(jointMap["14"]);
	jointMap["12"]->children.push_back(jointMap["15"]);
	jointMap["13"]->children.push_back(jointMap["16"]);
	jointMap["14"]->children.push_back(jointMap["17"]);
	jointMap["16"]->children.push_back(jointMap["18"]);
	jointMap["17"]->children.push_back(jointMap["19"]);
	jointMap["18"]->children.push_back(jointMap["20"]);
	jointMap["19"]->children.push_back(jointMap["21"]);
}


/*-----------------------
Apply pose
-----------------------*/
void Pose::apply(map<string, vector<float> > &rotMap)
{
	for(map<string, vector<float> >::iterator it = rotMap.begin(); it != rotMap.end(); ++it)
	{
		string name = it->first;
		Joint *joint = jointMap[name];
		vector<float> rotAngle = rotMap[name];
		vec3 angle(0.0);

		if(name.compare("root") == 0)
		{
			angle.x = rotAngle[3] * ANGLE_TO_RAD;
			angle.y = rotAngle[4] * ANGLE_TO_RAD;
			angle.z = rotAngle[5] * ANGLE_TO_RAD;

			mat4 rx = rotate(mat4(1.0), angle.x, root->axisX);
			mat4 ry = rotate(mat4(1.0), angle.y, root->axisY);
			mat4 rz = rotate(mat4(1.0), angle.z, root->axisZ);
			root->rot = quat_cast(rz * ry * rx);
		}
		else
		{
			int i = 0;

			if(joint->dof.x == 1)
				angle.x = rotAngle[i++] * ANGLE_TO_RAD;

			if(joint->dof.y == 1)
				angle.y = rotAngle[i++] * ANGLE_TO_RAD;

			if(joint->dof.z == 1)
				angle.z = rotAngle[i++] * ANGLE_TO_RAD;
		}

		mat4 rx = rotate(mat4(1.0), angle.x, joint->axisX);
		mat4 ry = rotate(mat4(1.0), angle.y, joint->axisY);
		mat4 rz = rotate(mat4(1.0), angle.z, joint->axisZ);
		joint->rot = quat_cast(rz * ry * rx);
	}
}


void Pose::applyAMASS(vector<float> posVec)
{
	static mat4 ident = mat4(1.0);
	static vector<string> names{
		"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", 
		"12", "13", "14", "15", "16", "17", "18", "19", "20", "21"
	};

	for(int i = 0; i < 21; i++)
	{
		Joint* joint = jointMap[names[i]];

		joint->rot = quat_cast(
			rotate(ident, posVec[i*3+2], joint->axisZ)
		  * rotate(ident, posVec[i*3+1], joint->axisY)
		  * rotate(ident, posVec[i*3  ], joint->axisX)
		);
	}
}


/*-----------------------
Set root position
-----------------------*/
void Pose::setRootPosition(vec3 &pos)
{
	root->obj.setPosition(pos);
}


/*-----------------------
Set root rotation
-----------------------*/
void Pose::setRootRotation(const vec3 &eulerAngles)
{
	root->rot = quat_cast(
		rotate(mat4(1.0), eulerAngles.z, root->axisZ)
	  * rotate(mat4(1.0), eulerAngles.y, root->axisY)
	  * rotate(mat4(1.0), eulerAngles.x, root->axisX)
	);
}


/*-----------------------
Let the skeleton contact the ground
-----------------------*/
void Pose::contactGround(const float val)
{
	float minY = INFINITY;

	for(map<string, Joint*>::iterator it = jointMap.begin(); it != jointMap.end(); ++it)
	{
		float y = it->second->obj.getGlobalPosition().y;
		if(y < minY) minY = y;
	}

	root->obj.setPosition(root->obj.getPosition() + vec3(0.0f, val - minY + 0.5f, 0.0f));
}


/*-----------------------
Draw joints recursively
-----------------------*/
void Pose::drawJoint(Joint *joint, Shader *shaderJoint, Shader *shaderLimb)
{
	//Draw joint
	/*
	if(joint->parent == nullptr)
		joint->obj.setRotation(joint->rot);
	else
		joint->parent->obj.setRotation(joint->rot);
	*/
	joint->obj.setRotation(joint->rot);
	joint->obj.draw(shaderJoint);

	//Draw limb
	Obj3d *parentObj = joint->obj.getParent();
	if(parentObj != nullptr)
	{
		vec3 p1 = parentObj->getGlobalPosition();
		vec3 p2 = joint->obj.getGlobalPosition();
		vec3 limbVec = p2 - p1;

		objLimb.setPosition((p1+p2) / 2.0f);
		objLimb.setScale(vec3(0.6f, glm::length(limbVec) / 2.0f, 0.6f));
		objLimb.setRotation(fromToRotation(vec3(0.0f, 1.0f, 0.0f), limbVec));
		objLimb.draw(shaderLimb);
	}

	//Draw child joints
	for(int i = 0; i < joint->children.size(); ++i)
		drawJoint(joint->children[i], shaderJoint, shaderLimb);
}


/*-----------------------
Draw the pose
-----------------------*/
void Pose::draw(Shader *shaderJoint, Shader *shaderLimb)
{
	drawJoint(root, shaderJoint, shaderLimb);
}