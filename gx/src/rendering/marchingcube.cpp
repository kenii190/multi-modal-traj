#include <cmath>
#include <iostream>
#include "marchingcube.h"
#include "mctable.hpp"

using namespace glm;


/*----------------------------
Constructor
----------------------------*/
MarchingCube::MarchingCube()
{
	resol = 32;
	voxelLen = 0.5f;
	voxelData = new voxel[resol*resol*resol];

	for(int x = 0; x < resol; ++x)
		for(int y = 0; y < resol; ++y)
			for(int z = 0; z < resol; ++z)
			{
				voxelData[INDEX(x, y, z, resol)].active = false;	
			}
}


MarchingCube::MarchingCube(int resol, float voxelLen)
{
	this->resol = resol;
	this->voxelLen = voxelLen;
	voxelData = new voxel[resol*resol*resol];

	for(int x = 0; x < resol; ++x)
		for(int y = 0; y < resol; ++y)
			for(int z = 0; z < resol; ++z)
			{
				voxelData[INDEX(x, y, z, resol)].active = false;	
			}
}


/*----------------------------
Destructor
----------------------------*/
MarchingCube::~MarchingCube()
{
	delete voxelData;
}


/*----------------------------
Process point cloud
----------------------------*/
void MarchingCube::processPointCloud(const GLfloat vertex[], const int vertexNum, const vec3 trans)
{
	for(int i = 0; i < vertexNum*3; i+=3)
	{
		int x = round((vertex[i  ] + trans.x) / voxelLen);
		int y = round((vertex[i+1] + trans.y) / voxelLen);
		int z = round((vertex[i+2] + trans.z) / voxelLen);

		if(x >= 0 && x < resol && y >= 0 && y < resol && z >= 0 && z < resol)
			voxelData[INDEX(x, y, z, resol)].active = true;
	}
}


void MarchingCube::processPointCloud(const std::vector<GLfloat> vertex, const vec3 trans)
{
	for(int i = 0; i < vertex.size(); i+=3)
	{
		int x = round((vertex[i  ] + trans.x) / voxelLen);
		int y = round((vertex[i+1] + trans.y) / voxelLen);
		int z = round((vertex[i+2] + trans.z) / voxelLen);

		if(x >= 0 && x < resol && y >= 0 && y < resol && z >= 0 && z < resol)
			voxelData[INDEX(x, y, z, resol)].active = true;
	}
}


/*----------------------------
Activate a voxel
----------------------------*/
void MarchingCube::activate(const vec3 pos, const bool act)
{
	int x = round(pos.x / voxelLen);
	int y = round(pos.y / voxelLen);
	int z = round(pos.z / voxelLen);

	if(x >= 0 && x < resol && y >= 0 && y < resol && z >= 0 && z < resol)
		voxelData[INDEX(x, y, z, resol)].active = act;
}


void MarchingCube::activate(const vec3 pos, const bool act, int radius)
{
	int x = round(pos.x / voxelLen);
	int y = round(pos.y / voxelLen);
	int z = round(pos.z / voxelLen);

	for(int i = -radius; i <= radius; ++i)
		for(int j = -radius; j <= radius; ++j)
			for(int k = -radius; k <= radius; ++k)
			{
				int xx = x + i;
				int yy = y + j;
				int zz = z + k;

				if(xx >= 0 && xx < resol && yy >= 0 && yy < resol && zz >= 0 && zz < resol)
					voxelData[INDEX(xx, yy, zz, resol)].active = act;
			}
}


/*----------------------------
Get the middle point from the 
voxel
----------------------------*/
vec3 MarchingCube::getMidPoint(const int x, const int y, const int z, const int code)
{
	switch(code)
	{
		case 0:  return vec3(x+1.0, y    , z+0.5) * voxelLen;
		case 1:  return vec3(x+0.5, y    , z+1.0) * voxelLen;
		case 2:  return vec3(x    , y    , z+0.5) * voxelLen;
		case 3:  return vec3(x+0.5, y    , z    ) * voxelLen;
		case 4:  return vec3(x+1.0, y+0.5, z    ) * voxelLen;
		case 5:  return vec3(x+1.0, y+0.5, z+1.0) * voxelLen;
		case 6:  return vec3(x    , y+0.5, z    ) * voxelLen;
		case 7:  return vec3(x    , y+0.5, z+1.0) * voxelLen;
		case 8:  return vec3(x+1.0, y+1.0, z+0.5) * voxelLen;
		case 9:  return vec3(x+0.5, y+1.0, z+1.0) * voxelLen;
		case 10: return vec3(x    , y+1.0, z+0.5) * voxelLen;
		case 11: return vec3(x+0.5, y+1.0, z    ) * voxelLen;
	}

	return vec3(0, 0, 0);
}


/*----------------------------
Get Triangle normal
----------------------------*/
vec3 MarchingCube::getNormal(vec3 u, vec3 v)
{
	return normalize(vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x));
}


/*----------------------------
Reconstruct surface
----------------------------*/
void MarchingCube::reconstruct(std::vector<vec3> &vertexData, std::vector<vec3> &normalData, std::vector<GLuint> &idxData)
{
	int idxCounter = 0;

	for(int x = 0; x < resol-1; ++x)
		for(int y = 0; y < resol-1; ++y)
			for(int z = 0; z < resol-1; ++z)
			{
				int idx = 0;

				if(voxelData[INDEX(x+1, y  , z  , resol)].active) idx |= 1;
				if(voxelData[INDEX(x+1, y  , z+1, resol)].active) idx |= 2;
				if(voxelData[INDEX(x  , y  , z  , resol)].active) idx |= 4;
				if(voxelData[INDEX(x  , y  , z+1, resol)].active) idx |= 8;
				if(voxelData[INDEX(x+1, y+1, z  , resol)].active) idx |= 16;
				if(voxelData[INDEX(x+1, y+1, z+1, resol)].active) idx |= 32;
				if(voxelData[INDEX(x  , y+1, z  , resol)].active) idx |= 64;
				if(voxelData[INDEX(x  , y+1, z+1, resol)].active) idx |= 128;

				if(idx != 0)
				{
					for(int i = 0; i < mctable[idx][0]; ++i)
					{
						vec3 p1 = getMidPoint(x, y, z, mctable[idx][i*3+1]);
						vec3 p2 = getMidPoint(x, y, z, mctable[idx][i*3+2]);
						vec3 p3 = getMidPoint(x, y, z, mctable[idx][i*3+3]);
						vec3 n = getNormal(p2 - p1, p3 - p1);

						vertexData.push_back(p1);
						vertexData.push_back(p2);
						vertexData.push_back(p3);

						normalData.push_back(n);
						normalData.push_back(n);
						normalData.push_back(n);

						idxData.push_back(idxCounter++);
						idxData.push_back(idxCounter++);
						idxData.push_back(idxCounter++);
					}
				}
			}
}


/*----------------------------
Get voxel position
----------------------------*/
GLfloat* MarchingCube::getVoxelPos()
{
	GLfloat *vertexData = new GLfloat[resol*resol*resol*3];

	for(int x = 0; x < resol; ++x)
		for(int y = 0; y < resol; ++y)
			for(int z = 0; z < resol; ++z)
			{
				vertexData[INDEX(x, y, z, resol)*3    ] = x * voxelLen;
				vertexData[INDEX(x, y, z, resol)*3 + 1] = y * voxelLen;
				vertexData[INDEX(x, y, z, resol)*3 + 2] = z * voxelLen;
			}

	return vertexData;
}


std::vector<vec3> MarchingCube::getVoxelPos(bool state)
{
	std::vector<vec3> vertexData;

	for(int x = 0; x < resol; ++x)
		for(int y = 0; y < resol; ++y)
			for(int z = 0; z < resol; ++z)
			{
				if(voxelData[INDEX(x, y, z, resol)].active == state)
					vertexData.push_back(vec3(x * voxelLen, y * voxelLen, z * voxelLen));
			}

	return vertexData;
}


/*----------------------------
Get voxel resolution
----------------------------*/
int MarchingCube::getResol()
{
	return resol;
}


/*----------------------------
Get max length of the volume
----------------------------*/
float MarchingCube::getMaxLen()
{
	return resol * voxelLen;
}