#ifndef _MARCHING_CUBE_H_
#define _MARCHING_CUBE_H_

#include "glad/glad.h"
#include <glm/glm.hpp>
#include <vector>

#define INDEX(x, y, z, resol) ((x)*(resol)*(resol) + (y)*(resol) + (z))


typedef struct{
	bool active;

}voxel; 


class MarchingCube
{
public:
	MarchingCube();
	MarchingCube(int resol, float voxelLen);
	~MarchingCube();
	void processPointCloud(const GLfloat vertex[], const int vertexNum, const glm::vec3 trans);
	void processPointCloud(const std::vector<GLfloat> vertex, const glm::vec3 trans);
	void activate(const glm::vec3 pos, const bool act);
	void activate(const glm::vec3 pos, const bool act, int radius);
	void reconstruct(std::vector<glm::vec3> &vertexData, std::vector<glm::vec3> &normalData, std::vector<GLuint> &idxData);
	GLfloat* getVoxelPos();
	std::vector<glm::vec3> getVoxelPos(bool state);
	int getResol();
	float getMaxLen();

private:
	glm::vec3 getMidPoint(const int x, const int y, const int z, const int code);
	glm::vec3 getNormal(glm::vec3 u, glm::vec3 v);

	int resol;
	GLfloat voxelLen;
	voxel *voxelData;
};

#endif