#ifndef _PRIMITIVE_HPP_
#define _PRIMITIVE_HPP_

#include "glad/glad.h"
#include <glm/glm.hpp>
#include <vector>

using namespace std;


//Primitive
namespace Primitive
{
/*--------------------------
Create a 2D quad
--------------------------*/
void createQuad2d(vector<GLfloat> &vData, vector<GLfloat> &uvData, vector<GLuint> &idxData)
{
	GLfloat v[] = {
		-1, -1,
		 1, -1,
		-1,  1,
		 1,  1
	};
	GLfloat uv[] = {
		0.0, 0.0,
		1.0, 0.0,
		0.0, 1.0,
		1.0, 1.0,
	};
	GLuint idx[] = {
		0, 1, 2, 
		2, 1, 3
	};

	vData.insert(vData.begin(), v, v + 8);
	uvData.insert(uvData.begin(), uv, uv + 8);
	idxData.insert(idxData.begin(), idx, idx + 6);
}


/*--------------------------
Create a quad
--------------------------*/
void createQuad(vector<GLfloat> &vData, vector<GLfloat> &nData, vector<GLfloat> &uvData, vector<GLuint> &idxData)
{
	GLfloat v[] = {
		-0.5, 0.0, -0.5,
		-0.5, 0.0,  0.5,
		 0.5, 0.0, -0.5,
		 0.5, 0.0,  0.5
	};
	GLfloat uv[] = {
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0
	};
	GLfloat n[] = {
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0
	};
	GLuint idx[] = {
		0, 1, 2, 
		2, 1, 3
	};

	vData.insert(vData.begin(), v, v + 12);
	nData.insert(nData.begin(), n, n + 12);
	uvData.insert(uvData.begin(), uv, uv + 8);
	idxData.insert(idxData.begin(), idx, idx + 6);
}


/*--------------------------
Create a cube
--------------------------*/
void createCube(vector<GLfloat> &vData, vector<GLfloat> &nData, vector<GLfloat> &uvData, vector<GLuint> &idxData)
{
	GLfloat v[] = {
		//Top
		-0.5, 0.5, -0.5,
		-0.5, 0.5,  0.5,
		 0.5, 0.5, -0.5,
		 0.5, 0.5,  0.5,
		//Down
		-0.5, -0.5, -0.5,
		-0.5, -0.5,  0.5,
		 0.5, -0.5, -0.5,
		 0.5, -0.5,  0.5,
		//Left
		 0.5, -0.5, -0.5,
		 0.5, -0.5,  0.5,
		 0.5,  0.5, -0.5,
		 0.5,  0.5,  0.5,
		//Right
		-0.5, -0.5, -0.5,
		-0.5, -0.5,  0.5,
		-0.5,  0.5, -0.5,
		-0.5,  0.5,  0.5,
		//Front
		-0.5, -0.5, 0.5,
		-0.5,  0.5, 0.5,
		 0.5, -0.5, 0.5,
		 0.5,  0.5, 0.5,
		//Back
		-0.5, -0.5, -0.5,
		-0.5,  0.5, -0.5,
		 0.5, -0.5, -0.5,
		 0.5,  0.5, -0.5
	};
	GLfloat uv[] = {
		//Top
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
		//Down
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
		//Left
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
		//Right
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
		//Front
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
		//Back
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0
	};
	GLfloat n[] = {
		//Top
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		//Down
		0.0, -1.0, 0.0,
		0.0, -1.0, 0.0,
		0.0, -1.0, 0.0,
		0.0, -1.0, 0.0,
		//Left
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		//Right
		-1.0, 0.0, 0.0,
		-1.0, 0.0, 0.0,
		-1.0, 0.0, 0.0,
		-1.0, 0.0, 0.0,
		//Front
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		//Back
		0.0, 0.0, -1.0,
		0.0, 0.0, -1.0,
		0.0, 0.0, -1.0,
		0.0, 0.0, -1.0
	};
	GLuint idx[] = {
		//Top
		0, 1, 2,
		2, 1, 3,
		//Down
		4, 6, 5,
		6, 7, 5,
		//Left
		8, 10, 9,
		10, 11, 9,
		//Right
		12, 13, 14,
		14, 13, 15,
		//Front
		16, 18, 17,
		18, 19, 17,
		//Back
		20, 21, 22,
		22, 21, 23
	};

	vData.insert(vData.begin(), v, v + 72);
	nData.insert(nData.begin(), n, n + 72);
	uvData.insert(uvData.begin(), uv, uv + 48);
	idxData.insert(idxData.begin(), idx, idx + 36);
}


/*--------------------------
Create a skybox
--------------------------*/
void createSkybox(vector<GLfloat> &vData)
{
	GLfloat v[] = {
	    -1.0f,  1.0f, -1.0f,
	    -1.0f, -1.0f, -1.0f,
	     1.0f, -1.0f, -1.0f,
	     1.0f, -1.0f, -1.0f,
	     1.0f,  1.0f, -1.0f,
	    -1.0f,  1.0f, -1.0f,

	    -1.0f, -1.0f,  1.0f,
	    -1.0f, -1.0f, -1.0f,
	    -1.0f,  1.0f, -1.0f,
	    -1.0f,  1.0f, -1.0f,
	    -1.0f,  1.0f,  1.0f,
	    -1.0f, -1.0f,  1.0f,

	     1.0f, -1.0f, -1.0f,
	     1.0f, -1.0f,  1.0f,
	     1.0f,  1.0f,  1.0f,
	     1.0f,  1.0f,  1.0f,
	     1.0f,  1.0f, -1.0f,
	     1.0f, -1.0f, -1.0f,

	    -1.0f, -1.0f,  1.0f,
	    -1.0f,  1.0f,  1.0f,
	     1.0f,  1.0f,  1.0f,
	     1.0f,  1.0f,  1.0f,
	     1.0f, -1.0f,  1.0f,
	    -1.0f, -1.0f,  1.0f,

	    -1.0f,  1.0f, -1.0f,
	     1.0f,  1.0f, -1.0f,
	     1.0f,  1.0f,  1.0f,
	     1.0f,  1.0f,  1.0f,
	    -1.0f,  1.0f,  1.0f,
	    -1.0f,  1.0f, -1.0f,

	    -1.0f, -1.0f, -1.0f,
	    -1.0f, -1.0f,  1.0f,
	     1.0f, -1.0f, -1.0f,
	     1.0f, -1.0f, -1.0f,
	    -1.0f, -1.0f,  1.0f,
	     1.0f, -1.0f,  1.0f
	};

	vData.insert(vData.begin(), v, v + 108);
}


/*--------------------------
Create a sphere
--------------------------*/
void createSphere(vector<GLfloat> &vData, vector<GLfloat> &nData, vector<GLfloat> &uvData, vector<GLuint> &idxData)
{
	unsigned int xSegments = 64;
	unsigned int ySegments = 64;
	const float PI = 3.14159265359f;

	//Attributes
	for(unsigned int y = 0; y <= ySegments; ++y)
		for(unsigned int x = 0; x <= xSegments; ++x)
		{
			float xSeg = (float)x / (float)xSegments;
			float ySeg = (float)y / (float)ySegments;
			float xPos = std::cos(xSeg * 2.0f * PI) * std::sin(ySeg * PI);
			float yPos = std::cos(ySeg * PI);
			float zPos = std::sin(xSeg * 2.0f * PI) * std::sin(ySeg * PI);

			vData.push_back(xPos);
			vData.push_back(yPos);
			vData.push_back(zPos);
			nData.push_back(xPos);
			nData.push_back(yPos);
			nData.push_back(zPos);
			uvData.push_back(xSeg);
			uvData.push_back(ySeg);
		}

	//Indices
	bool oddRow = false;

	for(int y = 0; y < ySegments; ++y)
	{
		//Odd rows
		if(!oddRow)
		{
			for(int x = xSegments; x >= 0; --x)
			{
				idxData.push_back((y+1)*(xSegments+1) + x);
				idxData.push_back(y*(xSegments+1) + x);
			}
		}
		//Even rows
		else
		{
			for(int x = 0; x <= xSegments; ++x)
			{
				idxData.push_back(y*(xSegments+1) + x);
				idxData.push_back((y+1)*(xSegments+1) + x);
			}
		}
		oddRow = !oddRow;
	}
}

}

#endif