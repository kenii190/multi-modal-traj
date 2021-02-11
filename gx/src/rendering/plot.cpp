#include "plot.h"
#include <glm/gtc/matrix_transform.hpp>


/*-------------------------
Constructor
-------------------------*/
Plot::Plot(Mesh *meshPoint, Mesh *meshLine, float scale, float pointScale)
{
	this->meshPoint  = meshPoint;
	this->meshLine   = meshLine;
	this->objPoint   = new Obj3d(meshPoint);
	this->objLine    = new Obj3d(meshLine);
	this->scale      = scale;
	this->pointScale = pointScale;
}


/*-------------------------
Destructor
-------------------------*/
Plot::~Plot()
{
	delete objPoint;
	delete objLine;
}


/*-------------------------
Access points
-------------------------*/
void Plot::setPoints(const vector<vec3> &points)
{
	this->points = points;
	vector<mat4> pointModelMats(points.size(), mat4(1.0));
	vector<mat4> lineModelMats(points.size()-1, mat4(1.0));

	pointModelMats[0] = 
	  glm::translate(pointModelMats[0], points[0] * scale) 
	* glm::scale(pointModelMats[0], vec3(pointScale));

	for(int i = 1; i < points.size(); ++i)
	{
		//Points
		pointModelMats[i] = 
		  glm::translate(pointModelMats[i], points[i] * scale) 
		* glm::scale(pointModelMats[i], vec3(pointScale));
	
		//Lines
		vec3 p1 = points[i-1] * scale;
		vec3 p2 = points[i] * scale;
		vec3 lineVec = p2 - p1;

		lineModelMats[i-1] = 
			glm::translate(lineModelMats[i-1], (p1 + p2) / 2.0f)
		  *	glm::mat4_cast(fromToRotation(vec3(0.0f, 1.0f, 0.0f), lineVec))
		  * glm::scale(lineModelMats[i-1], vec3(0.1f, glm::length(lineVec)/2.0f, 0.1f));
	}
	meshPoint->uploadInstanceMats(pointModelMats);
	meshLine->uploadInstanceMats(lineModelMats);
}


vector<vec3> Plot::getPoints()
{
	return points;
}


/*-------------------------
Draw the plot
-------------------------*/
void Plot::draw(Shader *shaderPoint, Shader *shaderLine)
{
	objPoint->drawInstanced(shaderPoint, points.size());
	objLine->drawInstanced(shaderLine, points.size()-1);
}