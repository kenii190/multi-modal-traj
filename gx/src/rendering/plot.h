#ifndef PLOT_H
#define PLOT_H

#include <vector>
#include <glm/glm.hpp>
#include "obj3d.h"

using namespace std;
using namespace glm;


class Plot
{
public:
	Plot(Mesh *meshPoint, Mesh *meshLine, float scale=1.0f, float pointScale=0.5f);
	~Plot();
	void setPoints(const vector<vec3> &points);
	vector<vec3> getPoints();
	void draw(Shader *shaderPoint, Shader *shaderLine);

private:
	float scale;
	float pointScale;
	vector<vec3> points;
	Mesh *meshPoint, *meshLine;
	Obj3d *objPoint, *objLine;
};

#endif