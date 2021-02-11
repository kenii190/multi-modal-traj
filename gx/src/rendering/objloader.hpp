#ifndef _OBJ_LOADER_HPP_
#define _OBJ_LOADER_HPP_

#include <glm/glm.hpp>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include "tiny_obj_loader.h"
#include "glad/glad.h"

using namespace std;

/*------------------------
Process vec3
------------------------*/
static vec3 processVec3(char *split)
{
	vec3 v;

	if((split = strtok(NULL, " ")) == NULL) return v;
	v.x = atof(split);

	if((split = strtok(NULL, " ")) == NULL) return v;
	v.y = atof(split);

	if((split = strtok(NULL, " ")) == NULL) return v;
	v.z = atof(split);

	return v;
}


/*------------------------
Process vec2
------------------------*/
static vec2 processVec2(char *split)
{
	vec2 v;

	if((split = strtok(NULL, " ")) == NULL) return v;
	v.x = atof(split);

	if((split = strtok(NULL, " ")) == NULL) return v;
	v.y = atof(split);

	return v;
}


/*------------------------
Process face
------------------------*/
static void processFace(char *split, vector<GLuint> &idx, vector<GLuint> &nidx)
{
	vector<char*> s;

	if((split = strtok(NULL, " ")) == NULL) return;
	while(split != NULL)
	{
		s.push_back(split);
		split = strtok(NULL, " ");
	}

	vector<GLuint> quadIdx(s.size());
	vector<GLuint> quadNidx(s.size());

	for(int i = 0; i < s.size(); ++i)
	{
		quadIdx[i] = ((split = strtok(s[i], "/")) == NULL) ? 0 : atoi(split);
		int tmp = ((split = strtok(NULL, "/")) == NULL) ? 0 : atoi(split);
		quadNidx[i] = ((split = strtok(NULL, "/")) == NULL) ? 0 : atoi(split);
	}

	if(s.size() == 3)
	{
		idx.push_back(quadIdx[0]-1);
		idx.push_back(quadIdx[1]-1);
		idx.push_back(quadIdx[2]-1);

		nidx.push_back(quadNidx[0]-1);
		nidx.push_back(quadNidx[1]-1);
		nidx.push_back(quadNidx[2]-1);
	}
	else if(s.size() == 4)
	{
		idx.push_back(quadIdx[0]-1);
		idx.push_back(quadIdx[1]-1);
		idx.push_back(quadIdx[2]-1);
		idx.push_back(quadIdx[0]-1);
		idx.push_back(quadIdx[2]-1);
		idx.push_back(quadIdx[3]-1);

		nidx.push_back(quadNidx[0]-1);
		nidx.push_back(quadNidx[1]-1);
		nidx.push_back(quadNidx[2]-1);
		nidx.push_back(quadNidx[0]-1);
		nidx.push_back(quadNidx[2]-1);
		nidx.push_back(quadNidx[3]-1);
	}
}


//ObjLoader
namespace ObjLoader
{
/*------------------------
Load from *.obj file
------------------------*/
void loadObj(
	const char *filename, 
	vector<GLfloat> &vertexData, 
	vector<GLfloat> &normalData, 
	vector<GLfloat> &uvData, 
	vector<GLuint> &idxData
){
	std::ifstream fp(filename, std::ifstream::in);
	if(!fp.is_open())
	{
		printf("ERROR: Failed to read %s\n", filename);
		return;
	}

	string line;
	char buf[128];
	size_t nbytes = 0;
	int bytes_read = 0;
	vector<GLfloat> nData;
	vector<GLuint> nidxData;

	while(getline(fp, line))
	{
		if(line.length() > 0)
		{
			line[line.length() - 1] = '\0';
			strcpy(buf, line.c_str());
			char *split = strtok(buf, " ");

			//Vertex-----------------------------
			if(strcmp(split, "v") == 0)
			{
				vec3 v = processVec3(split);
				vertexData.push_back(v.x);
				vertexData.push_back(v.y);
				vertexData.push_back(v.z);
			}

			//UV----------------------------------
			else if(strcmp(split, "vt") == 0)
			{
				vec2 uv = processVec2(split);
				uvData.push_back(uv.x);
				uvData.push_back(uv.y);
			}

			//Normal------------------------------
			else if(strcmp(split, "vn") == 0)
			{
				vec3 n = processVec3(split);
				nData.push_back(n.x);
				nData.push_back(n.y);
				nData.push_back(n.z);
			}

			//Polygonal face----------------------
			else if(strcmp(split, "f") == 0)
			{
				vector<GLuint> idx;
				vector<GLuint> nidx;

				processFace(split, idx, nidx);
				for(int i = 0; i < idx.size(); ++i)
				{
					idxData.push_back(idx[i]);
					nidxData.push_back(nidx[i]);
				}
			}
		}
	}

	normalData.resize(vertexData.size());
	for(int i = 0; i < idxData.size(); ++i)
	{
		normalData[idxData[i]*3  ] = nData[nidxData[i]*3  ];
		normalData[idxData[i]*3+1] = nData[nidxData[i]*3+1];
		normalData[idxData[i]*3+2] = nData[nidxData[i]*3+2];
	}
	fp.close();
}


void loadObj2(
	const char* filename,
	vector<GLfloat> &vertexData, 
	vector<GLfloat> &normalData, 
	vector<GLfloat> &uvData, 
	vector<GLuint> &idxData
){
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;
	string warn, err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

	if(!warn.empty()) cout << "[Warning] " << warn << endl;
	if(!err.empty()) cout << "[Error] " << err << endl;
	if(!ret) return;

	vertexData = attrib.vertices;
	normalData.resize(vertexData.size());
	uvData.resize(vertexData.size() * 2 / 3);

	//For each shape
	for(size_t s = 0; s < shapes.size(); ++s)
	{
		size_t idx_offset = 0;

		//For each face
		for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f)
		{
			int fv = shapes[s].mesh.num_face_vertices[f];

			//For each vertex
			for(size_t v = 0; v < fv; ++v)
			{
				tinyobj::index_t idx = shapes[s].mesh.indices[idx_offset + v];
				idxData.push_back(idx.vertex_index);
				
				normalData[3*idx.vertex_index    ] = attrib.normals[3*idx.normal_index    ];
				normalData[3*idx.vertex_index + 1] = attrib.normals[3*idx.normal_index + 1];
				normalData[3*idx.vertex_index + 2] = attrib.normals[3*idx.normal_index + 2];
				
				uvData[2*idx.vertex_index    ] = attrib.texcoords[2*idx.texcoord_index    ];
				uvData[2*idx.vertex_index + 1] = attrib.texcoords[2*idx.texcoord_index + 1];
			}
			idx_offset += fv;
		}
	}
}

}

#endif