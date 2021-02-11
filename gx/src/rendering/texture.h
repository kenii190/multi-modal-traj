#ifndef TEXTURE_H
#define TEXTURE_H

#include "glad/glad.h"
#include <string>
#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;


class Texture
{
public:
	Texture();
	void genBuffer(const unsigned int width, const unsigned int height, const glm::vec3 &color);
	void genBuffer(const unsigned int width, const unsigned int height, const unsigned int type=0);
	void genBuffer(const char *filename, const GLint interFormat=GL_SRGB, const GLenum format=GL_BGR);
	void genBuffer(const string filenames[], const GLint interFormat=GL_RGB, const GLenum format=GL_RGB);
	void destroy();
	void loadFromFile(const char *filename, const GLint interFormat=GL_SRGB, const GLenum format=GL_BGR);
	void loadCubemapFromFile(const string filenames[], const GLint interFormat=GL_RGB, const GLenum format=GL_RGB);
	void clear(const glm::vec3 &color);
	void drawCircle(const glm::vec3 &color, const int x, const int y, const int radius);
	void drawLine(const glm::vec3 &color, const int x1, const int y1, const int x2, const int y2);
	void upload();
	unsigned int getTexID();
	unsigned int getWidth();
	unsigned int getHeight();
	GLuint getTexType();
	void bind(const GLenum tex=GL_TEXTURE0);

private:
	cv::Mat img;
	unsigned int texID;
	unsigned int width;
	unsigned int height;
	GLuint texType;
};

#endif