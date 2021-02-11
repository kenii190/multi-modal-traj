#include "texture.h"
#include <iostream>


/*---------------------
Constructor
---------------------*/
Texture::Texture()
{
	texID   = 0;
	width   = 0;
	height  = 0;
	texType = 0;
}


/*---------------------
Generate buffers
---------------------*/
void Texture::genBuffer(const unsigned int width, const unsigned int height, const glm::vec3 &color)
{
	this->width  = width;
	this->height = height;
	texType = GL_TEXTURE_2D;
	
	img = cv::Mat(width, height, CV_8UC3, cv::Scalar(color.x, color.y, color.z));

	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGB,
		img.rows,
		img.cols,
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		img.data
	);
}


void Texture::genBuffer(const unsigned int width, const unsigned int height, const unsigned int type)
{
	this->width  = width;
	this->height = height;
	texType = GL_TEXTURE_2D;
	glGenTextures(1, &texID);

	//Render to depth texture
	if(type == 0)
	{
		float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
		texType = GL_TEXTURE_2D;
		glBindTexture(GL_TEXTURE_2D, texID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_DEPTH_COMPONENT,
			width,
			height,
			0,
			GL_DEPTH_COMPONENT,
			GL_FLOAT,
			nullptr
		);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	//Render to depth cubemap
	else if(type == 1)
	{
		texType = GL_TEXTURE_CUBE_MAP;
		glBindTexture(GL_TEXTURE_CUBE_MAP, texID);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		for(int i = 0; i < 6; ++i)
			glTexImage2D(
				GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
				0,
				GL_DEPTH_COMPONENT,
				width,
				height,
				0,
				GL_DEPTH_COMPONENT,
				GL_FLOAT,
				nullptr
			);
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	}

	//Render to RGB texture
	if(type == 2)
	{
		texType = GL_TEXTURE_2D;
		glBindTexture(GL_TEXTURE_2D, texID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			width,
			height,
			0,
			GL_RGB,
			GL_UNSIGNED_BYTE,
			nullptr
		);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	//Render to floating point RGBA texture
	else if(type == 3)
	{
		texType = GL_TEXTURE_2D;
		glBindTexture(GL_TEXTURE_2D, texID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(
			GL_TEXTURE_2D, 
			0, 
			GL_RGBA16F, 
			width, 
			height,
			0,
			GL_RGBA,
			GL_FLOAT,
			nullptr
		);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	//Render to multisampled RGB texture
	else if(type == 4)
	{
		texType = GL_TEXTURE_2D_MULTISAMPLE;
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texID);
		glTexImage2DMultisample(
			GL_TEXTURE_2D_MULTISAMPLE, 
			4, 
			GL_RGB,
			width,
			height,
			GL_TRUE
		);
		glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
	}
}


void Texture::genBuffer(const char *filename, const GLint interFormat, const GLenum format)
{
	width   = 0;
	height  = 0;
	texType = GL_TEXTURE_2D;
	glGenTextures(1, &texID);
	loadFromFile(filename, interFormat, format);
}


void Texture::genBuffer(const string filenames[], const GLint interFormat, const GLenum format)
{
	width   = 0;
	height  = 0;
	texType = GL_TEXTURE_CUBE_MAP;
	glGenTextures(1, &texID);
	loadCubemapFromFile(filenames, interFormat, format);
}


/*---------------------
Destroy buffers
---------------------*/
void Texture::destroy()
{
	glDeleteTextures(1, &texID);
}


/*---------------------
Load texture from file
---------------------*/
void Texture::loadFromFile(const char *filename, const GLint interFormat, const GLenum format)
{
	img = cv::imread(filename, cv::IMREAD_COLOR);
	if(!img.data)
	{
		cout << "Failed to read the image" << endl;
		return;
	}

	this->width  = img.rows;
	this->height = img.cols;

	glBindTexture(GL_TEXTURE_2D, texID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		interFormat,
		img.rows,
		img.cols,
		0,
		format,
		GL_UNSIGNED_BYTE,
		img.data
	);
	glGenerateMipmap(GL_TEXTURE_2D);

	img.release();
}


/*---------------------
Load cubemap from file
---------------------*/
void Texture::loadCubemapFromFile(const string filenames[], const GLint interFormat, const GLenum format)
{
	glBindTexture(GL_TEXTURE_CUBE_MAP, texID);

	for(int i = 0; i < 6; ++i)
	{
		img = cv::imread(filenames[i].c_str(), cv::IMREAD_COLOR);
		if(!img.data)
		{
			cout << "Failed to read the image" << endl;
			return;
		}

		if(i == 0)
		{
			this->width  = img.rows;
			this->height = img.cols;
		}

		glTexImage2D(
			GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
			0,
			interFormat,
			img.rows,
			img.cols,
			0,
			format,
			GL_UNSIGNED_BYTE,
			img.data
		);
		img.release();
	}

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}


/*---------------------
clear the tex with color
---------------------*/
void Texture::clear(const glm::vec3 &color)
{
	img.setTo(cv::Scalar(color.x, color.y, color.z));
}


/*---------------------
Draw a circle on the tex
---------------------*/
void Texture::drawCircle(const glm::vec3 &color, const int x, const int y, const int radius)
{
	cv::circle(img, cv::Point(x, y), radius, cv::Scalar(color.x, color.y, color.z));
}


/*---------------------
Draw a line on the tex
---------------------*/
void Texture::drawLine(const glm::vec3 &color, const int x1, const int y1, const int x2, const int y2)
{
	cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(color.x, color.y, color.z));
}


/*---------------------
Upload the tex (2D)
---------------------*/
void Texture::upload()
{
	glBindTexture(texType, texID);
	glTexImage2D(
		texType,
		0,
		GL_RGB,
		img.rows,
		img.cols,
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		img.data
	);
}


/*---------------------
Get texture ID
---------------------*/
unsigned int Texture::getTexID()
{
	return texID;
}


/*---------------------
Get texture width
---------------------*/
unsigned int Texture::getWidth()
{
	return width;
}


/*---------------------
Get texture height
---------------------*/
unsigned int Texture::getHeight()
{
	return height;
}


/*---------------------
Get texture type
---------------------*/
GLuint Texture::getTexType()
{
	return texType;
}


/*---------------------
Bind the texture
---------------------*/
void Texture::bind(const GLenum tex)
{
	glActiveTexture(tex);
	glBindTexture(texType, texID);
}