#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "glad/glad.h"
#include "texture.h"

using namespace std;


class Framebuffer
{
public:
	Framebuffer();
	void generate();
	void destroy();
	void bindTexture(Texture *tex, const int type);
	unsigned int getFBO();

private:
	unsigned int FBO, RBO;
};

#endif