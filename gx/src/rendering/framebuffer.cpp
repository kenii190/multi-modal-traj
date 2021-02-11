#include "framebuffer.h"
#include <iostream>


/*---------------------
Constructor
---------------------*/
Framebuffer::Framebuffer()
{
	RBO = 0;
	FBO = 0;
}


/*---------------------
Generate FBO
---------------------*/
void Framebuffer::generate()
{
	if(FBO <= 0)
		glGenFramebuffers(1, &FBO);
}


/*---------------------
Destroy FBO & RBO
---------------------*/
void Framebuffer::destroy()
{
	glDeleteFramebuffers(1, &FBO);

	if(RBO != 0)
		glDeleteRenderbuffers(1, &RBO);
}


/*---------------------
Bind texture to framebuffer
---------------------*/
void Framebuffer::bindTexture(Texture *tex, const int type)
{
	//2D depth attachment
	if(type == 0)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex->getTexID(), 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	//3D depth attachment
	else if(type == 1)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, tex->getTexID(), 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	//2D color attachment
	else if(type == 2)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex->getTexID(), 0);

		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete" << std::endl;

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	//2D color attachment with depth
	else if(type == 3)
	{
		//Depth render buffer
		glGenRenderbuffers(1, &RBO);
		glBindRenderbuffer(GL_RENDERBUFFER, RBO);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, tex->getWidth(), tex->getHeight());

		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex->getTexID(), 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RBO);

		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete" << std::endl;

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	//2D color attachment with multisample
	else if(type == 4)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex->getTexID(), 0);

		glGenRenderbuffers(1, &RBO);
		glBindRenderbuffer(GL_RENDERBUFFER, RBO);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, tex->getWidth(), tex->getHeight());
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

		if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete" << std::endl;

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}


/*---------------------
Get FBO
---------------------*/
unsigned int Framebuffer::getFBO()
{
	return FBO;
}