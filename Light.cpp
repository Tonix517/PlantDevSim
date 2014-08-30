#include "Light.h"

#include <memory.h>

#include "GL/glut.h"

Light::Light()
{
	memset(light_ambient, 0, 4 * sizeof(float));
	memset(light_diffuse, 0, 4 * sizeof(float));
	memset(light_specular, 0, 4 * sizeof(float));
	memset(light_position, 0, 4 * sizeof(float));
}
	
void Light::Set(float *pAmb, float *pDiff, float *pSpec, float *pPos)
{
	memcpy(light_ambient, pAmb, 4 * sizeof(float));
	memcpy(light_diffuse, pDiff, 4 * sizeof(float));
	memcpy(light_specular, pSpec, 4 * sizeof(float));
	memcpy(light_position, pPos, 4 * sizeof(float));
}

void Light::Put()
{
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
}