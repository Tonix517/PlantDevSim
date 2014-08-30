#include "Scene.h"

#include "GL/glut.h"

#include "Config.h"

void Scene::_RenderSky()
{

}

void Scene::_RenderGround()
{
	//	TODO : texture
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_LIGHTING);
	glBegin(GL_QUADS);
		glColor3ub(180,180,180);
		glVertex3f(-SCENESIZE, SCENEHEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT,  SCENESIZE);
		glVertex3f(-SCENESIZE, SCENEHEIGHT,  SCENESIZE);
	glEnd();
	glEnable(GL_LIGHTING);
}