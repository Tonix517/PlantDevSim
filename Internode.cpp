#include "Internode.h"

#include <math.h>
#include "GL/glut.h"
#include "Config.h"
#include "Common.h"

int nInternodeCount = 0;

float Internode::_specular[] = {128.f/255, 64.f/255, 0, 1};

void Internode::Render()
{	
	glPushMatrix();

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, _specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, _specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, _specular);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	GLUquadricObj *pInternode = gluNewQuadric(); // TODO: it is slow
	glPushMatrix();		

		glTranslatef(pos.x(), pos.y(), pos.z());
		
		Vect3d axis = Vect3d(0, 1, 0).Cross(vec);
		axis.Normalize();

		float a = acos(Vect3d(0, 1, 0).Dot(vec) / vec.Length()) * 180 / PI;
		glRotatef(a, axis.x(), axis.y(), axis.z());
		
		glRotatef(-90, 1, 0, 0);

		width = pow(sqrt(1.6),(int)(GrownStep - nGrowthStep)) * 0.04;
		gluCylinder(pInternode, width, width, vec.Length(), 15, 15);			
		gluDeleteQuadric(pInternode);

	glPopMatrix();	

	//	Render Buds
	if(iRenderBud != 0)
	{
		if(pTerminalBud != NULL)
		{
			pTerminalBud->Render();	
		}

		if(pAnxillaryBud != NULL)
		{
			pAnxillaryBud->Render();	
		}
	}
	////	Render Leaf
	//if(pLeaf != NULL)
	//{
	//	pLeaf->Render();	
	//}
	//
	glPopMatrix();
}
