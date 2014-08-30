#include "Leaf.h"

#include <math.h>

#include "Common.h"
#include "Config.h"
#include "GL/glut.h"

float Leaf::_specular[] = {169/255.f, 208/255.f, 107/255.f, 1}; 

void Leaf::Render()
{
	float ctr[3] = {0};
	ctr[0] = pos.x() + vec.x() / 2;
	ctr[1] = pos.y() + vec.y() / 2;
	ctr[2] = pos.z() + vec.z() / 2;

	glPushMatrix();

		glColor3f(169/255.f, 208/255.f, 107/255.f);
		
		glTranslatef(ctr[0], ctr[1], ctr[2]);

		// Rotate
		Vect3d axis = Vect3d(0, 1, 0).Cross(vec);
		axis.Normalize();

		float a = acos(Vect3d(0, 1, 0).Dot(vec) / vec.Length()) * 180 / PI;
		glRotatef(a, axis.x(), axis.y(), axis.z());

		// Scale to a ellipsoid
		glScalef(0.2, 0.8, 0.6);

		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, _specular);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, _specular);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, _specular);
		glutSolidSphere( vec.Length() / 2, 10, 10);
		
	glPopMatrix();
}
