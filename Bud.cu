#include "Bud.h"

#include <assert.h>
#include <vector>
#include <math.h>
#include "GL/glut.h"
#include "Config.h"
#include "util.h"
#include "Internode.h"

#include "common.h"
#include "space_col.h"

//struct OptVec optVecs[MAX_INTERNODES];
struct OptVec *optVecs;
struct OptVec *d_optVec;

float Bud::_specular[] = {158.f/255, 189.f/255, 25.f/255, 1};

//

static
float dot_product(float vec1[3], float vec2[3])
{
	return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
}

static
float len(float vec[3])
{
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

static
void normalize(float vec[3])
{
	float fLen = len(vec);
	if(fLen > 0)
	{
		vec[0] /= fLen;
		vec[1] /= fLen;
		vec[2] /= fLen;
	}
}

//

float Bud::getLightReceived(float budOrt[3], float sunVec[3])
{
	float reversedSunVec[3] = {0};
	reversedSunVec[0] = -sunVec[0];
	reversedSunVec[1] = -sunVec[1];
	reversedSunVec[2] = -sunVec[2];
	normalize(reversedSunVec);

	float cBudOrt[3] = {0};
	cBudOrt[0] = budOrt[0];
	cBudOrt[1] = budOrt[1];
	cBudOrt[2] = budOrt[2];
	normalize(cBudOrt);

	//
	return dot_product(cBudOrt, reversedSunVec);
}

//

void Bud::Render()
{
	glPushMatrix();	
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, _specular);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, _specular);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, _specular);
		glTranslatef( pos.x(), pos.y(), pos.z());
		glutSolidSphere(0.1, 10, 10);			
	glPopMatrix();

	float myOptVec[3] = {0};
	if(nId != -1)	// Terminal Bud
	{
		memcpy(myOptVec, optVecs[nId].vec , sizeof(float) * 3);
	}
	else	//	Anxillary Bud
	{
		memcpy(myOptVec, ort, sizeof(float) * 3);
	}

	//	display
	//glDisable(GL_LIGHTING);
	//glColor3f(1, 1, 1);
	//const int factor = 1;
	//glBegin(GL_LINES);
	//	glVertex3f(pos.x(), pos.y(), pos.z());
	//	glVertex3f(	pos.x() + myOptVec[0] * factor, 
	//				pos.y() + myOptVec[1] * factor, 
	//				pos.z() + myOptVec[2] * factor);
	//glEnd();
	//glEnable(GL_LIGHTING);
	//glLineWidth(1);
}
