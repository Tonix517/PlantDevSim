#include "Camera.h"
#include "Config.h"

#include "GL/glut.h"

#include "vect3d.h"

const float Camera::RotateCount = 0.4f;

Camera::Camera()
	:_fposx(0), _fposy(0), _fposz(0),
	 _fctrx(0), _fctry(0), _fctrz(0),
	 _fupx(0), _fupy(0), _fupz(0),
	 _bRotate(false),_fAngle(0)
{

}

void Camera::Rotating(bool bRotate)
{
	_bRotate = bRotate;
}

void Camera::Put()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	const float fv = 0.1;
	glFrustum(-fv, fv, -fv, fv, 1, 1000);
	gluLookAt(	_fposx, _fposy, _fposz, 
				_fctrx, _fctry, _fctrz,
				_fupx, _fupy, _fupz
			 );
	//gluPerspective(40,(GLfloat)WindowWidth/(GLfloat)WindowHeight,0.01,1000);

	if(_bRotate)
	{
		_fAngle += GetInterval() / 1000.f * 360.f * RotateCount;		
	}	

	glRotatef(-_fAngle, 0, 1, 0);
}

void Camera::Set(	float posx, float posy, float posz,
					float cx, float cy, float cz,
			        float upx, float upy, float upz)
{
	_fposx = posx; _fposy = posy; _fposz = posz;
	_fctrx = cx; _fctry = cy; _fctrz = cz;
	_fupx = upx; _fupy = upy; _fupz  =upz;
}

void Camera::ZoomIn()
{
	Vect3d vOrigViewVec( _fposx - _fctrx, 
						 _fposy - _fctry,
						 _fposz - _fctrz);

	vOrigViewVec *= 0.95;
	_fposx = _fctrx + vOrigViewVec.x();
	_fposy = _fctry + vOrigViewVec.y();
	_fposz = _fctrz + vOrigViewVec.z();
}

void Camera::ZoomOut()
{
	Vect3d vOrigViewVec( _fposx - _fctrx, 
						 _fposy - _fctry,
						 _fposz - _fctrz);

	vOrigViewVec *= 1.05;
	_fposx = _fctrx + vOrigViewVec.x();
	_fposy = _fctry + vOrigViewVec.y();
	_fposz = _fctrz + vOrigViewVec.z();
}

void Camera::Rotate(float fAngle)
{
	_fAngle += fAngle;
}