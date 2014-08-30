#ifndef CAMERA_H
#define CAMERA_H

#include "Timer.h"

class Camera : public Timer
{
public:
	
	Camera();

	void Set( float posx, float posy, float posz,
		      float cx, float cy, float cz,
			  float upx, float upy, float upz);
	void Put( );
	
	void Rotating(bool);

	//	interaction
	void ZoomIn();
	void ZoomOut();
	void Rotate(float fAngle);

private:

	//	as a const first
	static const float RotateCount;

	bool _bRotate;
	float _fAngle;

	float _fposx, _fposy, _fposz;
	float _fctrx, _fctry, _fctrz;
	float _fupx, _fupy, _fupz;

};

#endif