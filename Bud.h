#ifndef BUD_H
#define BUD_H

#include "GenMarker.h"
#include "Config.h"
#include "Vect3d.h"

//	Bud Optimized Vectors
struct OptVec
{
	float vec[3];
};
extern struct OptVec *optVecs;
extern struct OptVec *d_optVec;

///
///
///


struct Bud : GenMarker
{
	Bud()
	{		
		nId = -1;
		fLightFactor = 0;
	}

	Vect3d pos;// bud position
	Vect3d ort;// original bud orientation

	//	Light received Factor
	float fLightFactor;

	//
	void Render();

	//	for Speed Up - space partitioning
	//NOTE: for each Space unit, an extention will be added when executing
	//	for example, (SpaceDim + Extra)^3
	unsigned SpaceInx;

	int nId;

public:
	static float getLightReceived(float budOrt[3], float sunVec[3]);

private:

	static float _specular[];// green
};


#endif