#ifndef LEAF_H
#define LEAF_H

#include "GenMarker.h"
#include "Vect3d.h"

//	I will not make a Leaf too large too, 
//	so I'd like just shadow it in the middle point
//
struct Leaf : GenMarker
{
	void Render();

	Vect3d pos;// root position
	Vect3d vec;

private:

	static float _specular[];// green
};

#endif