#ifndef INTERNODE_H
#define INTERNODE_H

#include "shared_ptr.h"
#include "Bud.h"
#include "Leaf.h"
#include "Vect3d.h"
#include "GenMarker.h"

///
//	as Internode ID, NOTE it is also the index that the node in the tree vector
extern int nInternodeCount;

///
///	Note: I will not make a Internode too long
///		  so I will only pick 3 points from the Internode
///		  to make the shadow.
///

struct Internode : GenMarker
{
	Internode()
		: width(0.2f)
		, pTerminalBud(NULL)
		, pAnxillaryBud(NULL)
		, pLeaf(NULL)
		, nGrowthStep(0)
		, id(-1)
	    , ancestorId(-1)
		, bFromAnBud(false)
	{	}

	//
	void Render();

	//	meta info
	int id;
	int ancestorId;
	bool bFromAnBud;

	//
	Vect3d pos;// base pos
	Vect3d vec; 

	float width; 

	unsigned nGrowthStep;

	//	
	SharedPtr<Bud> pTerminalBud;	
	SharedPtr<Bud> pAnxillaryBud;	
	SharedPtr<Leaf> pLeaf;

private:

	static float _specular[];// green
};

#endif