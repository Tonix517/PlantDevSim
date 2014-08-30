#ifndef DOMAIN_H
#define DOMAIN_H

#include "vect3d.h"
#include "shared_ptr.h"
#include "Config.h"

#include "Bud.h"
#include "Leaf.h"
#include "Internode.h"

#include "GL/glut.h"

extern float *h_shadow;
extern float *d_shadow;

///
///
class Domain
{
public:

	Domain()
		: _nLastGen(-1)
	{ }

	void Init();
	void Destroy();

	void Render();

	//	growth control
	//	one stage ahead per calling
	//	NOTE: should be called only after voxels processed
	void grow();

private:

	//
	void makeInternode(	Internode &rNode, int nParentId, 
						Vect3d &vNodePos, Vect3d &vNodeVec, 
						Vect3d &vTermBudVec,
						bool bHasLeaf = false, bool bHasAnBud = false);
	
	void addInternode( Internode &rNode);

	//	
	void gpu_SpaceColonization();

	//
	void RenderFrame();	
	void RenderTree();
	void RenderShadow();
	void CalcGroundShadow();
	void RenderGroundShadow();
	void RenderBarriers();
	
	void checkCells();
	void checkShadow();

private:

	unsigned _nLastGen;

};

#endif