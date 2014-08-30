#ifndef BARRIER_H
#define BARRIER_H

#include "Config.h"
#include "Common.h"

enum BTYPE {SPHERE, CUBE};

//	struct
//
struct Barrier
{
public:

	Barrier()
	{
		center[0] = 0;
		center[1] = 0;
		center[2] = 0;
		fSize = 1;

		eType = SPHERE;
	}

public:
		
	void Render();

private:
	void renderSphere();
	void renderCube();

public:

	float center[3];
	
	//	For Sphere, it is radius
	//	For Cube, it is side length
	float fSize;

	enum BTYPE eType;	

private:
	static float _material[];
};

// Barrier List
extern struct Barrier barriers[BARRIER_COUNT];
extern struct Barrier *d_barriers;
extern unsigned nBarrierCount;
void copyBarriersToGPU();

//	GPU barrier processing 
__global__
void gpu_process_barriers(struct Barrier *pBarrier, unsigned nCount, struct Cell *pCell);

#endif