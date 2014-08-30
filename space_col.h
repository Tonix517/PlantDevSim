#ifndef SPACE_COL_H
#define SPACE_COL_H

#include "Internode.h"
#include "GpuDelegates.h"
#include "Bud.h"
#include "Common.h"

__device__
extern struct Cell *_mat;

//	Init
//
__global__
void gpuInit(void *pmat);

//	By Voxels
//
__global__
void space_colonization_on_voxels(InternodeDelegate *pBuf, unsigned nBudCount);

__global__
void shadow_calculation_on_voxels(InternodeDelegate *pBuf, unsigned nInternodeCount);

//	NOTE: should only be called after calling the above two
//	POST: user needs to Normalize the vec
//
__global__
void get_optimized_bud_vector(InternodeDelegate *pBuf, struct OptVec *pOptVecs);

#endif