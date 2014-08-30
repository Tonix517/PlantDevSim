#ifndef SHADOW_H
#define SHADOW_H

#include "Config.h"
#include "space_col.h"

typedef float MARK;

extern MARK *h_ground_shadow;
extern MARK *d_ground_shadow;

extern MARK *h_smoothed_shadow;
extern MARK *d_smoothed_shadow;

extern const unsigned GroundDim; 

///
void reset_ground_shadow();

__global__
void gpu_calc_shadow( MARK *pBuf, struct Cell *mat);

__global__
void gpu_smooth_shadow( MARK *pBuf, MARK *pSmoothedShadow);

#endif