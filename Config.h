#ifndef CONFIG_H
#define CONFIG_H

#define GPU_SPEED_UP 1

//	! WARNINGS !
//	Please be extremely careful to change the following numbers
//	There're limited block\thread resources in CUDA 1.0
//
#define SCENESIZE 20
#define DOMAIN_HEIGHT 40
#define SCENEHEIGHT 0

//	GPU Kernel Func
#define BLOCK_DIM_1D 2048
#define THREAD_DIM_1D 256

#define PI (3.1415926)

extern int WindowWidth;
extern int WindowHeight;

//	How fine the WHOLE simulation process will work on
#define SpaceGranularity 0.5

#define SUN_VEC {-1,-1,-1}

//	Space Colonization Params
#define OccuRadius 1.8
#define PerceptionRadius 3.6
#define PerceptionAngle 30

#define MaxShadowLayer 2

//	Space Partitioning Param
extern unsigned PartNum;
extern float SpaceUnitDim;
extern float ExtraUnitDim;

///	Growth Param
#define GEN_COUNT 18
extern unsigned GrownStep;

#define MAX_INTERNODES (262114)

// Diameter control
//	when calculating diameters - (d^n = d1^n + d2^n), 
//	how much the terminal node occupies (d1)
#define T_NODE_D_RATIO 0.6

//	Barrier Count
#define BARRIER_COUNT 3

extern int iRenderBud;
extern int iShowCellSpaceStatus;
extern int iShowCellShadow;
extern int iRenderGroundShadow;

//	LIGHT PARAM
#define LAMDA 0.3

//	SMOOTH SHADOW - how fine the smoothed shadow will be
#define SMOOTH_GRAN (0.05)
#define SMOOTH_THREAD_DIM 256
#define SMOOTH_BLOCK_DIM ( (SCENESIZE * 2 / SMOOTH_GRAN) * (SCENESIZE * 2 / SMOOTH_GRAN) / SMOOTH_THREAD_DIM)

#endif