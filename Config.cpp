#include "Config.h"

int WindowWidth = 600;
int WindowHeight = 600;

unsigned PartNum = 10;
float SpaceUnitDim = SCENESIZE * 2.f / PartNum;
float ExtraUnitDim = SpaceUnitDim * 0.1f;

unsigned GrownStep = 0;

int iRenderBud = 0;
int iShowCellSpaceStatus = 0;
int iShowCellShadow = 0;
int iRenderGroundShadow = 1;