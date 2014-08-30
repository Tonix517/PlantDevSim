#ifndef GPU_DELEGATES_H
#define GPU_DELEGATES_H

#include <memory.h>
#include "Config.h"
//
#define INVALID_POS -999.f

struct InternodeDelegate
{
	float pos[3];
	float vec[3];
	float budPos[3];
	float budOrt[3];
	float anBudPos[3];
	float leafPos[3];
};

extern InternodeDelegate *d_internodes;
extern InternodeDelegate *h_internodes;

void copy_internodes();

#endif