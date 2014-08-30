#include "util.h"

#include <assert.h>
#include <math.h>

//
float dist(float *p1, float *p2)
{
	assert(p1 && p2);
	float vec[3] = {0};
	vec[0] = p1[0] - p2[0];
	vec[1] = p1[1] - p2[1];
	vec[2] = p1[2] - p2[2];

	return sqrt( vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

void normalize(float vec[3])
{
	assert(vec);
	float fLen = len(vec);
	if(fLen > 0)
	{
		vec[0] /= fLen;
		vec[1] /= fLen;
		vec[2] /= fLen;
	}
}

float len(float vec[3])
{
	assert(vec);
	return sqrt( vec[0] * vec[0] + 
				 vec[1] * vec[1] + 
				 vec[2] * vec[2] );
}
