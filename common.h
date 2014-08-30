#ifndef COMMON_H
#define COMMON_H

#include "Internode.h"
#include "Config.h"
#include <vector>

struct Cell
{
	Cell()
		: fIllum(1)
		, bOccupied(false)
		, nBudId(-1)
	{}

	bool bOccupied;
	int nBudId;
	float fIllum;
};

const unsigned X_Dim = SCENESIZE * 2 / SpaceGranularity;
const unsigned Y_Dim = (DOMAIN_HEIGHT - SCENEHEIGHT) / SpaceGranularity;
const unsigned Z_Dim = SCENESIZE * 2 / SpaceGranularity;

extern struct Cell mat[X_Dim][Y_Dim][Z_Dim];
extern struct Cell *d_mat;

struct Tree
{
	std::vector<Internode> internodes;
};
extern struct Tree tree;

#endif