#include "common.h"

#include <math.h>
#include <assert.h>
#include "GL/glut.h"

struct Tree tree;
struct Cell mat[X_Dim][Y_Dim][Z_Dim];
struct Cell *d_mat;

static float SunVec[3] = SUN_VEC;
