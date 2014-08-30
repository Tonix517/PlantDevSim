#include "Barrier.h"
#include "Common.h"
#include "space_col.h"

#include "GL/glut.h"
#include <stdio.h>
#include <math.h>

unsigned nBarrierCount = 0;
struct Barrier barriers[BARRIER_COUNT];
struct Barrier *d_barriers;


__device__
Cell* getCell( unsigned xi, unsigned yi, unsigned zi)
{
	return _mat + xi * Y_Dim * Z_Dim + yi * Z_Dim + zi;
}

__device__
void getCurrentCellInx( unsigned *pxi, unsigned *pyi, unsigned *pzi)
{
	unsigned nAbsTid = blockIdx.x * blockDim.x + threadIdx.x;

	*pzi = nAbsTid % Z_Dim;
	*pxi = (nAbsTid - *pzi) / (Y_Dim * Z_Dim);
	*pyi = (nAbsTid - *pzi) / Z_Dim - *pxi * Y_Dim;
}

__device__
float dist( float *pos1, float *pos2 )
{
	return sqrt( pow(pos1[0] - pos2[0], 2) + 
				 pow(pos1[1] - pos2[1], 2) +
				 pow(pos1[2] - pos2[2], 2) );
}

///

__global__
void gpu_process_barriers(struct Barrier *pBarrier, unsigned nCount, struct Cell *pCell)
{
	//	TODO: ...
	_mat = pCell;

	unsigned xi = 0, yi = 0, zi = 0;
	getCurrentCellInx(&xi, &yi, &zi);

	if( (xi < X_Dim) &&
		(yi < Y_Dim) &&
		(zi < Z_Dim) )
	{

		float pos[3] = {0};
		pos[0] = xi * SpaceGranularity + (-SCENESIZE);
		pos[1] = yi * SpaceGranularity + (SCENEHEIGHT);
		pos[2] = zi * SpaceGranularity + (-SCENESIZE);

		Cell *pCell = getCell( xi, yi, zi);

		if( !pCell->bOccupied )
		{
			for(unsigned i = 0; i < nCount; i ++)
			{
				switch(pBarrier[i].eType)
				{
				case SPHERE:
					if(dist(pos, pBarrier[i].center) <= pBarrier[i].fSize + 0.1)// expand a little bit
					{
						pCell->bOccupied = true;
						break;
					}
					break;

				case CUBE:
					float dx = abs(pos[0] - pBarrier[i].center[0]);
					float dy = abs(pos[1] - pBarrier[i].center[1]);
					float dz = abs(pos[2] - pBarrier[i].center[2]);
					if( dx < (pBarrier[i].fSize/2) ||
						dy < (pBarrier[i].fSize/2) ||
						dz < (pBarrier[i].fSize/2) )
					{
						pCell->bOccupied = true;
						break;
					}
					break;
				}
			}// for
		}//	if
	}
}

void copyBarriersToGPU()
{
	if(nBarrierCount > 0)
	{
		cudaMemcpy( d_barriers, barriers, sizeof(Barrier) * nBarrierCount, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();
	}
}

float Barrier::_material[] = {0.6, 0.6, 0.6, 1};

void Barrier::Render()
{
	switch(eType)
	{
	case SPHERE:
		renderSphere();
		break;

	case CUBE:
		renderCube();
		break;

	default:
		printf("Dude no such barrier type! \n");
		break;
	}
}

void Barrier::renderSphere()
{
	glPushMatrix();
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, _material);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, _material);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, _material);
		glTranslatef(center[0], center[1], center[2]);
		glutSolidSphere(fSize, 20, 20);			
	glPopMatrix();
}

void Barrier::renderCube()
{
	glPushMatrix();
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, _material);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, _material);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, _material);
		glTranslatef(center[0], center[1], center[2]);
		glutSolidCube(fSize);			
	glPopMatrix();
}