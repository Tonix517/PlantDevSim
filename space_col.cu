#include "space_col.h"
#include "Config.h"
#include "common.h"

#ifdef __DEVICE_EMULATION__
	#include <stdio.h>
	static float SunVec[3] = SUN_VEC;
#else
	static __device__ float SunVec[3] = SUN_VEC;
#endif

///
///		Some Utils functions
///

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

__device__
void vecAdd(float vec1[3], float vec2[3], float ret[3])
{
	ret[0] = vec1[0] + vec2[0];
	ret[1] = vec1[1] + vec2[1];
	ret[2] = vec1[2] + vec2[2];
}

__device__
void vecScale(float vec1[3], float fScale, float ret[3])
{
	ret[0] = vec1[0] * fScale;
	ret[1] = vec1[1] * fScale;
	ret[2] = vec1[2] * fScale;
}

__device__
float len(float vec[3])
{
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

__device__
void normalize(float vec[3])
{
	float fLen = len(vec);
	if(fLen > 0)
	{
		vec[0] /= fLen;
		vec[1] /= fLen;
		vec[2] /= fLen;
	}
}

__device__
float dot_product(float vec1[3], float vec2[3])
{
	return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
}

__device__
float getAngle(float vec1[3], float vec2[3])
{
	return acos( dot_product(vec1, vec2) / (len(vec1) * len(vec2))) * 180.f / PI;
}

__device__
Cell* getCell( unsigned xi, unsigned yi, unsigned zi)
{
	return _mat + xi * Y_Dim * Z_Dim + yi * Z_Dim + zi;
}

__global__
void gpuInit(void *pMat)
{
	_mat = (Cell *)pMat;

	unsigned xi = 0, yi = 0, zi = 0;
	getCurrentCellInx(&xi, &yi, &zi);

	if( (xi < X_Dim) &&
		(yi < Y_Dim) &&
		(zi < Z_Dim) )
	{
		getCell(xi, yi, zi)->bOccupied = false;
		getCell(xi, yi, zi)->nBudId = -1;
		getCell(xi, yi, zi)->fIllum = 1;
	}
}


///
///		Space Colonization
///

__global__
void space_colonization_on_voxels(InternodeDelegate *pBuf, unsigned nBudCount)
{
	unsigned ix = 0, iy = 0, iz = 0;
	getCurrentCellInx(&ix, &iy, &iz);

	if( (ix < X_Dim) &&
		(iy < Y_Dim) &&
		(iz < Z_Dim) )
	{

		float pos[3] = {0};
		pos[0] = ix * SpaceGranularity + (-SCENESIZE);
		pos[1] = iy * SpaceGranularity + (SCENEHEIGHT);
		pos[2] = iz * SpaceGranularity + (-SCENESIZE);

		if( getCell(ix, iy, iz)->bOccupied )
		{
			return;
		}

		//	1. Remove the Occupied Voxels
		//	
		for(int i = 0; i < nBudCount; i ++)
		{		
			//	Terminal Bud
			float fDist = dist(pBuf[i].budPos, pos);
			if(fDist <= OccuRadius)
			{
				getCell(ix, iy, iz)->bOccupied = true;
				return;
			}		

			//	Anxillary Bud
			fDist = dist(pBuf[i].anBudPos, pos);
			if(fDist <= OccuRadius)
			{
				getCell(ix, iy, iz)->bOccupied = true;
				return;
			}		
		}

		//	2. Assign the Perception Voxels
		//
		int nMinBudId = -1; float fMinDist = 0xFFFFFF;

		for(int i = 0; i < nBudCount; i ++)
		{
			//	angle?
			float vCurrVec[3] = {0};
			vCurrVec[0] = ix * SpaceGranularity + (-SCENESIZE) - pBuf[i].budPos[0];
			vCurrVec[1] = iy * SpaceGranularity + (SCENEHEIGHT) - pBuf[i].budPos[1];
			vCurrVec[2] = iz * SpaceGranularity + (-SCENESIZE) - pBuf[i].budPos[2];

			float pOrtVec[3];
			memcpy(pOrtVec, pBuf[i].budOrt, sizeof(float) * 3);

			float angle = getAngle(vCurrVec, pOrtVec);
			if( (angle < 0)  || ( angle > PerceptionAngle/2 ) )
			{
				continue;
			}		

			//
			float fDist = dist(pBuf[i].budPos, pos);
			if( (fDist <= PerceptionRadius) && (fDist < fMinDist) )
			{
				nMinBudId = i;
				fMinDist = fDist;
			}		
		}
		if(nMinBudId != -1)
		{
			getCell(ix, iy, iz)->nBudId = nMinBudId;
		}
	}
}


///
///		Shadow Propagation Calculation
///

__device__
int getCoveredLayer(float cellPos[3], float pointPos[3])
{
	//	Delta-Y is not between [0, MaxShadowLayer)?
	int nDeltaLayer = (pointPos[1] - cellPos[1]) / SpaceGranularity;
	
	if( (cellPos[1] > pointPos[1]) ||
		(nDeltaLayer>= MaxShadowLayer)
		)
	{
		return -1;
	}

	//	get input Cells X & Z index
	int cellInx[3] = {0};
	cellInx[0] = (cellPos[0] - (-SCENESIZE))/ SpaceGranularity;
	cellInx[2] = (cellPos[2] - (-SCENESIZE))/ SpaceGranularity;

	int posInx[3] = {0};
	posInx[0] = (pointPos[0] - (-SCENESIZE))/ SpaceGranularity;
	posInx[2] = (pointPos[2] - (-SCENESIZE))/ SpaceGranularity;

	//	get current Center Cell index
	float ratio = abs(SpaceGranularity / SunVec[1]);
	unsigned ctrInx[3] = {0};
	ctrInx[0] = posInx[0] + ratio * SunVec[0] / SpaceGranularity * nDeltaLayer;
	ctrInx[2] = posInx[2] + ratio * SunVec[2] / SpaceGranularity * nDeltaLayer;
	
	int nDeltaX = (cellInx[0] - ctrInx[0]);
	int nDeltaZ = (cellInx[2] - ctrInx[2]);
	//	Delta-X\Delta-Z with the center out of bound?
	if( abs(nDeltaX) > nDeltaLayer ||
		abs(nDeltaZ) > nDeltaLayer )
	{
		return -1;
	}
	
	return nDeltaLayer;
}

__device__
float getShadowValue(int nLayer)
{
	//	const from the paper
	const float a = 0.008;
	const float b = 1.2;

	return a * pow(b, (float)-nLayer);
}

__global__
void shadow_calculation_on_voxels(InternodeDelegate *pBuf, unsigned nInternodeCount)
{
	unsigned xi = 0, yi = 0, zi = 0;
	getCurrentCellInx(&xi, &yi, &zi);

	if( (xi < X_Dim) &&
		(yi < Y_Dim) &&
		(zi < Z_Dim) )
	{
		float fCellPos[3] = {0};
		fCellPos[0] = xi * SpaceGranularity + (-SCENESIZE);
		fCellPos[1] = yi * SpaceGranularity + (SCENEHEIGHT);
		fCellPos[2] = zi * SpaceGranularity + (-SCENESIZE);

		int nLayer = 0; 
		for(int i = 0; i < nInternodeCount; i ++)
		{
			InternodeDelegate *pCurrInd = pBuf + i;

 			//	Internode

			if( (nLayer = getCoveredLayer(fCellPos, pCurrInd->pos)) != -1 )
			{
				if(getCell(xi, yi, zi)->fIllum > 0)
				{
					getCell(xi, yi, zi)->fIllum -= getShadowValue(nLayer);
					if(getCell(xi, yi, zi)->fIllum < 0)
					{
						getCell(xi, yi, zi)->fIllum = 0;
					}
				}
			}

			float pTmp[3] = {0};
			
			vecAdd(pCurrInd->pos, pCurrInd->vec, pTmp);
			if( (nLayer = getCoveredLayer(fCellPos, pTmp)) != -1)
			{
				if(getCell(xi, yi, zi)->fIllum > 0)
				{
					getCell(xi, yi, zi)->fIllum -= getShadowValue(nLayer);
					if(getCell(xi, yi, zi)->fIllum < 0)
					{
						getCell(xi, yi, zi)->fIllum = 0;
					}
				}
			}

			vecScale(pCurrInd->vec, 0.5, pTmp);
			vecAdd(pCurrInd->pos, pTmp, pTmp);
			if( (nLayer = getCoveredLayer(fCellPos, pTmp)) != -1)
			{
				if(getCell(xi, yi, zi)->fIllum > 0)
				{
					getCell(xi, yi, zi)->fIllum -= getShadowValue(nLayer);
					if(getCell(xi, yi, zi)->fIllum < 0)
					{
						getCell(xi, yi, zi)->fIllum = 0;
					}
				}
			}

			//	Leaf
			if( pCurrInd->leafPos[0] != INVALID_POS)
			if( (nLayer = getCoveredLayer(fCellPos, pCurrInd->leafPos)) != -1 )
			{
				if(getCell(xi, yi, zi)->fIllum > 0)
				{
					getCell(xi, yi, zi)->fIllum -= getShadowValue(nLayer);
					if(getCell(xi, yi, zi)->fIllum < 0)
					{
						getCell(xi, yi, zi)->fIllum = 0;
					}
				}
			}

		}//	for
	}// if
}


///
///		Get Optimized Vector
///
__global__
void get_optimized_bud_vector(InternodeDelegate *pBuf, struct OptVec *pOptVecs)
{
	unsigned xi = 0, yi = 0, zi = 0;
	getCurrentCellInx(&xi, &yi, &zi);

	if( (xi < X_Dim) &&
		(yi < Y_Dim) &&
		(zi < Z_Dim) )
	{
		int id = getCell(xi, yi, zi)->nBudId;
		if( (id != -1))
		{
			//	TODO: synchronized
			float pBudPos[3];
			memcpy(pBudPos, pBuf[id].budPos, sizeof(float) * 3);

			float vCurrVec[3] = {0};
			vCurrVec[0] = xi * SpaceGranularity + (-SCENESIZE) - pBudPos[0];
			vCurrVec[1] = yi * SpaceGranularity + (SCENEHEIGHT) - pBudPos[1];
			vCurrVec[2] = zi * SpaceGranularity + (-SCENESIZE) - pBudPos[2];

			{
				normalize(vCurrVec);
				
				pOptVecs[id].vec[0] += vCurrVec[0];
				pOptVecs[id].vec[1] += vCurrVec[1];
				pOptVecs[id].vec[2] += vCurrVec[2];
			}		
		}
	}
}
