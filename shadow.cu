#include "shadow.h"

#include <assert.h>

MARK *h_ground_shadow;
MARK *d_ground_shadow;

MARK *h_smoothed_shadow;
MARK *d_smoothed_shadow;

const unsigned GroundDim = SCENESIZE * 2/ SpaceGranularity;

void reset_ground_shadow()
{
	assert(h_ground_shadow);
	assert(d_ground_shadow);
	assert(h_smoothed_shadow);
	assert(d_smoothed_shadow);

	memset(h_ground_shadow, 0, sizeof(MARK) * GroundDim * GroundDim);
	cudaMemset(d_ground_shadow, 0, sizeof(MARK) * GroundDim * GroundDim);

	memset(h_smoothed_shadow, 0, sizeof(MARK) * pow(SCENESIZE * 2 / SMOOTH_GRAN, 2) );
	cudaMemset(d_smoothed_shadow, 0, sizeof(MARK) * pow(SCENESIZE * 2 / SMOOTH_GRAN, 2));
}



///
__device__
int MAX_X_INX = (SCENESIZE * 2) / SpaceGranularity - 1;

__device__
int MAX_Y_INX = (DOMAIN_HEIGHT - SCENEHEIGHT) / SpaceGranularity - 1;

__device__
int MAX_Z_INX = (SCENESIZE * 2) / SpaceGranularity - 1;

__device__
bool isOver(int x, int y, int z)
{
	if( (x < 0 || y < 0 || z < 0) || 
		(x > MAX_X_INX || y > MAX_Y_INX || z > MAX_Z_INX) )
	{
		return true;
	}

	return false;
}

///
__device__
Cell* getCell( unsigned xi, unsigned yi, unsigned zi)
{
	//	haha...
	return _mat + xi * Y_Dim * Z_Dim + yi * Z_Dim + zi;
}

///
///		This piece of code is from Dr. Benes
///		Some changes are made.
///
__device__
float dda_ray_casting(int fromInx[3], int toInx[3])
{

	float fShadowValue = 0;

	int deltaX = toInx[0] - fromInx[0];
	int deltaY = toInx[1] - fromInx[1];
	int deltaZ = toInx[2] - fromInx[2];

	if ( abs(deltaX) >= abs(deltaY) && 
  	     abs(deltaX) >= abs(deltaZ) ) // delta X
	{
		double my = deltaY * 1.f / deltaX;
		double mz = deltaZ * 1.f / deltaX;
		double y = fromInx[1];
		double z = fromInx[2];

		for (int i = fromInx[0]; i < toInx[0]; i++)
		{			
			//	check inx bounds
			if(isOver(i, (int)y, (int)z))
			{
				return fShadowValue;
			}

			fShadowValue += 1 - getCell(i, (int)y, (int)z)->fIllum;
			
			y += my;
			z += mz;
		}
	}
	else if( abs(deltaY) >= abs(deltaX) && 
			 abs(deltaY) >= abs(deltaZ) ) //delta Y
	{
		double mx = deltaX * 1.f / deltaY;
		double mz = deltaZ * 1.f / deltaY;
		double x = fromInx[0];
		double z = fromInx[2];

		for (int j = fromInx[1]; j < toInx[1]; j++)
		{
			//	check inx bounds
			if(isOver((int)x, j, (int)z))
			{
				return fShadowValue;
			}

			fShadowValue += 1 - getCell((int)x, j, (int)z)->fIllum;

			x += mx;
			z += mz;
		}
	}
	else // delta Z
	{
		double mx = deltaX * 1.f / deltaZ;
		double my = deltaY * 1.f / deltaZ;
		double x = fromInx[0];
		double y = fromInx[1];
		for (int k = fromInx[2]; k < toInx[2]; k++)
		{
			//	check inx bounds
			if(isOver((int)x, (int)y, k))
			{
				return fShadowValue;
			}

			fShadowValue += 1 - getCell((int)x, (int)y, k)->fIllum;

			x += mx;
			y += my;
		}
	}

	return 0;
}

__device__
void getCurrentGroundCellInx( unsigned *pxi, unsigned *pyi)
{
	unsigned nAbsTid = blockIdx.x * blockDim.x + threadIdx.x;

	*pxi = nAbsTid % (unsigned)(SCENESIZE * 2 / SpaceGranularity);
	*pyi = nAbsTid / (SCENESIZE * 2 / SpaceGranularity);
}

__device__
void getCurrentSmoothedInx( unsigned *pxi, unsigned *pyi)
{
	unsigned nAbsTid = blockIdx.x * blockDim.x + threadIdx.x;

	*pxi = nAbsTid % (unsigned)(SCENESIZE * 2 / SMOOTH_GRAN);
	*pyi = nAbsTid / (SCENESIZE * 2 / SMOOTH_GRAN);
}

//	Kernel for calculating the shadow
//
__global__
void gpu_calc_shadow( MARK *pBuf, struct Cell *mat)
{
	//
	_mat = mat;

	//
	unsigned xi = 0, zi = 0;
	getCurrentGroundCellInx(&xi, &zi);
	if( (xi + 1) >  GroundDim || (zi + 1) >  GroundDim )
	{
		return;
	}

	//	Get From-Cell index
	int fromInx[3] = {xi, 0, zi};

	int nVerticalCount = (DOMAIN_HEIGHT - SCENEHEIGHT) / SpaceGranularity;
	int nHorizontalCount = (SCENESIZE * 2) / SpaceGranularity;

	//	View Vec
	float sun_vec[3] = SUN_VEC;
	float view_vec[3] = { -sun_vec[0], -sun_vec[1], -sun_vec[2] };
	//assert(view_vec[1] >= 0);

	//	Call To-Cell index
	int toInx[3] = {0};
	toInx[1] = nVerticalCount - 1;
	//	WARNING: the X\Z index may be over the available cube.
	//			 this will be checked in the dda_ray_casting()
	//
	toInx[0] = xi + (nVerticalCount - 1) * (view_vec[0] * 1.f / view_vec[1]);
	toInx[2] = zi + (nVerticalCount - 1) * (view_vec[2] * 1.f / view_vec[1]);
	
	//	Go casting !
	float fShadowValue = dda_ray_casting(fromInx, toInx);
	*(pBuf + zi * nHorizontalCount + xi) = fShadowValue;
}

__global__
void gpu_smooth_shadow( MARK *pBuf, MARK *pSmoothedShadow)
{
	//	Get cell inx first
	unsigned xi = 0, zi = 0;	
	getCurrentSmoothedInx(&xi, &zi);

	const int nSmoothedSize = (SCENESIZE * 2)/ SMOOTH_GRAN;
	if( (xi + 1) >  nSmoothedSize || (zi + 1) >  nSmoothedSize)
	{
		return;
	}

	const int nHorizontalCount = (SCENESIZE * 2) / SpaceGranularity;

	int xi_in_ori = xi * SMOOTH_GRAN / SpaceGranularity;
	int zi_in_ori = zi * SMOOTH_GRAN / SpaceGranularity;
	float deltaX = xi * SMOOTH_GRAN - xi_in_ori * SpaceGranularity;
	float deltaZ = zi * SMOOTH_GRAN - zi_in_ori * SpaceGranularity;

	// Bi-Linear Interpolation
	//  --------
	// | 1 | 2 |
	//  --------
	// | 3 | 4 |
	//  --------
	
	float fUpLeft = 0, fUpRight = 0, fDownLeft = 0, fDownRight = 0; 
	int iFieldNum = 0;

	///	TODO: Yes these if() are stupid enough for CUDA
	///		  there's a better way actually
	///
	if( deltaZ <= (SpaceGranularity / 2))
	{
		if( deltaX <= (SpaceGranularity / 2))	// Field 1
		{
			iFieldNum = 1;

			if(xi_in_ori > 0 && zi_in_ori > 0)
			{
				fUpLeft = *(pBuf + (zi_in_ori - 1) * nHorizontalCount + xi_in_ori - 1);
			}
			if(zi_in_ori > 0)
			{
				fUpRight = *(pBuf + (zi_in_ori - 1) * nHorizontalCount + xi_in_ori);
			}
			if(xi_in_ori > 0)
			{
				fDownLeft =*(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori - 1);
			}			

			fDownRight = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori);
		}
		else	// Field 2
		{
			iFieldNum = 2;

			if(zi_in_ori > 0)
			{
				fUpLeft = *(pBuf + (zi_in_ori - 1) * nHorizontalCount + xi_in_ori);
			}
			if(zi_in_ori > 0 && xi_in_ori < nHorizontalCount)
			{
				fUpRight = *(pBuf + (zi_in_ori - 1) * nHorizontalCount + xi_in_ori + 1);
			}
			
			fDownLeft = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori);
			
			if(xi_in_ori < nHorizontalCount)
			{
				fDownRight = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori + 1);
			}
		}
	}
	else 
	{
		if( deltaX <= (SpaceGranularity / 2) )	//	Field 3
		{
			iFieldNum = 3;

			if(xi_in_ori > 0)
			{
				fUpLeft = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori - 1);
			}
			
			fUpRight = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori);
			
			if(xi_in_ori > 0 && zi_in_ori < nHorizontalCount)
			{
				fDownLeft = *(pBuf + (zi_in_ori + 1) * nHorizontalCount + xi_in_ori - 1);
			}			
			if(zi_in_ori < nHorizontalCount)
			{
				fDownRight = *(pBuf + (zi_in_ori + 1) * nHorizontalCount + xi_in_ori);
			}
		}
		else	//	Field 4
		{
			iFieldNum = 4;

			fUpLeft = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori);

			if(xi_in_ori < nHorizontalCount)
			{
				fUpRight = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori + 1);
			}
			if(zi_in_ori < nHorizontalCount)
			{
				fDownLeft = *(pBuf + (zi_in_ori + 1) * nHorizontalCount + xi_in_ori);
			}
			if(xi_in_ori < nHorizontalCount && zi_in_ori < nHorizontalCount)
			{
				fDownRight = *(pBuf + (zi_in_ori + 1) * nHorizontalCount + xi_in_ori + 1);
			}
		}
	}

	//	All ZERO?
	const float EP = 0.005;
	if( abs(fUpLeft) < EP && 
		abs(fDownRight) < EP && 
		abs(fDownLeft) < EP && 
		abs(fUpRight) < EP )
	{
		*(pSmoothedShadow + zi * nSmoothedSize + xi) = 0;
		return;
	}

	//	Bi-Linear Interpolation
	const float HalfOriCellSize = (SpaceGranularity / 2);
	float fValue = 0;
	float ratioX =   deltaX > HalfOriCellSize ? 
					(deltaX - HalfOriCellSize) / SpaceGranularity : ( (deltaX + HalfOriCellSize) / SpaceGranularity );
	float ratioZ =   deltaZ > HalfOriCellSize ? 
					(deltaZ - HalfOriCellSize) / SpaceGranularity : ( (deltaZ + HalfOriCellSize) / SpaceGranularity );

	fValue = (1 - ratioX) * ( (1 - ratioZ) * fUpLeft + ratioZ * fDownLeft) + 
				   ratioX * ( (1 - ratioZ) * fUpRight + ratioZ * fDownRight) ;

	*(pSmoothedShadow + zi * nSmoothedSize + xi) = fValue;	
	//*(pSmoothedShadow + zi * nSmoothedSize + xi) = *(pBuf + zi_in_ori * nHorizontalCount + xi_in_ori);	
}