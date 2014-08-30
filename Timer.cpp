#include "Timer.h"

#include <time.h>

Timer::Timer()
	:_nLastTick(0)
{ }

unsigned Timer::GetInterval()
{
	if(_nLastTick == 0)
	{
		_nLastTick = clock();
		return 0;
	}

	unsigned nCurrTick = clock();
	unsigned nInterval = nCurrTick - _nLastTick;
	_nLastTick = nCurrTick;
	
	return nInterval;
}