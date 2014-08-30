#ifndef TIMER_H
#define TIMER_H

class Timer
{
protected:
	
	Timer();

	//	in milli-seconds
	unsigned GetInterval();

protected:
	unsigned _nLastTick;
};

#endif