/****** head.h*************
 * A very simple Shared Ptr
 * Written by Riching Lee
 * 2008-1-1
 */

#pragma once 

#include <iostream>
using namespace std;

template<class T>
struct CorePtr{  
    
    T*    realPtr;            //The real pointer pointing to an object
    int     refCnt;            //Record how many "pointers" shares it.
};

template<class T>
class SharedPtr{
public:
    SharedPtr(){            
        pCore  = new CorePtr<T>;
        pCore->realPtr = NULL; 
        pCore->refCnt = 0;
    };
    SharedPtr(T* ptr){
        pCore  = new CorePtr<T>;
        pCore->realPtr = ptr;
        pCore->refCnt = 1;
    }
    SharedPtr(const SharedPtr<T>& sp){//Share the core ptr with sp, no need to create a new one
        pCore = sp.pCore;
        pCore->refCnt++;
    }
    ~SharedPtr(){
        releasePtr();    
    }

	void reset(T* ptr)
	{
		releasePtr();
		pCore  = new CorePtr<T>;
        pCore->realPtr = ptr;
        pCore->refCnt = 1;
	}

        void releasePtr()throw() //Used when to release a shared ptr
        {
            if(pCore){
                if(pCore->refCnt == 1)
                {
                    delete pCore->realPtr;
                    delete pCore;            //Don''t forget to free pCore
                }
                else if(pCore->refCnt == 0)
                    delete pCore;
                else
                    pCore->refCnt--;
            }
        }

        //Overload dereference
        T*     operator->(){return pCore->realPtr;}
        T&    operator*(){return *(pCore->realPtr);}
        
        //Copy assignmetn
        void operator=(const SharedPtr<T>& sp)
        {
            releasePtr();
            pCore = sp.pCore;
            pCore->refCnt++;
        //    return *(pCore->realPtr);
        }

		bool operator ==(const T* p)
		{
			return (pCore->realPtr) == p;
		}

		bool operator !=(const T* p)
		{
			return (pCore->realPtr) != p;
		}

private:
    CorePtr<T>* pCore;
};