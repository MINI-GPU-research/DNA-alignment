#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <random>
#include <float.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>



#define NOT_ONE_BIT_SET(x) (x & (x - 1))

#define uint unsigned int
#define ull unsigned long long



__device__ int letterToInt(char c)
{
	switch(c)
	{
		case 'A':
		{
			return 0;
		}
		case 'C':
		{
			return 1;
		}
		case 'T':
		{
			return 2;
		}
		case 'G':
		{
			return 3;
		}
	}
	return -1;
}

template<int MerLength, int HashLength, int noBlocks>
__global__ void CountEdges(
		char* data,
		uint dataLength,
		uint* tree,
		uint* treeLength
		)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < dataLength - MerLength - 1)
	{
		ull hash = 0;
		int i = 0;
		for(; i<HashLength; ++i)
		{
			hash += letterToInt(data[tid + i]) << 2 * i;
		}

		uint currentNode = 4 * hash;

		while(i < MerLength)
		{
			int letterId = letterToInt(data[tid + i]);
			int nextNode = tree[currentNode + letterId];

			if (atomicCAS(tree + currentNode + letterId, 0, -1) == 0)
			{
				int newNode = atomicAdd(treeLength, 4);
				tree[currentNode + letterId] = newNode;
				currentNode = newNode;
				++i;
			}
			else if (nextNode != -1 && nextNode != 0)
			{
				currentNode = nextNode;
				++i;
			}
		}

		int lastLetter = letterToInt(data[tid + i]);
		if(lastLetter > -1)
		{
			atomicAdd(tree + currentNode + lastLetter, 1);
		}

		tid += blockDim.x * noBlocks;
	}
}

////////////////////////////////////////////////////////////////////////////////

using namespace std;


#define ERR(source) (perror(source),\
                     fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     exit(EXIT_FAILURE))

void usage(char* pname){
	fprintf(stderr,"USAGE:%s\n",pname);
	exit(EXIT_FAILURE);
}


template<int MerLength, int HashLength>
void Init();


int main(int argc, char ** argv)
{
	Init<3,1>();
}


template<int MerLength, int HashLength>
void Init()
{

	char* data_d;
	uint dataLength_d;
	uint* tree_d;
	uint* treeLength_d;


	checkCudaErrors(cudaMalloc((void**)&data_d,  2048 * sizeof(char)));
	checkCudaErrors(cudaMemset(data_d, 'A', 2048 * sizeof(char)));
	checkCudaErrors(cudaMalloc((void**)&tree_d,  4*2048*sizeof(uint)));
	checkCudaErrors(cudaMemset(tree_d, 0, 2048 * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&treeLength_d,  sizeof(uint)));
	const uint startingTreeLength = 4 * (1 << (2*HashLength));
	checkCudaErrors(cudaMemcpy(treeLength_d, &startingTreeLength, sizeof(uint), cudaMemcpyHostToDevice));



	CountEdges<MerLength,HashLength,1><<<1, 256>>>(data_d, 2048, tree_d, treeLength_d);
	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
	{
		fprintf(stderr, "kernelAssert: %s\n", cudaGetErrorString(code));
		if (abort) exit(code);
	}

    checkCudaErrors(cudaFree(data_d));
    checkCudaErrors(cudaFree(tree_d));
    checkCudaErrors(cudaFree(treeLength_d));
}

