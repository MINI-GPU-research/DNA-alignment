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
	while (tid < dataLength - MerLength)
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


template<int MerLenght, int HashLength>
string indexToString(int index)
{
	string result = "";

	for(int i = 0; i < HashLength; ++i)
	{
		switch(index & 3)
		{
		case 0:
		{
			result+="A";
			break;
		}
		case 1:
		{
			result+="C";
			break;
		}
		case 2:
		{
			result+="T";
			break;
		}
		case 3:
		{
			result+="G";
			break;
		}
		}
	}
	return result;
}

template<int MerLength, int HashLength>
void PrintResultInternal(uint* tree, int index, string s, int i)
{
	if(i == MerLength)
	{
		if(tree[index]) cout << s << "A " << tree[index] << endl;
		if(tree[index+1]) cout << s << "C " << tree[index + 1] << endl;
		if(tree[index+2]) cout << s << "T " << tree[index + 2] << endl;
		if(tree[index+3]) cout << s << "G " << tree[index + 3] << endl;
	}
	else
	{
		++i;
		if(tree[index] != 0)
		{
			PrintResultInternal<MerLength, HashLength>(tree, tree[index], s+"A", i);
		}
		if(tree[index+1] != 0)
		{
			PrintResultInternal<MerLength, HashLength>(tree, tree[index+1], s+"C", i);
		}
		if(tree[index+2] != 0)
		{
			PrintResultInternal<MerLength, HashLength>(tree, tree[index+2], s+"T", i);
		}
		if(tree[index+3] != 0)
		{
			PrintResultInternal<MerLength, HashLength>(tree, tree[index+3], s+"G", i);
		}

	}
}

template<int MerLength, int HashLength>
void PrintResult(uint* tree)
{
	const int size = 4 * (1 << (2*HashLength));
	for(int i = 0; i < size; i+=4)
	{
		string s = indexToString<MerLength, HashLength>(i/4);

		PrintResultInternal<MerLength, HashLength>(tree, i, s, HashLength);

	}
}

template<int MerLength, int HashLength>
void Init()
{

	char* data_d;
	uint dataLength_d;
	uint* tree_d;
	uint* treeLength_d;


	checkCudaErrors(cudaMalloc((void**)&data_d,  2048 * sizeof(char)));
	checkCudaErrors(cudaMemset(data_d, 'G', 2048 * sizeof(char)));
	checkCudaErrors(cudaMemset(data_d, 'C', 1024 * sizeof(char)));
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

	uint* tree_h = new uint[4*2048];

	checkCudaErrors(cudaMemcpy((void*)tree_h, (void*)tree_d, 4*2048 * sizeof(uint), cudaMemcpyDeviceToHost));

	PrintResult<MerLength, HashLength>(tree_h);

	delete tree_h;

    checkCudaErrors(cudaFree(data_d));
    checkCudaErrors(cudaFree(tree_d));
    checkCudaErrors(cudaFree(treeLength_d));
}

