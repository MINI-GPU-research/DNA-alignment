
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>


#define uint unsigned int
#define ull unsigned long long
#define DATA_SIZE 2048

using namespace std;


__host__ __device__ int letterToInt(char c)
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

template<int MerLength, int HashLength, int NoBlocks>
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
			hash += letterToInt(data[tid + i]) << (2 * i);
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

		tid += blockDim.x * NoBlocks;
	}
}

template<int MerLength, int HashLength>
class EdgeCounter
{
public:
	char* data_d;
	uint* tree_d;
	uint* treeLength_d;
	uint* tree_h;

	EdgeCounter()
	{
		checkCudaErrors(cudaMalloc((void**)&data_d,  DATA_SIZE * sizeof(char)));
		checkCudaErrors(cudaMalloc((void**)&tree_d,  4*2048*sizeof(uint)));
		tree_h = new uint[4*2048];
		checkCudaErrors(cudaMemset(tree_d, 0, 2048 * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&treeLength_d,  sizeof(uint)));
		const uint startingTreeLength = 4 * (1 << (2*HashLength));
		checkCudaErrors(cudaMemcpy(treeLength_d, &startingTreeLength, sizeof(uint), cudaMemcpyHostToDevice));
	}

	~EdgeCounter()
	{
	    checkCudaErrors(cudaFree(data_d));
	    checkCudaErrors(cudaFree(tree_d));
	    checkCudaErrors(cudaFree(treeLength_d));
	    delete tree_h;
	}

	void AddLine(char* line, uint length)
	{
		for(int i = 0; i < length; i+=DATA_SIZE)
		{
			if(i - (MerLength - 1) > 0) i -= (MerLength - 1); // AAAA AAAA

			int len = length - i > DATA_SIZE  ? DATA_SIZE : length - i;
			checkCudaErrors(cudaMemset(data_d, 'G', 2048 * sizeof(char)));

			CountEdges<MerLength,HashLength, 1><<<1, 256>>>(
					data_d,
					len,
					tree_d,
					treeLength_d);

			cudaDeviceSynchronize();
			cudaError_t code = cudaGetLastError();
			if (code != cudaSuccess)
			{
				fprintf(stderr, "kernelAssert: %s\n", cudaGetErrorString(code));
				if (abort) exit(code);
			}
		}
	}

	void Result()
	{
		checkCudaErrors(cudaMemcpy((void*)tree_h, (void*)tree_d, 4*2048 * sizeof(uint), cudaMemcpyDeviceToHost));
	}

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
private:
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
				PrintResultInternal(tree, tree[index], s+"A", i);
			}
			if(tree[index+1] != 0)
			{
				PrintResultInternal(tree, tree[index+1], s+"C", i);
			}
			if(tree[index+2] != 0)
			{
				PrintResultInternal(tree, tree[index+2], s+"T", i);
			}
			if(tree[index+3] != 0)
			{
				PrintResultInternal(tree, tree[index+3], s+"G", i);
			}

		}
	}
public:
	void PrintResult()
	{
		const int size = 4 * (1 << (2*HashLength));
		for(int i = 0; i < size; i+=4)
		{
			string s = indexToString(i/4);

			PrintResultInternal(tree_h, i, s, HashLength);

		}
	}

	uint GetEdgeWeigthInternal(uint* tree, int index,string mer, int i)
	{
		if(i == MerLength)
		{
			return tree[index + letterToInt(mer[i])];
		}

		int c = letterToInt(mer[i]);
		if(tree[index + c] == 0) return 0;

		return GetEdgeWeigthInternal(tree, tree[index + c], mer, i+1);
	}

	uint GetEdgeWeight(uint* tree, string mer)
	{
		if(mer.length() != MerLength+1)
		{
			return 0;
		}

		ull hash = 0;
		for(int i=0; i < HashLength; ++i)
		{
			hash+=letterToInt(mer[i]) << (2 * i);
		}

		return GetEdgeWeigthInternal(tree, 4 * hash, mer, HashLength);
	}

};

