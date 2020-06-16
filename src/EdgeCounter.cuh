
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
#include <cmath>


#define uint unsigned int
#define ull unsigned long long
#define DATA_SIZE 4096

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

template<int MerLength, int HashLength, int FirstLettersCut, int FirstLettersRestrictionLength>
__global__ void CountEdges(
		char* data,
		uint dataLength,
		uint* tree,
		uint* treeLength,
		uint startingLetters,
		long long int firstLettersRestriction
		)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < dataLength - (MerLength))
	{
		ull hash = 0;
		int i = 0;
		for(; i < FirstLettersRestrictionLength; ++i)
		{
			if ((firstLettersRestriction & 3) != letterToInt(data[tid + i]))
			{
				return;
			}

			firstLettersRestriction = firstLettersRestriction >> 2;
		}
		for(; i < FirstLettersCut+FirstLettersRestrictionLength; ++i)
		{
			if ((startingLetters & 3) != letterToInt(data[tid + i]))
			{
				return;
			}

			startingLetters = startingLetters >> 2;
		}
		for(; i<(HashLength + FirstLettersCut + FirstLettersRestrictionLength); ++i)
		{
			hash += letterToInt(data[tid + i]) << (2 * (i - FirstLettersCut - FirstLettersRestrictionLength));
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
		__syncthreads();
	}
}
#define DEVICE_TREE_SIZE 1024*1024*128-1
template<int MerLength, int HashLength, int FirstLetters, int FirstLettersRestrictionLength>
class EdgeCounter
{
public:
	char* data_d;
	uint* tree_d;
	uint* treeLength_d;

	uint* tree_h;
	uint* treeLength_h;

	int currentDeviceTree;

	long long int letterRestriction_h = 0;

	EdgeCounter(string lettersRestrictions, uint* tree, uint* treeLen)
	{
		checkCudaErrors(cudaMalloc((void**)&data_d,  DATA_SIZE * sizeof(char)));
		checkCudaErrors(cudaMalloc((void**)&tree_d,  DEVICE_TREE_SIZE*sizeof(uint)));
		tree_h = tree;
		treeLength_h = treeLen;
		checkCudaErrors(cudaMemset(tree_d, 0, DEVICE_TREE_SIZE * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&treeLength_d,  sizeof(uint)));
		const uint startingTreeLength = 4 * (1 << (2*HashLength));
		checkCudaErrors(cudaMemcpy(treeLength_d, &startingTreeLength, sizeof(uint), cudaMemcpyHostToDevice));

		for(int i = 0; i < (1 << (2 * FirstLetters)); ++i)
		{
			treeLength_h[i] = startingTreeLength;
		}
		for(int i = 0; i < ((1 << (2 * FirstLetters)) * DEVICE_TREE_SIZE); ++i)
		{
			tree_h[i]=0;
		}

		for(int i = 0; i < FirstLettersRestrictionLength; ++i)
		{
			letterRestriction_h += letterToInt(lettersRestrictions[i]) << (2*i);
		}
		currentDeviceTree = 0;
	}

	~EdgeCounter()
	{
	    checkCudaErrors(cudaFree(data_d));
	    checkCudaErrors(cudaFree(tree_d));
	    checkCudaErrors(cudaFree(treeLength_d));
	}


	void AddLineFirstLetters(char* line, uint length, int fl)
	{
		if(currentDeviceTree != fl)
		{
			checkCudaErrors(cudaMemcpy((void*)(tree_h + (currentDeviceTree * DEVICE_TREE_SIZE)), (void*)tree_d, DEVICE_TREE_SIZE * sizeof(uint), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy((void*)(treeLength_h + currentDeviceTree), (void*)treeLength_d, sizeof(uint), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(treeLength_d, treeLength_h + fl, sizeof(uint), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(tree_d, (tree_h + (fl * DEVICE_TREE_SIZE)) , DEVICE_TREE_SIZE*sizeof(uint), cudaMemcpyHostToDevice));
			currentDeviceTree = fl;
		}

		for(int i = 0; i < length; i+=DATA_SIZE)
		{
			if(i - ((MerLength + FirstLetters) - 1) > 0) i -= ((MerLength + FirstLetters) - 1); // AAAA AAAA

			int len = length - i > DATA_SIZE  ? DATA_SIZE : length - i;
			checkCudaErrors(cudaMemcpy(data_d, line + i, len*sizeof(char), cudaMemcpyHostToDevice));

			CountEdges<MerLength,HashLength,FirstLetters,FirstLettersRestrictionLength><<<ceil(static_cast<float>(len)/256), 256>>>(
					data_d,
					len,
					tree_d,
					treeLength_d,
					fl,
					letterRestriction_h);

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
		checkCudaErrors(cudaMemcpy((void*)(tree_h + (currentDeviceTree * DEVICE_TREE_SIZE)), (void*)tree_d, DEVICE_TREE_SIZE * sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void*)(treeLength_h + currentDeviceTree), (void*)treeLength_d, sizeof(uint), cudaMemcpyDeviceToHost));
	}

	string indexToString(int index, int no)
	{
		string result = "";

		for(int i = 0; i < no; ++i)
		{
			switch((index >> (2*i)) & 3 )
			{
			case 0:
			{
				result +="A";
				break;
			}
			case 1:
			{
				result += "C";
				break;
			}
			case 2:
			{
				result += "T";
				break;
			}
			case 3:
			{
				result += "G";
				break;
			}
			default:
			{
				result = "?" + result;
				break;
			}
			}
		}
		return result;
	}
	uint edgesCount = 0;
	uint maximumEdgeWeight = 0;
	uint cumulativeEdgeWeighht = 0;
	uint existingNodes = 0;
private:
	void PrintResultInternal(uint* tree, int index, string s, int i, bool print = false)
	{
		if(i == (MerLength - FirstLetters - FirstLettersRestrictionLength))
		{
			if(tree[index] || tree[index + 1] || tree[index + 2] || tree[index+3]) existingNodes+=1;
			if(tree[index]) {
				if(print) cout << s << "A " << tree[index] << endl;
				++edgesCount;
				if(tree[index] > maximumEdgeWeight) maximumEdgeWeight = tree[index];
				cumulativeEdgeWeighht += tree[index];
			}
			if(tree[index+1]) {
				if(print) cout << s << "C " << tree[index + 1] << endl;
				++edgesCount;
				if(tree[index + 1] > maximumEdgeWeight) maximumEdgeWeight = tree[index + 1];
				cumulativeEdgeWeighht += tree[index + 1];
			}
			if(tree[index+2]) {
				if(print) cout << s << "T " << tree[index + 2] << endl;
				++edgesCount;
				if(tree[index + 2] > maximumEdgeWeight) maximumEdgeWeight = tree[index + 2];
				cumulativeEdgeWeighht += tree[index + 2];
			}
			if(tree[index+3]) {
				if(print) cout << s << "G " << tree[index + 3] << endl;
				++edgesCount;
				if(tree[index + 3] > maximumEdgeWeight) maximumEdgeWeight = tree[index + 3];
				cumulativeEdgeWeighht += tree[index + 3];
			}
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
	void PrintResult(bool printAllEdges = false)
	{
		for(int fl = 0; fl < (1 << (2 * FirstLetters)); ++fl)
		{
			string ss =  indexToString(fl, FirstLetters);
			const int size = 4 * (1 << (2*HashLength));
			for(int i = 0; i < size; i+=4)
			{
				string s = indexToString(i/4, HashLength);

				PrintResultInternal(tree_h + (fl * DEVICE_TREE_SIZE), i, ss + s, HashLength,printAllEdges);
			}
		}
		uint sumLength = 0;
		for(int fl = 0; fl < (1 << (2 * FirstLetters)); ++fl)
		{
			sumLength += treeLength_h[fl];
		}
		cout << "avg edge weight:" << (double)cumulativeEdgeWeighht/edgesCount << endl;
		cout << "maximumEdgeWeight" << maximumEdgeWeight << endl;
		cout << "edges count:    " << edgesCount << endl;
		cout << "edges %:        " << (double)edgesCount / pow(4, (1 + MerLength - FirstLettersRestrictionLength)) << endl;
	}
private:
	uint GetEdgeWeigthInternal(uint* tree, int index,string mer, int i)
	{
		if(i == (MerLength - FirstLettersRestrictionLength))
		{
			return tree[index + letterToInt(mer[i])];
		}

		int c = letterToInt(mer[i]);
		if(tree[index + c] == 0) return 0;

		return GetEdgeWeigthInternal(tree, tree[index + c], mer, i+1);
	}
public:
	uint GetEdgeWeight(string mer)
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

		return GetEdgeWeigthInternal(tree_h, 4 * hash, mer, HashLength);
	}

};

