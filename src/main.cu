#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <random>
#include <float.h>
#include "/home/bartek/cuda/nanopore/src/EdgeCounter.cuh"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>



#define NOT_ONE_BIT_SET(x) (x & (x - 1))

#define uint unsigned int
#define ull unsigned long long





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
	EdgeCounter<5,3> ec;
	char* line = new char[2300];
	memset(line, 'A', 2300);
	ec.AddLine(line, 2300);
	ec.Result();
	ec.PrintResult();
}



/*
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
	checkCudaErrors(cudaMemset(data_d, 'T', 1023 * sizeof(char)));
	checkCudaErrors(cudaMemset(data_d, 'A', 1022 * sizeof(char)));
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
	cout << GetEdgeWeight<MerLength, HashLength>(tree_h, "ATCG") << endl;
	delete tree_h;

    checkCudaErrors(cudaFree(data_d));
    checkCudaErrors(cudaFree(tree_d));
    checkCudaErrors(cudaFree(treeLength_d));
}
*/

