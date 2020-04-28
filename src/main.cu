#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
#include <random>
#include <float.h>
#include "EdgeCounter.cuh"
#include "SimpleFastQReader.cpp"

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
	SimpleFastQReader sfqr ("/home/michal/cuda-workspace/Testing/Debug/sequence.fastq");

	while(!sfqr.Eof()){
		std::string sequence = sfqr.ReadNextGenome();
		char *line = new char[sequence.size()+1];
		strcpy(line, sequence.c_str());
		ec.AddLine(line,sequence.size());
		delete line;
	}

//	char* line = new char[5000];
//	memset(line, 'A', 2300);
//	ec.AddLine(line, 2300);
//	memset(line, 'C', 1900);
//	ec.AddLine(line, 1900);
//	memset(line, 'T', 3300);
//	ec.AddLine(line, 3300);
//
//	memset(line, 'G', 1000);
//	memset(line, 'C', 500);
//	ec.AddLine(line, 1500);

	ec.Result();
	ec.PrintResult();
	cout << "GGGGG -> GGGGGT " << ec.GetEdgeWeight("GGGGGT") << endl;
//	delete line;
}

