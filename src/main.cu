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
	EdgeCounter<20,8,8> ec;
	for(int i=0; i < (1 << (2*8)); ++i)
	{
		SimpleFastQReader sfqr ("/home/michal/chr1.fastq");

		int line_ = 0;
		while(!sfqr.Eof()){
			std::string sequence = sfqr.ReadNextGenome();
			cout << line_++ << endl;
			char *line = new char[sequence.size()+1];
			strcpy(line, sequence.c_str());
			ec.AddLineFirstLetters(line,sequence.size(), i);
			delete line;
		}
	}

	/*char* line = new char[5000];
	memset(line, 'A', 100);
	line[7]='T';
	ec.AddLine(line, 100);*/
	ec.Result();
	ec.PrintResult();
//	cout << "GGGGG -> GGGGGT " << ec.GetEdgeWeight("GGGGGT") << endl;
	//delete line;
}

