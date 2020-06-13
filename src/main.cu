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
	if(argc < 2) {
		std::cout<<"No path to file, please provide one"<<std::endl;
		return 2;
	}
	EdgeCounter<16,8,2> ec;
	if(argc == 3) {
		std::cout<<"Caching of the file to RAM, considerably faster, but uses way more RAM \n";
		std::cout<<"Please use only if you have more free RAM than the size of processed file";
		std::vector<std::string> catched_file;
		SimpleFastQReader sfqr (argv[1]);
		while(!sfqr.Eof()){
			catched_file.push_back(sfqr.ReadNextGenome());
		}
		for(int i=0; i < (1 << (2*2)); ++i)
		{
			SimpleFastQReader sfqr (argv[1]);
			for (auto it = catched_file.begin(); it != catched_file.end(); it++)
			{
				std::string sequence = *it;
				char *line = new char[sequence.size()+1];
				strcpy(line, sequence.c_str());
				ec.AddLineFirstLetters(line,sequence.size(), i);
				delete line;
			}
		}
		ec.Result();
		ec.PrintResult();
	}
	else {
		std::cout<<"No caching of the file to RAM, considerably slower, but uses way less RAM";
		for(int i=0; i < (1 << (2*2)); ++i)
		{
			SimpleFastQReader sfqr (argv[1]);

			int line = 0;
			while(!sfqr.Eof()){
				std::string sequence = sfqr.ReadNextGenome();
				char *line = new char[sequence.size()+1];
				strcpy(line, sequence.c_str());
				ec.AddLineFirstLetters(line,sequence.size(), i);
				delete line;
			}
		}
		ec.Result();
		ec.PrintResult();
	}


}

