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

using namespace std;
#define FirstLetters 1
#define DEVICE_TREE_SIZE 1024*1024*128-1

template<int prefixLen>
void Test(string s, uint* tree_h, uint* treeLength_h)
{
	EdgeCounter<12 + prefixLen,6,1,prefixLen> ec(s, tree_h, treeLength_h);
	cout << prefixLen << endl;
	for(int i=0; i < (1 << (1*2)); ++i)
	{
		SimpleFastQReader sfqr ("/home/bartek/Downloads/chr1.fastq");

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

int main(int argc, char ** argv)
{
	string s = "ACATATATGATAGACAATGACATAGACAGATACCGAGATAGACAGATAGACCAGACTGCTAGACTAGACCATGAGAGACTTAGACAGATATAGACCATATTTAGAGAG";
	uint* tree_h = new uint[(1 << (2 * FirstLetters)) * DEVICE_TREE_SIZE];
	uint* treeLength_h = new uint[(1 << (2 * FirstLetters))];
	Test<0>(s, tree_h,treeLength_h);
	Test<1>(s, tree_h,treeLength_h);
	Test<2>(s, tree_h,treeLength_h);
	Test<3>(s, tree_h,treeLength_h);
	Test<4>(s, tree_h,treeLength_h);
	Test<5>(s, tree_h,treeLength_h);
	Test<6>(s, tree_h,treeLength_h);
	Test<7>(s, tree_h,treeLength_h);
	Test<8>(s, tree_h,treeLength_h);
	Test<9>(s, tree_h,treeLength_h);
	Test<10>(s, tree_h,treeLength_h);
	Test<11>(s, tree_h,treeLength_h);
	Test<12>(s, tree_h,treeLength_h);
	Test<13>(s, tree_h,treeLength_h);
	Test<14>(s, tree_h,treeLength_h);
	Test<15>(s, tree_h,treeLength_h);
	Test<16>(s, tree_h,treeLength_h);
	Test<17>(s, tree_h,treeLength_h);
	Test<18>(s, tree_h,treeLength_h);
	Test<19>(s, tree_h,treeLength_h);
	Test<20>(s, tree_h,treeLength_h);
	Test<21>(s, tree_h,treeLength_h);
	Test<22>(s, tree_h,treeLength_h);
	Test<23>(s, tree_h,treeLength_h);
	Test<24>(s, tree_h,treeLength_h);
	Test<25>(s, tree_h,treeLength_h);
	Test<26>(s, tree_h,treeLength_h);

	delete[] tree_h;
	delete[] treeLength_h;
}

