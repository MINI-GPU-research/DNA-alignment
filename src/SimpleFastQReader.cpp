#include<fstream>
#include<iostream>
#include<string>

class SimpleFastQReader{
public:
	SimpleFastQReader(std::string path_to_file){
		file.open(path_to_file.c_str(),std::ios::in);
	}

	std::string ReadNextGenome(){
		std::string check;
		if(file.is_open() && !file.eof()){
			if(!std::getline(file,metadata))
				return {};
			if(!std::getline(file,genome))
				return {};
			if(!std::getline(file,check))
				return {};
			if(!std::getline(file,data_weight))
				return {};
			if(check != "+")
				return {};
			return genome;
		}
		std::cout<< "something went wrong, file cannot be read...";
		return {};
	}
	std::string ReadCurrentGenome(){
		return genome;
	}

	std::string ReadCurrentMetadata(){
		return metadata;
	}

	std::string ReadCurrentDataWeight(){
		return data_weight;
	}

	bool Eof(){
		if(file.is_open())
			return file.eof();
		return false;
	}

	~SimpleFastQReader(){
		file.close();
	}


private:
	std::ifstream file;
	std::string metadata;
	std::string genome;
	std::string data_weight;
};
