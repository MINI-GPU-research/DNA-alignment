################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/main.cu 

CPP_SRCS += \
../src/SimpleFastQReader.cpp 

OBJS += \
./src/SimpleFastQReader.o \
./src/main.o 

CU_DEPS += \
./src/main.d 

CPP_DEPS += \
./src/SimpleFastQReader.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -I"/usr/local/cuda-10.2/samples/0_Simple" -I"/usr/local/cuda-10.2/samples/common/inc" -I"/home/michal/cuda-workspace/Testing" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -I"/usr/local/cuda-10.2/samples/0_Simple" -I"/usr/local/cuda-10.2/samples/common/inc" -I"/home/michal/cuda-workspace/Testing" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -I"/usr/local/cuda-10.2/samples/0_Simple" -I"/usr/local/cuda-10.2/samples/common/inc" -I"/home/michal/cuda-workspace/Testing" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -I"/usr/local/cuda-10.2/samples/0_Simple" -I"/usr/local/cuda-10.2/samples/common/inc" -I"/home/michal/cuda-workspace/Testing" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


