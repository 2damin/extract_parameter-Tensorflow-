#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "changeDatatype.cuh"

using namespace std;

__global__ void changeType(float* srcData, float* dstData, int n, int c, int h, int w, int filtersPerThread_x, int filtersPerThread_y) {
	const int idxCol = threadIdx.y + blockDim.y*blockIdx.y;
	const int idxRow = threadIdx.x + blockDim.x*blockIdx.x;
	int maxBlock = (n * c) / (filtersPerThread_x * filtersPerThread_y);
	int idxBlock = (int)fminf((float)(blockIdx.y * gridDim.x + blockIdx.x), (float)(maxBlock));

	const int idxfilterW = threadIdx.x % w;
	const int idxfilterH = threadIdx.y % h;
	int threadChannelX = threadIdx.x / w;
	int threadChannelY = threadIdx.y / h;
	int idxChannel_a =idxBlock * filtersPerThread_x * filtersPerThread_y + threadChannelY *filtersPerThread_x + threadChannelX;
	int idxChannel = idxChannel_a % c;
	int idxN = (int)fminf((float)(idxChannel_a / c), (float)(n-1));	

	dstData[idxN * (c * w* h) + idxChannel * (w*h) + idxfilterH * w + idxfilterW] = srcData[idxfilterH * (n * c * w) + idxfilterW * (c * n) + idxChannel * n + idxN];

}


void changeDataType(float* srcData, float* dstData, int n, int c, int h, int w) {
	
	int filtersPerThread_x = 30 / w;
	int filtersPerThread_y = 30 / h;

	int totalBlocks = (c * n) / (filtersPerThread_x * filtersPerThread_y) + 1;
	int numBlock_y = totalBlocks / 255 + 1;

	dim3 numOfBlocks(255, numBlock_y, 1);
	dim3 threadsPerBlock(30, 30, 1);
	changeType <<< numOfBlocks, threadsPerBlock >> > (srcData, dstData, n, c, h, w, filtersPerThread_x, filtersPerThread_y);
}