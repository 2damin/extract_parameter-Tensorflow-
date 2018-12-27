#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

void changeDataType(float* srcData, float* dstData, int n, int c, int h, int w);