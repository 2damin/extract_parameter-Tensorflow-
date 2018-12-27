#include <iostream>
#include <fstream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda.h>
#include "changeDatatype.cuh"

using namespace std;

void datTobin_outputdata(const char* datname, const char* binname, int input_channel, int width, int height, int output_channel)
{ //string srcname_dw, string srcname_db, string srcname_dg, string srcname_dmean, string srcname_dvar,
  //string dstname_dw, string dstname_db, string dstname_dg, string dstname_dmean, string dstname_dvar
	std::string srcname_dw = (std::string(datname) + std::string(".dat"));
	std::string dstname_dw = (std::string(binname) + std::string(".bin"));


	ofstream binFile_dw(dstname_dw, ios::binary);

	ifstream datFile_dw(srcname_dw, ios::in | ios::binary);


	if (!datFile_dw)
	{
		std::cout << "error opening file" << std::endl;
	}
	int size_depthW = input_channel*width*height*output_channel;


	float* data_dw = new float[size_depthW];

	float* dstData_dw = NULL;
	float* srcData_dw = NULL;

	memset(data_dw, 0, sizeof(float)*size_depthW);

	cudaMalloc(&dstData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&srcData_dw, sizeof(float)*size_depthW);

	cudaMemset(srcData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_depthW);

	for (int i = 0; i < size_depthW; i++)
	{
		//cout << data_dw[i] << endl;
		datFile_dw.is_open();
		datFile_dw >> data_dw[i];
		//binFile_dw.write(reinterpret_cast<const char*>(&data_dw[i]), sizeof(float));
	}

	//cudaMemcpy(srcData_dw, data_dw, sizeof(float)*size_depthW, cudaMemcpyHostToDevice);
	//changeDataType(srcData_dw, dstData_dw, output_channel, input_channel, height, width);

	//cout << "====dataChange====" << endl;
	//float* dsctmp = new float[size_depthW];
	//memset(dsctmp, 0, sizeof(float) * size_depthW);
	//cudaMemcpy(dsctmp, dstData_dw, sizeof(float)* size_depthW, cudaMemcpyDeviceToHost);
	
	//for (int n = 0; n < output_channel; n++) {
	//	for (int h = 0; h < height; h++) {
	//		for (int w = 0; w < width; w++) {
	//			for (int c = 0; c < input_channel; c++) {
	//				//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
	//				//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
	//				if (dsctmp[n* input_channel* width*height + c* width*height + h * width + w] !=
	//					data_dw[n * height * width * input_channel + h*width*input_channel + w*input_channel + c]) {
	//					cout << "error" << endl;
	//				}
	//			}
	//		}
	//	}
	//}

	for (int i = 0; i < size_depthW; i++)
	{
		binFile_dw.write(reinterpret_cast<const char*>(&data_dw[i]), sizeof(float));
	}

	binFile_dw.close();
	datFile_dw.close();

	delete[] data_dw;
	//delete[] dsctmp;
	cudaFree(srcData_dw);
	cudaFree(dstData_dw);
}

void binTodat_outputdata(const char* datname, const char* binname, int input_channel, int width, int height, int output_channel)
{ //string srcname_dw, string srcname_db, string srcname_dg, string srcname_dmean, string srcname_dvar,
  //string dstname_dw, string dstname_db, string dstname_dg, string dstname_dmean, string dstname_dvar
	std::string srcname_dw = (std::string(datname) + std::string(".bin"));
	std::string dstname_dw = (std::string(binname) + std::string(".dat"));


	ofstream binFile_dw(dstname_dw, ios::binary);

	ifstream datFile_dw(srcname_dw, ios::in | ios::binary);


	if (!datFile_dw)
	{
		std::cout << "error opening file" << std::endl;
	}
	int size_depthW = input_channel*width*height*output_channel;


	float* data_dw = new float[size_depthW];

	float* dstData_dw = NULL;
	float* srcData_dw = NULL;

	memset(data_dw, 0, sizeof(float)*size_depthW);

	cudaMalloc(&dstData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&srcData_dw, sizeof(float)*size_depthW);

	cudaMemset(srcData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_depthW);

	datFile_dw.read((char*)data_dw, sizeof(float) * size_depthW);
	for (int i = 0; i < size_depthW; i++)
	{
		//cout << data_dw[i] << endl;
		//datFile_dw.is_open();
		//datFile_dw >> data_dw[i];
		binFile_dw.is_open();
		binFile_dw << data_dw[i];
		binFile_dw << " ";
		//binFile_dw.write(reinterpret_cast<const char*>(&data_dw[i]), sizeof(float));
	}

	//cudaMemcpy(srcData_dw, data_dw, sizeof(float)*size_depthW, cudaMemcpyHostToDevice);
	//changeDataType(srcData_dw, dstData_dw, output_channel, input_channel, height, width);

	//cout << "====dataChange====" << endl;
	//float* dsctmp = new float[size_depthW];
	//memset(dsctmp, 0, sizeof(float) * size_depthW);
	//cudaMemcpy(dsctmp, dstData_dw, sizeof(float)* size_depthW, cudaMemcpyDeviceToHost);

	//for (int n = 0; n < output_channel; n++) {
	//	for (int h = 0; h < height; h++) {
	//		for (int w = 0; w < width; w++) {
	//			for (int c = 0; c < input_channel; c++) {
	//				//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
	//				//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
	//				if (dsctmp[n* input_channel* width*height + c* width*height + h * width + w] !=
	//					data_dw[n * height * width * input_channel + h*width*input_channel + w*input_channel + c]) {
	//					cout << "error" << endl;
	//				}
	//			}
	//		}
	//	}
	//}

	//for (int i = 0; i < size_depthW; i++)
	//{
	//	binFile_dw.write(reinterpret_cast<const char*>(&data_dw[i]), sizeof(float));
	//}

	binFile_dw.close();
	datFile_dw.close();

	delete[] data_dw;
	//delete[] dsctmp;
	cudaFree(srcData_dw);
	cudaFree(dstData_dw);
}

void datTobin_layer0(const char* datname, const char* binname, int input_channel, int output_channel, int convFiltersize)
{ //string srcname_dw, string srcname_db, string srcname_dg, string srcname_dmean, string srcname_dvar,
	//string dstname_dw, string dstname_db, string dstname_dg, string dstname_dmean, string dstname_dvar
	std::string srcname_dw = (std::string(datname) + std::string("d.dat"));
	std::string dstname_dw = (std::string(binname) + std::string("d.bin"));
	
	std::string srcname_db = (std::string(datname) + std::string("dbeta.dat"));
	std::string dstname_db = (std::string(binname) + std::string("dbeta.bin"));
	
	std::string srcname_dg = (std::string(datname) + std::string("dgamma.dat"));
	std::string dstname_dg = (std::string(binname) + std::string("dgamma.bin"));
	
	std::string srcname_dmean = (std::string(datname) + std::string("dMean.dat"));
	std::string dstname_dmean = (std::string(binname) + std::string("dMean.bin"));
	
	std::string srcname_dvar = (std::string(datname) + std::string("dVar.dat"));
	std::string dstname_dvar = (std::string(binname) + std::string("dVar.bin"));


	ofstream binFile_dw(dstname_dw, ios::binary);
	ofstream binFile_db(dstname_db, ios::binary);
	ofstream binFile_dg(dstname_dg, ios::binary);
	ofstream binFile_dmean(dstname_dmean, ios::binary);
	ofstream binFile_dvar(dstname_dvar, ios::binary);

	ifstream datFile_dw(srcname_dw, ios::in | ios::binary);
	ifstream datFile_db(srcname_db, ios::in | ios::binary);
	ifstream datFile_dg(srcname_dg, ios::in | ios::binary);
	ifstream datFile_dmean(srcname_dmean, ios::in | ios::binary);
	ifstream datFile_dvar(srcname_dvar, ios::in | ios::binary);


	if (!datFile_dw || !datFile_db || !datFile_dg || !datFile_dmean || !datFile_dvar)
	{
		std::cout << "error opening file in layer0" << std::endl;
		system("pause");
	}
	int size_depthW = input_channel*convFiltersize*convFiltersize*output_channel;
	int size_depthVar = output_channel;

	float* data_dw = new float[size_depthW];
	float* data_db = new float[size_depthVar];
	float* data_dg = new float[size_depthVar];
	float* data_dmean = new float[size_depthVar];
	float* data_dvar = new float[size_depthVar];

	float* dstData_dw = NULL;
	float* srcData_dw = NULL;

	memset(data_dw, 0, sizeof(float)*size_depthW);
	memset(data_db, 0, sizeof(float)*size_depthVar);
	memset(data_dg, 0, sizeof(float)*size_depthVar);
	memset(data_dmean, 0, sizeof(float)*size_depthVar);
	memset(data_dvar, 0, sizeof(float)*size_depthVar);

	cudaMalloc(&dstData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&srcData_dw, sizeof(float)*size_depthW);

	cudaMemset(srcData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_depthW);


	for (int i = 0; i < size_depthW; i++)
	{
		datFile_dw.is_open();
		datFile_dw >> data_dw[i];
		//binFile_dw.write(reinterpret_cast<const char*>(&data_dw[i]), sizeof(float));
	}
	cudaMemcpy(srcData_dw, data_dw, sizeof(float)*size_depthW, cudaMemcpyHostToDevice);
	changeDataType(srcData_dw, dstData_dw, output_channel, input_channel, convFiltersize, convFiltersize);
	
	cout << "====convTest====" << endl;
	float* dsctmp = new float[size_depthW];
	memset(dsctmp, 0, sizeof(float) * size_depthW);
	cudaMemcpy(dsctmp, dstData_dw, sizeof(float)* size_depthW, cudaMemcpyDeviceToHost);
	for (int h = 0; h < convFiltersize; h++) {
		for (int w = 0; w < convFiltersize; w++) {
			for (int c = 0; c < input_channel; c++) {
				for (int n = 0; n < output_channel; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] !=
						data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n]) {
						cout << "error" << endl;
					}
				}
			}
		}
	}
	for (int i = 0; i < size_depthW; i++)
	{
		binFile_dw.write(reinterpret_cast<const char*>(&dsctmp[i]), sizeof(float));
	}
		
	for (int i = 0; i < size_depthVar; i++)
	{
		datFile_db.is_open();
		datFile_db >> data_db[i];
		binFile_db.write(reinterpret_cast<const char*>(&data_db[i]), sizeof(float));
		//cout << data_db[i] << endl;

		datFile_dg.is_open();
		datFile_dg >> data_dg[i];
		binFile_dg.write(reinterpret_cast<const char*>(&data_dg[i]), sizeof(float));

		datFile_dmean.is_open();
		datFile_dmean >> data_dmean[i];
		binFile_dmean.write(reinterpret_cast<const char*>(&data_dmean[i]), sizeof(float));

		datFile_dvar.is_open();
		datFile_dvar >> data_dvar[i];
		binFile_dvar.write(reinterpret_cast<const char*>(&data_dvar[i]), sizeof(float));
	}


	binFile_dw.close();
	datFile_dw.close();
	binFile_db.close();
	datFile_db.close();
	binFile_dg.close();
	datFile_dg.close();
	binFile_dmean.close();
	datFile_dmean.close();
	binFile_dvar.close();
	datFile_dvar.close();

	delete[] data_dw;
	delete[] data_db;
	delete[] data_dg;
	delete[] data_dmean;
	delete[] data_dvar;
	delete[] dsctmp;
	cudaFree(srcData_dw);
	cudaFree(dstData_dw);
}

void datTobin_layer1to13(const char* datname, const char* binname,
	int input_channel, int output_channel, int convFiltersize)
{
	std::string srcname_dw = (std::string(datname) + std::string("d.dat"));
	std::string dstname_dw = (std::string(binname) + std::string("d.bin"));
	std::string srcname_pw = (std::string(datname) + std::string("p.dat"));
	std::string dstname_pw = (std::string(binname) + std::string("p.bin"));
	std::string srcname_db = (std::string(datname) + std::string("dbeta.dat"));
	std::string dstname_db = (std::string(binname) + std::string("dbeta.bin"));
	std::string srcname_pb = (std::string(datname) + std::string("pbeta.dat"));
	std::string dstname_pb = (std::string(binname) + std::string("pbeta.bin"));
	std::string srcname_dg = (std::string(datname) + std::string("dgamma.dat"));
	std::string dstname_dg = (std::string(binname) + std::string("dgamma.bin"));
	std::string srcname_pg = (std::string(datname) + std::string("pgamma.dat"));
	std::string dstname_pg = (std::string(binname) + std::string("pgamma.bin"));
	std::string srcname_dmean = (std::string(datname) + std::string("dMean.dat"));
	std::string dstname_dmean = (std::string(binname) + std::string("dMean.bin"));
	std::string srcname_pmean = (std::string(datname) + std::string("pMean.dat"));
	std::string dstname_pmean = (std::string(binname) + std::string("pMean.bin"));
	std::string srcname_dvar = (std::string(datname) + std::string("dVar.dat"));
	std::string dstname_dvar = (std::string(binname) + std::string("dVar.bin"));
	std::string srcname_pvar = (std::string(datname) + std::string("pVar.dat"));
	std::string dstname_pvar = (std::string(binname) + std::string("pVar.bin"));

	ofstream binFile_dw(dstname_dw, ios::binary);
	ofstream binFile_db(dstname_db, ios::binary);
	ofstream binFile_dg(dstname_dg, ios::binary);
	ofstream binFile_dmean(dstname_dmean, ios::binary);
	ofstream binFile_dvar(dstname_dvar, ios::binary);
	ofstream binFile_pw(dstname_pw, ios::binary);
	ofstream binFile_pb(dstname_pb, ios::binary);
	ofstream binFile_pg(dstname_pg, ios::binary);
	ofstream binFile_pmean(dstname_pmean, ios::binary);
	ofstream binFile_pvar(dstname_pvar, ios::binary);

	ifstream datFile_dw(srcname_dw, ios::in | ios::binary);
	ifstream datFile_db(srcname_db, ios::in | ios::binary);
	ifstream datFile_dg(srcname_dg, ios::in | ios::binary);
	ifstream datFile_dmean(srcname_dmean, ios::in | ios::binary);
	ifstream datFile_dvar(srcname_dvar, ios::in | ios::binary);
	ifstream datFile_pw(srcname_pw, ios::in | ios::binary);
	ifstream datFile_pb(srcname_pb, ios::in | ios::binary);
	ifstream datFile_pg(srcname_pg, ios::in | ios::binary);
	ifstream datFile_pmean(srcname_pmean, ios::in | ios::binary);
	ifstream datFile_pvar(srcname_pvar, ios::in | ios::binary);

	if (!datFile_dw || !datFile_db || !datFile_dg || !datFile_dmean || !datFile_dvar || !datFile_pw ||
		!datFile_pb || !datFile_pg || !datFile_pmean || !datFile_pvar)
	{
		std::cout << "error opening file in layer1to13" << std::endl;
		system("pause");
	}
	int size_depthW = input_channel*convFiltersize*convFiltersize;
	int size_depthVar = input_channel;
	int size_pointW = input_channel * 1 * 1 * output_channel;
	int size_pointVar = output_channel;

	float* data_dw = new float[size_depthW];
	float* data_db = new float[size_depthVar];
	float* data_dg = new float[size_depthVar];
	float* data_dmean = new float[size_depthVar];
	float* data_dvar = new float[size_depthVar];

	float* data_pw = new float[size_pointW];
	float* data_pb = new float[size_pointVar];
	float* data_pg = new float[size_pointVar];
	float* data_pmean = new float[size_pointVar];
	float* data_pvar = new float[size_pointVar];

	memset(data_dw, 0, sizeof(float)*size_depthW);
	memset(data_db, 0, sizeof(float)*size_depthVar);
	memset(data_dg, 0, sizeof(float)*size_depthVar);
	memset(data_dmean, 0, sizeof(float)*size_depthVar);
	memset(data_dvar, 0, sizeof(float)*size_depthVar);
	memset(data_pw, 0, sizeof(float)*size_pointW);
	memset(data_pb, 0, sizeof(float)*size_pointVar);
	memset(data_pg, 0, sizeof(float)*size_pointVar);
	memset(data_pmean, 0, sizeof(float)*size_pointVar);
	memset(data_pvar, 0, sizeof(float)*size_pointVar);

	//dataType change
	float* dstData_dw = NULL;
	float* srcData_dw = NULL;
	float* srcData_pw = NULL;
	float* dstData_pw = NULL;

	cudaMalloc(&dstData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&srcData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&dstData_pw, sizeof(float)*size_pointW);
	cudaMalloc(&srcData_pw, sizeof(float)*size_pointW);

	cudaMemset(srcData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(srcData_pw, 0, sizeof(float)*size_pointW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_pointW);

	for (int i = 0; i < size_depthW; i++)
	{
		datFile_dw.is_open();
		datFile_dw >> data_dw[i];
	}

	cudaMemcpy(srcData_dw, data_dw, sizeof(float)*size_depthW, cudaMemcpyHostToDevice);
	changeDataType(srcData_dw, dstData_dw, 1, input_channel, convFiltersize, convFiltersize);

	cout << "====DEPTHTest====" << endl;
	float* dsctmp = new float[size_depthW];
	memset(dsctmp, 0, sizeof(float) * size_depthW);
	cudaMemcpy(dsctmp, dstData_dw, sizeof(float)* size_depthW, cudaMemcpyDeviceToHost);
	for (int h = 0; h < convFiltersize; h++) {
		for (int w = 0; w < convFiltersize; w++) {
			for (int c = 0; c < input_channel; c++) {
				for (int n = 0; n < 1; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] !=
						data_dw[h*convFiltersize*input_channel*1 + w*input_channel*1 + c * 1 + n]) {
						cout << "error" << endl;
						system("pause");
					}
				}
			}
		}
	}
	//write [NCHW] binary data.
	for (int i = 0; i < size_depthW; i++)
	{
		binFile_dw.write(reinterpret_cast<const char*>(&dsctmp[i]), sizeof(float));
	}
	for (int i = 0; i < size_depthVar; i++)
	{
		datFile_db.is_open();
		datFile_db >> data_db[i];
		binFile_db.write(reinterpret_cast<const char*>(&data_db[i]), sizeof(float));
		

		datFile_dg.is_open();
		datFile_dg >> data_dg[i];
		binFile_dg.write(reinterpret_cast<const char*>(&data_dg[i]), sizeof(float));

		datFile_dmean.is_open();
		datFile_dmean >> data_dmean[i];
		binFile_dmean.write(reinterpret_cast<const char*>(&data_dmean[i]), sizeof(float));

		datFile_dvar.is_open();
		datFile_dvar >> data_dvar[i];
		binFile_dvar.write(reinterpret_cast<const char*>(&data_dvar[i]), sizeof(float));
	}
	
	for (int i = 0; i < size_pointW; i++)
	{
		datFile_pw.is_open();
		datFile_pw >> data_pw[i];
	}

	cudaMemcpy(srcData_pw, data_pw, sizeof(float)*size_pointW, cudaMemcpyHostToDevice);
	changeDataType(srcData_pw, dstData_pw, output_channel, input_channel, 1, 1);

	cout << "====POINTTest====" << endl;
	float* dsctmp_p = new float[size_pointW];
	memset(dsctmp_p, 0, sizeof(float) * size_pointW);
	cudaMemcpy(dsctmp_p, dstData_pw, sizeof(float)* size_pointW, cudaMemcpyDeviceToHost);
	for (int h = 0; h < 1; h++) {
		for (int w = 0; w < 1; w++) {
			for (int c = 0; c < input_channel; c++) {
				for (int n = 0; n < output_channel; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp_p[n* input_channel* 1*1 + c*1*1 + h * 1 + w] !=
						data_pw[h*1*input_channel * output_channel + w*input_channel * output_channel + c * output_channel + n]) {
						cout << "error" << endl;
						system("pause");
					}
				}
			}
		}
	}
	//write [NCHW] binary data.
	for (int i = 0; i < size_pointW; i++)
	{
		binFile_pw.write(reinterpret_cast<const char*>(&dsctmp_p[i]), sizeof(float));
	}

	for (int i = 0; i < size_pointVar; i++)
	{
		datFile_pb.is_open();
		datFile_pb >> data_pb[i];
		binFile_pb.write(reinterpret_cast<const char*>(&data_pb[i]), sizeof(float));
		

		datFile_pg.is_open();
		datFile_pg >> data_pg[i];
		binFile_pg.write(reinterpret_cast<const char*>(&data_pg[i]), sizeof(float));

		datFile_pmean.is_open();
		datFile_pmean >> data_pmean[i];
		binFile_pmean.write(reinterpret_cast<const char*>(&data_pmean[i]), sizeof(float));

		datFile_pvar.is_open();
		datFile_pvar >> data_pvar[i];
		binFile_pvar.write(reinterpret_cast<const char*>(&data_pvar[i]), sizeof(float));
	}
	//system("pause");
	binFile_dw.close();
	datFile_dw.close();
	binFile_db.close();
	datFile_db.close();
	binFile_dg.close();
	datFile_dg.close();
	binFile_dmean.close();
	datFile_dmean.close();
	binFile_dvar.close();
	datFile_dvar.close();
	binFile_pw.close();
	datFile_pw.close();
	binFile_pb.close();
	datFile_pb.close();
	binFile_pg.close();
	datFile_pg.close();
	binFile_pmean.close();
	datFile_pmean.close();
	binFile_pvar.close();
	datFile_pvar.close();

	delete[] data_dw;
	delete[] data_db;
	delete[] data_dg;
	delete[] data_dmean;
	delete[] data_dvar;
	delete[] data_pw;
	delete[] data_pb;
	delete[] data_pg;
	delete[] data_pmean;
	delete[] data_pvar;

	cudaFree(srcData_dw);
	cudaFree(dstData_dw);
	cudaFree(srcData_pw);
	cudaFree(dstData_pw);
	//delete[] dstData_dw;
	//delete[] srcData_dw;
	//delete[] srcData_pw;
	//delete[] dstData_pw;
}

void datTobin_layer14to17(const char* datname, const char* binname,
	int input_channel, int mid_channel, int output_channel, int convFiltersize)
{

	std::string srcname_dw = (std::string(datname) + std::string("2_d.dat"));
	std::string dstname_dw = (std::string(binname) + std::string("2_d.bin"));
	std::string srcname_pw = (std::string(datname) + std::string("p.dat"));
	std::string dstname_pw = (std::string(binname) + std::string("p.bin"));
	std::string srcname_db = (std::string(datname) + std::string("2_dbeta.dat"));
	std::string dstname_db = (std::string(binname) + std::string("2_dbeta.bin"));
	std::string srcname_pb = (std::string(datname) + std::string("pbeta.dat"));
	std::string dstname_pb = (std::string(binname) + std::string("pbeta.bin"));
	std::string srcname_dg = (std::string(datname) + std::string("2_dgamma.dat"));
	std::string dstname_dg = (std::string(binname) + std::string("2_dgamma.bin"));
	std::string srcname_pg = (std::string(datname) + std::string("pgamma.dat"));
	std::string dstname_pg = (std::string(binname) + std::string("pgamma.bin"));
	std::string srcname_dmean = (std::string(datname) + std::string("2_dMean.dat"));
	std::string dstname_dmean = (std::string(binname) + std::string("2_dMean.bin"));
	std::string srcname_pmean = (std::string(datname) + std::string("pMean.dat"));
	std::string dstname_pmean = (std::string(binname) + std::string("pMean.bin"));
	std::string srcname_dvar = (std::string(datname) + std::string("2_dVar.dat"));
	std::string dstname_dvar = (std::string(binname) + std::string("2_dVar.bin"));
	std::string srcname_pvar = (std::string(datname) + std::string("pVar.dat"));
	std::string dstname_pvar = (std::string(binname) + std::string("pVar.bin"));

	ofstream binFile_dw(dstname_dw, ios::binary);
	ofstream binFile_db(dstname_db, ios::binary);
	ofstream binFile_dg(dstname_dg, ios::binary);
	ofstream binFile_dmean(dstname_dmean, ios::binary);
	ofstream binFile_dvar(dstname_dvar, ios::binary);
	ofstream binFile_pw(dstname_pw, ios::binary);
	ofstream binFile_pb(dstname_pb, ios::binary);
	ofstream binFile_pg(dstname_pg, ios::binary);
	ofstream binFile_pmean(dstname_pmean, ios::binary);
	ofstream binFile_pvar(dstname_pvar, ios::binary);

	ifstream datFile_dw(srcname_dw, ios::in | ios::binary);
	ifstream datFile_db(srcname_db, ios::in | ios::binary);
	ifstream datFile_dg(srcname_dg, ios::in | ios::binary);
	ifstream datFile_dmean(srcname_dmean, ios::in | ios::binary);
	ifstream datFile_dvar(srcname_dvar, ios::in | ios::binary);
	ifstream datFile_pw(srcname_pw, ios::in | ios::binary);
	ifstream datFile_pb(srcname_pb, ios::in | ios::binary);
	ifstream datFile_pg(srcname_pg, ios::in | ios::binary);
	ifstream datFile_pmean(srcname_pmean, ios::in | ios::binary);
	ifstream datFile_pvar(srcname_pvar, ios::in | ios::binary);

	if (!datFile_dw || !datFile_db || !datFile_dg || !datFile_dmean || !datFile_dvar || !datFile_pw ||
		!datFile_pb || !datFile_pg || !datFile_pmean || !datFile_pvar)
	{
		std::cout << "error opening file in layer14to17" << std::endl;
		system("pause");
	}
	int size_depthW = output_channel*convFiltersize*convFiltersize* mid_channel;
	int size_depthVar = output_channel;
	int size_pointW = input_channel * 1 * 1 * mid_channel;
	int size_pointVar = mid_channel;

	float* data_dw = new float[size_depthW];
	float* data_db = new float[size_depthVar];
	float* data_dg = new float[size_depthVar];
	float* data_dmean = new float[size_depthVar];
	float* data_dvar = new float[size_depthVar];

	float* data_pw = new float[size_pointW];
	float* data_pb = new float[size_pointVar];
	float* data_pg = new float[size_pointVar];
	float* data_pmean = new float[size_pointVar];
	float* data_pvar = new float[size_pointVar];

	memset(data_dw, 0, sizeof(float)*size_depthW);
	memset(data_db, 0, sizeof(float)*size_depthVar);
	memset(data_dg, 0, sizeof(float)*size_depthVar);
	memset(data_dmean, 0, sizeof(float)*size_depthVar);
	memset(data_dvar, 0, sizeof(float)*size_depthVar);
	memset(data_pw, 0, sizeof(float)*size_pointW);
	memset(data_pb, 0, sizeof(float)*size_pointVar);
	memset(data_pg, 0, sizeof(float)*size_pointVar);
	memset(data_pmean, 0, sizeof(float)*size_pointVar);
	memset(data_pvar, 0, sizeof(float)*size_pointVar);

	//dataType change
	float* dstData_dw = NULL;
	float* srcData_dw = NULL;
	float* srcData_pw = NULL;
	float* dstData_pw = NULL;

	cudaMalloc(&dstData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&srcData_dw, sizeof(float)*size_depthW);
	cudaMalloc(&dstData_pw, sizeof(float)*size_pointW);
	cudaMalloc(&srcData_pw, sizeof(float)*size_pointW);

	cudaMemset(srcData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_depthW);
	cudaMemset(srcData_pw, 0, sizeof(float)*size_pointW);
	cudaMemset(dstData_dw, 0, sizeof(float)*size_pointW);

	for (int i = 0; i < size_depthW; i++)
	{
		datFile_dw.is_open();
		datFile_dw >> data_dw[i];
		//binFile_dw.write(reinterpret_cast<const char*>(&data_dw[i]), sizeof(float));
	}
	cudaMemcpy(srcData_dw, data_dw, sizeof(float)*size_depthW, cudaMemcpyHostToDevice);
	changeDataType(srcData_dw, dstData_dw, output_channel, mid_channel, convFiltersize, convFiltersize);

	cout << "====DEPTHTest====" << endl;
	float* dsctmp = new float[size_depthW];
	memset(dsctmp, 0, sizeof(float) * size_depthW);
	cudaMemcpy(dsctmp, dstData_dw, sizeof(float)* size_depthW, cudaMemcpyDeviceToHost);
	for (int h = 0; h < convFiltersize; h++) {
		for (int w = 0; w < convFiltersize; w++) {
			for (int c = 0; c < mid_channel; c++) {
				for (int n = 0; n < output_channel; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp[n* mid_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] !=
						data_dw[h*convFiltersize*mid_channel * output_channel + w*mid_channel * output_channel + c * output_channel + n]) {
						cout << "error" << endl;
						system("pause");
					}
				}
			}
		}
	}

	//write [NCHW] binary data.
	for (int i = 0; i < size_depthW; i++)
	{
		binFile_dw.write(reinterpret_cast<const char*>(&dsctmp[i]), sizeof(float));
	}

	for (int i = 0; i < size_depthVar; i++)
	{
		datFile_db.is_open();
		datFile_db >> data_db[i];
		binFile_db.write(reinterpret_cast<const char*>(&data_db[i]), sizeof(float));
		//cout << data_db[i] << endl;

		datFile_dg.is_open();
		datFile_dg >> data_dg[i];
		binFile_dg.write(reinterpret_cast<const char*>(&data_dg[i]), sizeof(float));

		datFile_dmean.is_open();
		datFile_dmean >> data_dmean[i];
		binFile_dmean.write(reinterpret_cast<const char*>(&data_dmean[i]), sizeof(float));

		datFile_dvar.is_open();
		datFile_dvar >> data_dvar[i];
		binFile_dvar.write(reinterpret_cast<const char*>(&data_dvar[i]), sizeof(float));
	}

	for (int i = 0; i < size_pointW; i++)
	{
		datFile_pw.is_open();
		datFile_pw >> data_pw[i];
		//binFile_pw.write(reinterpret_cast<const char*>(&data_pw[i]), sizeof(float));
	}
	cudaMemcpy(srcData_pw, data_pw, sizeof(float)*size_pointW, cudaMemcpyHostToDevice);
	changeDataType(srcData_pw, dstData_pw, mid_channel, input_channel, 1, 1);

	cout << "====POINTTest====" << endl;
	float* dsctmp_p = new float[size_pointW];
	memset(dsctmp_p, 0, sizeof(float) * size_pointW);
	cudaMemcpy(dsctmp_p, dstData_pw, sizeof(float)* size_pointW, cudaMemcpyDeviceToHost);
	for (int h = 0; h < 1; h++) {
		for (int w = 0; w < 1; w++) {
			for (int c = 0; c < input_channel; c++) {
				for (int n = 0; n < mid_channel; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp_p[n* input_channel * 1 * 1 + c * 1 * 1 + h * 1 + w] !=
						data_pw[h * 1 * input_channel * mid_channel + w*input_channel * mid_channel + c * mid_channel + n]) {
						cout << "error" << endl;
						system("pause");
					}
				}
			}
		}
	}

	//write [NCHW] binary data.
	for (int i = 0; i < size_pointW; i++)
	{
		binFile_pw.write(reinterpret_cast<const char*>(&dsctmp_p[i]), sizeof(float));
	}

	for (int i = 0; i < size_pointVar; i++)
	{
		datFile_pb.is_open();
		datFile_pb >> data_pb[i];
		binFile_pb.write(reinterpret_cast<const char*>(&data_pb[i]), sizeof(float));
		
		datFile_pg.is_open();
		datFile_pg >> data_pg[i];
		binFile_pg.write(reinterpret_cast<const char*>(&data_pg[i]), sizeof(float));

		datFile_pmean.is_open();
		datFile_pmean >> data_pmean[i];
		binFile_pmean.write(reinterpret_cast<const char*>(&data_pmean[i]), sizeof(float));

		datFile_pvar.is_open();
		datFile_pvar >> data_pvar[i];
		binFile_pvar.write(reinterpret_cast<const char*>(&data_pvar[i]), sizeof(float));
	}

	binFile_dw.close();
	datFile_dw.close();
	binFile_db.close();
	datFile_db.close();
	binFile_dg.close();
	datFile_dg.close();
	binFile_dmean.close();
	datFile_dmean.close();
	binFile_dvar.close();
	datFile_dvar.close();
	binFile_pw.close();
	datFile_pw.close();
	binFile_pb.close();
	datFile_pb.close();
	binFile_pg.close();
	datFile_pg.close();
	binFile_pmean.close();
	datFile_pmean.close();
	binFile_pvar.close();
	datFile_pvar.close();

	delete[] data_dw;
	delete[] data_db;
	delete[] data_dg;
	delete[] data_dmean;
	delete[] data_dvar;
	delete[] data_pw;
	delete[] data_pb;
	delete[] data_pg;
	delete[] data_pmean;
	delete[] data_pvar;

	cudaFree(srcData_dw);
	cudaFree(dstData_dw);
	cudaFree(srcData_pw);
	cudaFree(dstData_pw);
	delete[] dsctmp;
	delete[] dsctmp_p;
}

void datTobin_layerbox(const char* datname_box, const char* binname_box,
	int input_channel, int class_num, int box_num, int box_code)
{
	std::string srcname_dw = (std::string(datname_box) + std::string("boxp_w.dat"));
	std::string dstname_dw = (std::string(binname_box) + std::string("boxp_w.bin"));
	std::string srcname_pw = (std::string(datname_box) + std::string("classp_w.dat"));
	std::string dstname_pw = (std::string(binname_box) + std::string("classp_w.bin"));
	std::string srcname_db = (std::string(datname_box) + std::string("boxp_b.dat"));
	std::string dstname_db = (std::string(binname_box) + std::string("boxp_b.bin"));
	std::string srcname_pb = (std::string(datname_box) + std::string("classp_b.dat"));
	std::string dstname_pb = (std::string(binname_box) + std::string("classp_b.bin"));


	ofstream binFile_locw(dstname_dw, ios::binary);
	ofstream binFile_locb(dstname_db, ios::binary);
	ofstream binFile_confw(dstname_pw, ios::binary);
	ofstream binFile_confb(dstname_pb, ios::binary);

	ifstream datFile_locw(srcname_dw, ios::in | ios::binary);
	ifstream datFile_locb(srcname_db, ios::in | ios::binary);
	ifstream datFile_confw(srcname_pw, ios::in | ios::binary);
	ifstream datFile_confb(srcname_pb, ios::in | ios::binary);

	if (!datFile_locw || !datFile_locb || !datFile_confw || !datFile_confb)
	{
		std::cout << "error opening file in boxlayer" << std::endl;
		system("pause");
	}
	
	int size_locw = input_channel * 1 * 1 * box_num * box_code;
	int size_locb = box_num * box_code;
	int size_confw = input_channel * 1 * 1 * (class_num + 1) * box_num;
	int size_confb = (class_num + 1) * box_num;

	float* data_locw = new float[size_locw];
	float* data_locb = new float[size_locb];
	float* data_confw = new float[size_confw];
	float* data_confb = new float[size_confb];

	memset(data_locw, 0, sizeof(float)*size_locw);
	memset(data_locb, 0, sizeof(float)*size_locb);
	memset(data_confw, 0, sizeof(float)*size_confw);
	memset(data_confb, 0, sizeof(float)*size_confb);

	//changed dataType variables 
	float* dstData_locw = NULL;
	float* srcData_locw = NULL;
	float* srcData_confw = NULL;
	float* dstData_confw = NULL;

	cudaMalloc(&dstData_locw, sizeof(float)*size_locw);
	cudaMalloc(&srcData_locw, sizeof(float)*size_locw);
	cudaMalloc(&dstData_confw, sizeof(float)*size_confw);
	cudaMalloc(&srcData_confw, sizeof(float)*size_confw);

	cudaMemset(srcData_locw, 0, sizeof(float)*size_locw);
	cudaMemset(dstData_locw, 0, sizeof(float)*size_locw);
	cudaMemset(srcData_confw, 0, sizeof(float)*size_confw);
	cudaMemset(dstData_confw, 0, sizeof(float)*size_confw);

	//Open & read the dat file
	for (int i = 0; i < size_locw; i++)
	{
		datFile_locw.is_open();
		datFile_locw >> data_locw[i];
		//binFile_locw.write(reinterpret_cast<const char*>(&data_locw[i]), sizeof(float));
	}
	cudaMemcpy(srcData_locw, data_locw, sizeof(float)*size_locw, cudaMemcpyHostToDevice);
	changeDataType(srcData_locw, dstData_locw, box_num * box_code, input_channel, 1, 1);

	cout << "====locTest====" << endl;
	float* dsctmp = new float[size_locw];
	memset(dsctmp, 0, sizeof(float) * size_locw);
	cudaMemcpy(dsctmp, dstData_locw, sizeof(float)* size_locw, cudaMemcpyDeviceToHost);
	for (int h = 0; h < 1; h++) {
		for (int w = 0; w < 1; w++) {
			for (int c = 0; c < input_channel; c++) {
				for (int n = 0; n < box_num * box_code; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp[n* input_channel* 1 * 1 + c*1*1 + h * 1 + w] !=
						data_locw[h*1*input_channel * box_num * box_code + w*input_channel * box_num * box_code + c * box_num * box_code + n]) {
						cout << "error" << endl;
						system("pause");
					}
				}
			}
		}
	}

	//write [NCHW] binary data.
	for (int i = 0; i < size_locw; i++)
	{
		binFile_locw.write(reinterpret_cast<const char*>(&dsctmp[i]), sizeof(float));
	}

	for (int i = 0; i < size_locb; i++)
	{
		datFile_locb.is_open();
		datFile_locb >> data_locb[i];
		binFile_locb.write(reinterpret_cast<const char*>(&data_locb[i]), sizeof(float));
		//cout << data_db[i] << endl;
	}

	//Open & read the dat file
	for (int i = 0; i < size_confw; i++)
	{
		datFile_confw.is_open();
		datFile_confw >> data_confw[i];
		//binFile_confw.write(reinterpret_cast<const char*>(&data_confw[i]), sizeof(float));
	}
	cudaMemcpy(srcData_confw, data_confw, sizeof(float)*size_confw, cudaMemcpyHostToDevice);
	changeDataType(srcData_confw, dstData_confw, (class_num +1)*box_num, input_channel, 1, 1);

	cout << "====confTest====" << endl;
	float* dsctmp_p = new float[size_confw];
	memset(dsctmp_p, 0, sizeof(float) * size_confw);
	cudaMemcpy(dsctmp_p, dstData_confw, sizeof(float)* size_confw, cudaMemcpyDeviceToHost);
	for (int h = 0; h < 1; h++) {
		for (int w = 0; w < 1; w++) {
			for (int c = 0; c < input_channel; c++) {
				for (int n = 0; n < (class_num + 1)*box_num; n++) {
					//cout << dsctmp[n* input_channel* convFiltersize*convFiltersize + c*convFiltersize*convFiltersize + h * convFiltersize + w] << "\t"
					//	<< data_dw[h*convFiltersize*input_channel*output_channel + w*input_channel*output_channel + c * output_channel + n] << endl;
					if (dsctmp_p[n* input_channel * 1 * 1 + c * 1 * 1 + h * 1 + w] !=
						data_confw[h * 1 * input_channel * (class_num + 1)*box_num + w*input_channel * (class_num + 1)*box_num + c * (class_num + 1)*box_num + n]) {
						cout << "error" << endl;
						system("pause");
					}
				}
			}
		}
	}

	//write [NCHW] binary data.
	for (int i = 0; i < size_confw; i++)
	{
		binFile_confw.write(reinterpret_cast<const char*>(&dsctmp_p[i]), sizeof(float));
	}


	for (int i = 0; i < size_confb; i++)
	{
		datFile_confb.is_open();
		datFile_confb >> data_confb[i];
		binFile_confb.write(reinterpret_cast<const char*>(&data_confb[i]), sizeof(float));
	}

	binFile_locw.close();
	datFile_locw.close();
	binFile_locb.close();
	datFile_locb.close();
	binFile_confw.close();
	datFile_confw.close();
	binFile_confb.close();
	datFile_confb.close();
	delete[] data_locw;
	delete[] data_locb;
	delete[] data_confw;
	delete[] data_confb;


	cudaFree(dstData_locw);
	cudaFree(srcData_locw);
	cudaFree(srcData_confw);
	cudaFree(dstData_confw);
	delete[] dsctmp;
	delete[] dsctmp_p;
}

int main()
{
	
	////////layer0 - 13
	//std::string srcname_dw = (std::string(datname) + std::string("d.dat"));
	//std::string dstname_dw = (std::string(binname) + std::string("d.bin"));
	//std::string srcname_pw = (std::string(datname) + std::string("p.dat"));
	//std::string dstname_pw = (std::string(binname) + std::string("p.bin"));
	//std::string srcname_db = (std::string(datname) + std::string("dbeta.dat"));
	//std::string dstname_db = (std::string(binname) + std::string("dbeta.bin"));
	//std::string srcname_pb = (std::string(datname) + std::string("pbeta.dat"));
	//std::string dstname_pb = (std::string(binname) + std::string("pbeta.bin"));
	//std::string srcname_dg = (std::string(datname) + std::string("dgamma.dat"));
	//std::string dstname_dg = (std::string(binname) + std::string("dgamma.bin"));
	//std::string srcname_pg = (std::string(datname) + std::string("pgamma.dat"));
	//std::string dstname_pg = (std::string(binname) + std::string("pgamma.bin"));
	//std::string srcname_dmean = (std::string(datname) + std::string("dMean.dat"));
	//std::string dstname_dmean = (std::string(binname) + std::string("dMean.bin"));
	//std::string srcname_pmean = (std::string(datname) + std::string("pMean.dat"));
	//std::string dstname_pmean = (std::string(binname) + std::string("pMean.bin"));
	//std::string srcname_dvar = (std::string(datname) + std::string("dVar.dat"));
	//std::string dstname_dvar = (std::string(binname) + std::string("dVar.bin"));
	//std::string srcname_pvar = (std::string(datname) + std::string("pVar.dat"));
	//std::string dstname_pvar = (std::string(binname) + std::string("pVar.bin"));

	/////////layer14-17
	//std::string srcname_dw = (std::string(datname) + std::string("2_d.dat"));
	//std::string dstname_dw = (std::string(binname) + std::string("2_d.bin"));
	//std::string srcname_pw = (std::string(datname) + std::string("p.dat"));
	//std::string dstname_pw = (std::string(binname) + std::string("p.bin"));
	//std::string srcname_db = (std::string(datname) + std::string("2_dbeta.dat"));
	//std::string dstname_db = (std::string(binname) + std::string("2_dbeta.bin"));
	//std::string srcname_pb = (std::string(datname) + std::string("pbeta.dat"));
	//std::string dstname_pb = (std::string(binname) + std::string("pbeta.bin"));
	//std::string srcname_dg = (std::string(datname) + std::string("2_dgamma.dat"));
	//std::string dstname_dg = (std::string(binname) + std::string("2_dgamma.bin"));
	//std::string srcname_pg = (std::string(datname) + std::string("pgamma.dat"));
	//std::string dstname_pg = (std::string(binname) + std::string("pgamma.bin"));
	//std::string srcname_dmean = (std::string(datname) + std::string("2_dMean.dat"));
	//std::string dstname_dmean = (std::string(binname) + std::string("2_dMean.bin"));
	//std::string srcname_pmean = (std::string(datname) + std::string("pMean.dat"));
	//std::string dstname_pmean = (std::string(binname) + std::string("pMean.bin"));
	//std::string srcname_dvar = (std::string(datname) + std::string("2_dVar.dat"));
	//std::string dstname_dvar = (std::string(binname) + std::string("2_dVar.bin"));
	//std::string srcname_pvar = (std::string(datname) + std::string("pVar.dat"));
	//std::string dstname_pvar = (std::string(binname) + std::string("pVar.bin"));

	///////box layer

	//std::string srcname_dw = (std::string(datname_box) + std::string("boxp_w.dat"));
	//std::string dstname_dw = (std::string(binname_box) + std::string("boxp_w.bin"));
	//std::string srcname_pw = (std::string(datname_box) + std::string("classp_w.dat"));
	//std::string dstname_pw = (std::string(binname_box) + std::string("classp_w.bin"));
	//std::string srcname_db = (std::string(datname_box) + std::string("boxp_b.dat"));
	//std::string dstname_db = (std::string(binname_box) + std::string("boxp_b.bin"));
	//std::string srcname_pb = (std::string(datname_box) + std::string("classp_b.dat"));
	//std::string dstname_pb = (std::string(binname_box) + std::string("classp_b.bin"));

	int conv_filtersize = 3;

	const char* datname0 = "datData/conv0_";
	const char* binname0 = "binData/conv0_";
	const char* datname1 = "datData/conv1_";
	const char* binname1 = "binData/conv1_";
	const char* datname2 = "datData/conv2_";
	const char* binname2 = "binData/conv2_";
	const char* datname3 = "datData/conv3_";
	const char* binname3 = "binData/conv3_";
	const char* datname4 = "datData/conv4_";
	const char* binname4 = "binData/conv4_";
	const char* datname5 = "datData/conv5_";
	const char* binname5 = "binData/conv5_";
	const char* datname6 = "datData/conv6_";
	const char* binname6 = "binData/conv6_";
	const char* datname7 = "datData/conv7_";
	const char* binname7 = "binData/conv7_";
	const char* datname8 = "datData/conv8_";
	const char* binname8 = "binData/conv8_";
	const char* datname9 = "datData/conv9_";
	const char* binname9 = "binData/conv9_";
	const char* datname10 = "datData/conv10_";
	const char* binname10 = "binData/conv10_";
	const char* datname11 = "datData/conv11_";
	const char* binname11 = "binData/conv11_";
	const char* datname12 = "datData/conv12_";
	const char* binname12 = "binData/conv12_";
	const char* datname13 = "datData/conv13_";
	const char* binname13 = "binData/conv13_";

	const char* datname14 = "datData/conv14_";
	const char* binname14 = "binData/conv14_";
	const char* datname15 = "datData/conv15_";
	const char* binname15 = "binData/conv15_";
	const char* datname16 = "datData/conv16_";
	const char* binname16 = "binData/conv16_";
	const char* datname17 = "datData/conv17_";
	const char* binname17 = "binData/conv17_";

	const char* datname_box0 = "datData/box0_";
	const char* binname_box0 = "binData/box0_";
	const char* datname_box1 = "datData/box1_";
	const char* binname_box1 = "binData/box1_";
	const char* datname_box2 = "datData/box2_";
	const char* binname_box2 = "binData/box2_";
	const char* datname_box3 = "datData/box3_";
	const char* binname_box3 = "binData/box3_";
	const char* datname_box4 = "datData/box4_";
	const char* binname_box4 = "binData/box4_";
	const char* datname_box5 = "datData/box5_";
	const char* binname_box5 = "binData/box5_";

	//const char* datname_test = "preprocessed_input";
	//const char* binname_test = "preprocessed_input";

	//datTobin_outputdata(datname_test, binname_test, 3, 300, 300, 1);
	
	datTobin_layer0(datname0, binname0, 3, 32, conv_filtersize);
	
	datTobin_layer1to13(datname1, binname1, 32, 64, 3);
	datTobin_layer1to13(datname2, binname2, 64, 128, 3);
	datTobin_layer1to13(datname3, binname3, 128, 128, 3);
	datTobin_layer1to13(datname4, binname4, 128, 256, 3);
	datTobin_layer1to13(datname5, binname5, 256, 256, 3);
	datTobin_layer1to13(datname6, binname6, 256, 512, 3);
	datTobin_layer1to13(datname7, binname7, 512, 512, 3);
	datTobin_layer1to13(datname8, binname8, 512, 512, 3);
	datTobin_layer1to13(datname9, binname9, 512, 512, 3);
	datTobin_layer1to13(datname10, binname10, 512, 512, 3);
	datTobin_layer1to13(datname11, binname11, 512, 512, 3);
	datTobin_layer1to13(datname12, binname12, 512, 1024, 3);
	datTobin_layer1to13(datname13, binname13, 1024, 1024, 3);
	cout << "layer1to13 done" << endl;
	datTobin_layer14to17(datname14, binname14, 1024, 256, 512, conv_filtersize);
	datTobin_layer14to17(datname15, binname15, 512, 128, 256, conv_filtersize);
	datTobin_layer14to17(datname16, binname16, 256, 128, 256, conv_filtersize);
	datTobin_layer14to17(datname17, binname17, 256, 64, 128, conv_filtersize);
	cout << "layer 14 to 17 done " << endl;
	int class_num = 35; //classes without backgroundclass
	int box_num = 3; //boxlayer0 = 3, the other layer = 6
	int box_code = 4; // [x_c, y_c, w, h]
	datTobin_layerbox(datname_box0, binname_box0, 512, class_num, box_num, box_code);
	datTobin_layerbox(datname_box1, binname_box1, 1024, class_num, 2 * box_num, box_code);
	datTobin_layerbox(datname_box2, binname_box2, 512, class_num, 2 * box_num, box_code);
	datTobin_layerbox(datname_box3, binname_box3, 256, class_num, 2 * box_num, box_code);
	datTobin_layerbox(datname_box4, binname_box4, 256, class_num, 2 * box_num, box_code);
	datTobin_layerbox(datname_box5, binname_box5, 128, class_num, 2 * box_num, box_code);
	cout << "boxLayer Done" << endl;
	cout << "work done" << endl;
	system("pause");
	return 0;
}