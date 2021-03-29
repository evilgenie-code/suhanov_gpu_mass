#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h> 

#include <time.h>

using namespace std;

__device__ double factorial(int n)
{
	int x = 1;
	for (int i = 1; i <= n; i++)
		x = x * i;
	return x;
}

__device__ double Cn2(double x, int n, double cn2) {

	double cn;

	cn = 1. / factorial(n) - x * cn2;

	return cn;
}

__device__ double Cn(double x, int n)
{
	double absx, sqrtx, nf, cn;
	absx = x;

	sqrtx = sqrt(absx);

	if (absx < 1.e-8)
	{
		nf = factorial(n);
		cn = (1.0 + x / ((n + 1.0) * (n + 2.0)) * (-1.0 + x / ((n + 3.0) * (n + 4.0)))) / nf;
	}
	else
	{
		if (x > 0)
		{
			if (n == 0)
			{
				cn = cos(sqrtx);
			}
			else if (n == 1)
			{
				cn = sin(sqrtx) / sqrtx;
			}
			else if (n == 2)
			{
				cn = (1 - cos(sqrtx)) / x;
			}
			else if (n == 3)
			{
				cn = (sqrtx - sin(sqrtx)) / (x * sqrtx);
			}
			else if (n > 3)
			{
				cn = -(Cn(x, n - 2) - 1. / factorial(n - 2)) / x;
			}
		}
		else
		{

			if (n == 0)
			{
				cn = cosh(sqrtx);
			}
			else if (n == 1)
			{
				cn = sinh(sqrtx) / sqrtx;
			}
			else if (n == 2)
			{
				cn = (cosh(sqrtx) - 1) / absx;
			}
			else if (n == 3)
			{
				cn = (sinh(sqrtx) - sqrtx) / (absx * sqrtx);
			}
			else if (n > 3)
			{
				cn = -(Cn(x, n - 2) - 1. / factorial(n - 2)) / x;
			}
		}
	}

	return cn;
}

__device__ double U(double x, double ro)
{
	static double cn1x, cn2x;

	cn1x = Cn(x, 1);
	cn2x = Cn(x, 2);

	return sqrt(1.0 - ro * cn1x / sqrt(cn2x));
}

__device__ double dU(double x, double ro, double c1, double c2, double c3, double c4, double u)
{
	static double dUc1, dUc2, dc1, dc2;

	dUc1 = -0.5 * ro / (sqrt(c2) * u);
	dUc2 = ro * c1 / (4.0 * sqrt(c2 * c2 * c2) * u);
	dc1 = 0.5 * (c3 - c2);
	dc2 = c4 - 0.5 * c3;

	return dUc1 * dc1 + dUc2 * dc2;
}

__device__ double f(double x, double ro)
{
	static double u, cn2x, cn3x;

	u = U(x, ro);

	cn2x = Cn(x, 2);
	cn3x = Cn(x, 3);

	return cn3x / sqrt(cn2x * cn2x * cn2x) * u * u * u + ro * u;
}

__device__ double fx(double x, double ro)
{
	static double u, c0, c1, c2, c3, c4, c5, dc2, dc3, du, c23, c232;

	c0 = Cn(x, 0);
	c1 = Cn(x, 1);
	c2 = Cn(x, 2);
	c3 = Cn(x, 3);
	c4 = Cn(x, 4);
	c5 = Cn(x, 5);

	dc2 = c4 - 0.5 * c3;
	dc3 = 0.5 * (3.0 * c5 - c4);

	u = U(x, ro);
	du = dU(x, ro, c1, c2, c3, c4, u);

	c23 = c2 * c2 * c2;
	c232 = sqrt(c23);

	return ((dc3 * c232 - c3 * 1.5 * sqrt(c2) * dc2) / c23 * u + 2.0 * c3 / c232 * du) * u * u +
		(c3 / c232 * u * u + ro) * du;

}

__device__ double newton(double x0, double ro, double sig, int& iter)
{
	double dx;
	iter = 0;
	dx = 1.0;

	while (std::abs(dx) > 1.e-11) {
		dx = (f(x0, ro) - sig) / fx(x0, ro);
		x0 = x0 - dx;
		iter++;
	}
	return x0;
}

__device__ double findX(int typeX, double ro, double SIG, double revs, double um)
{
	double M_PI = 3.14159265358979323846264338327950288419716939937510;
	static double x0, eps;
	x0 = 0.0;

	if (typeX == 1)
	{
		eps = cbrt((M_PI * revs / (sqrtf(2) * (SIG - ro * um)))) * um;
		x0 = 4.0 * pow((M_PI * revs + eps), 2);
	}
	else if (typeX == 2)
	{
		eps = pow((M_PI * (revs + 1) / ((2.0 / 3) * (pow(um, 3)) + SIG - ro * um)), (1.0 / 3.0)) * um;
		x0 = 4.0 * pow((M_PI * (revs + 1) - eps), 2);
	}
	else
	{
		printf("Unknown typeX, enter 1 or 2");
	}

	return x0;
}

__device__ double getX(int revs, double ro, double SIG, int typeX)
{
	double M_PI = 3.14159265358979323846264338327950288419716939937510;
	static double absRo, um, eps;

	absRo = fabs(ro);
	um = sqrt(1.0 - sqrt(2.0) * absRo);

	if (revs > 0)
	{
		return findX(typeX, ro, SIG, revs, um);
	}

	eps = pow((M_PI / (2.0 / 3.0 * pow(um, 3) + SIG - ro * um)), (1.0 / 3.0)) * um;

	return 4.0 * pow((M_PI - eps), 2);
}

__global__ void lambert(const float* r0, const float* rk, float *dt, int revs, int typeX, float* v1, float* v2)
{
	const double M_PI = 3.14159265358979323846264338327950288419716939937510;
	double R1[3], R2[3], R1R2[3];
	double r1 = 0.0, r2 = 0.0, r1r2 = 0.0;
	double fi, ro, SIG, SIGpar, x0, xSol;
	int iter, off = 3;

	int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
	int ThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
	int tid = blockIndex * blockDim.x * blockDim.y * blockDim.z + ThreadIndex;
	double t = dt[tid];

	printf("Hello");
	if (t < 0)
	{
		printf("Time is invalid");
		return;
	}

	for (int i = 0; i < 3; ++i)
	{
		R1[i] = r0[tid * off + i];
		R2[i] = rk[tid * off + i];
	}

	for (int i = 0; i < 3; ++i)
	{
		r1 += pow(R1[i], 2);
		r2 += pow(R2[i], 2);
	}

	r1 = sqrt(r1);
	r2 = sqrt(r2);

	for (int i = 0; i < 3; ++i)
	{
		r1r2 += R1[i] * R2[i];
	}

	R1R2[0] = R1[1] * R2[2] - R1[2] * R2[1];
	R1R2[1] = R1[2] * R2[0] - R1[0] * R2[2];
	R1R2[2] = R1[0] * R2[1] - R1[1] * R2[0];

	fi = acos(r1r2 / (r1 * r2));

	if (R1R2[2] < 0.0)
		fi = 2 * M_PI - fi;

	if (typeX < 0) {

		if (R1R2[2] >= 0.0) typeX = 0;
		else typeX = 1;

	}

	ro = sqrt(2.0 * r1 * r2) / (r1 + r2) * cos(fi / 2.0);

	SIG = t / pow(sqrt(r1 + r2), 3);
	SIGpar = 1.0 / 3.0 * (sqrt(2.0) + ro) * sqrt(1.0 - sqrt(2.0) * ro);

	if (SIG < SIGpar)
	{
		x0 = 0.0;
	}
	else
	{
		x0 = getX(revs, ro, SIG, typeX);
	}

	xSol = newton(x0, ro, SIG, iter);

	double cn1x = Cn(xSol, 1);
	double cn2x = Cn(xSol, 2);
	double cn3x = Cn(xSol, 3);
	double uxro = U(xSol, ro);

	double s = sqrt((r1 + r2) / cn2x) * uxro;
	double f = 1 - pow(s, 2) * cn2x / r1;
	double g = t - pow(s, 3) * cn3x;

	double df = -s * cn1x / (r1 * r2);
	double dg = 1.0 - pow(s, 2) * cn2x / r2;

	   for (int i = 0; i < 3; ++i)
	   {
	       v1[tid * off + i] = 1. / g * (R2[i] - f * R1[i]);
	       v2[tid * off + i] = df * R1[i] + dg * v1[tid * off + i];
	   }
	
	   printf("iter = %i\n", iter);
}

double** reading_data(char* name_file, double** DATA, char ADD[][1000], int ignore, int& SIZE1, int SIZE2) {

	SIZE1 = 1;
	int i = 0, j;
	DATA = (double**)malloc(SIZE1 * sizeof(double*));

	char line[1000], * tok, * next_token = NULL;


	ifstream file(name_file);
	while (file.getline(line, 1000) && ignore) {
		ignore--;
		strcpy(ADD[i], line);
		i++;
	}
	i = SIZE1 - 1;
	do {

		DATA = (double**)realloc(DATA, ++SIZE1 * sizeof(double*));
		DATA[i] = (double*)malloc(SIZE2 * sizeof(double));

		for (char* tok = strtok_s(line, " ", &next_token), j = 0; tok; tok = strtok_s(NULL, " ", &next_token)) {

			DATA[i][j] = atof(tok);
			j++;
			if (j == SIZE2) break;
		}

		i++;
		//if (i==3000) break;      	

	} while (file.getline(line, 1000));
	file.close();


	return DATA;
}

void writing_data(char* name_file, double** DATA, int SIZE1, int SIZE2, char ADD[][1000], int add) {

	int i, j;
	FILE* fileout;
	fileout = fopen(name_file, "w");

	for (i = 0; i < add; i++) fprintf(fileout, "%s\n", ADD[i]);

	for (i = 0; i < SIZE1 - 1; i++) {
		for (j = 0; j < SIZE2; j++)
			fprintf(fileout, "%26.16e", DATA[i][j]);
		fprintf(fileout, "\n");
	}
	fclose(fileout);
}

int div_up(int x, int y)
{
	return (x - 1) / y + 1;
}

void vett_cpu(const double* vet1, const double* vet2, double* prod)
{
	prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
	prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
	prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

int main() {

	cudaSetDevice(1);
	int threads = 384;
	int blocs;
	int leght = 3;

	printf("1. start program\n");

	double AU = 1.49597870691e8;
	double fMSun = 1.32712440018e11;             // km^3/sec^2

	double UnitR = AU;
	double UnitV = sqrt(fMSun / UnitR);          // km/sec
	double UnitT = (UnitR / UnitV) / 86400;      // day

	double mu = 1.;							// гравитационная постоянная
	int nrev = 0;							// число витков
	int lw = -1;

	double dv1[3], dv2[3], dV1, dV2;

	char name_file[] = { "data1.txt" };
	char name_file2[] = { "data1_izzo_cpu.txt" };
	double** DATA = NULL;
	int i, k, SIZE1, SIZE2 = 29;
	char boof[2][1000];
	double R0[3] = { 0.0,0.0,0.0 };

	printf("2. reading file \n");

	DATA = reading_data(name_file, DATA, boof, 2, SIZE1, SIZE2);

	printf("3. finish reading file\n");
	printf("4. count tasks %i \n", SIZE1);
	printf("5. start calculate \n");

	blocs = div_up(SIZE1, threads);
	const int countTasks = SIZE1;


	printf("6. Blocs = %i, Threads = %i \n", blocs, threads);

	int sizeCudaVariableBig = (SIZE1 - 1) * 3;
	int sizeCudaVariableSmall = (SIZE1 - 1);

	float* r0 = new float[sizeCudaVariableBig];
	float* r1 = new float[sizeCudaVariableBig];
	float* v1 = new float[sizeCudaVariableBig];
	float* v2 = new float[sizeCudaVariableBig];
	float* dt = new float[sizeCudaVariableSmall];
	int* lww = new int[sizeCudaVariableSmall];
	int lenght = 3;
	int off = 3;

	float* dev_r0, * dev_r1, * dev_v1, * dev_v2, * dev_dt;
	int* dev_lw;

	for (int n = 0; n < SIZE1 - 1; n++)
	{
		dt[n] = DATA[n][14] / UnitT;

		vett_cpu(&DATA[n][0], &DATA[n][6], R0);
		if (R0[2] >= 0.0) lww[n] = 0;
		else lww[n] = 1;

		for (int m = 0; m < lenght; m++)
		{
			r0[n * off + m] = DATA[n][m];
			r1[n * off + m] = DATA[n][6 + m];
		}
	}

	cudaMalloc((void**)&dev_r0, sizeCudaVariableBig * sizeof(float));
	cudaMalloc((void**)&dev_r1, sizeCudaVariableBig * sizeof(float));
	cudaMalloc((void**)&dev_v1, sizeCudaVariableBig * sizeof(float));
	cudaMalloc((void**)&dev_v2, sizeCudaVariableBig * sizeof(float));
	cudaMalloc((void**)&dev_dt, sizeCudaVariableSmall * sizeof(float));
	cudaMalloc((void**)&dev_lw, sizeCudaVariableSmall * sizeof(int));

	cudaMemcpy(dev_r0, r0, sizeCudaVariableBig * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r1, r1, sizeCudaVariableBig * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dt, dt, sizeCudaVariableSmall * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lw, lww, sizeCudaVariableSmall * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float gpuTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	lambert << <blocs, threads >> > (dev_r0, dev_r1, dev_dt, nrev, lw, dev_v1, dev_v2);

	printf("7. Time one on GPU = %16.10e miliseconds\n", (gpuTime / (SIZE1 - 1)));
	printf("7.1. Time on GPU = %16.10e miliseconds\n", gpuTime);

	cudaMemcpy(v1, dev_v1, sizeCudaVariableBig * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(v2, dev_v2, sizeCudaVariableBig * sizeof(float), cudaMemcpyDeviceToHost);

	printf("8. finish calculate \n");

	for (int n = 0; n < SIZE1 - 1; n++)
	{
		for (int m = 0; m < lenght; m++)
		{
			r0[n * off + m] = DATA[n][m];
			r1[n * off + m] = DATA[n][6 + m];
		}
	}

	for (i = 0; i < SIZE1 - 1; i++) {

		dV1 = 0; dV2 = 0;
		for (k = 0; k < 3; k++) {
			dv1[k] = DATA[i][18 + k] - v1[i * off + k];
			dv2[k] = DATA[i][21 + k] - v2[i * off + k];
			dV1 += dv1[k] * dv1[k];
			dV2 += dv2[k] * dv2[k];
		}

		dV1 = sqrt(dV1) * UnitV;
		dV2 = sqrt(dV2) * UnitV;

		DATA[i][SIZE2 - 2] = DATA[i][15] - dV1;
		DATA[i][SIZE2 - 1] = DATA[i][16] - dV2;

		DATA[i][15] = dV1;
		DATA[i][16] = dV2;
		DATA[i][17] = DATA[i][15] + DATA[i][16];

	}

	printf("9. start write data \n");

	writing_data(name_file2, DATA, SIZE1, SIZE2, boof, 2);

	printf("10. finish write data \n");

	cudaFree(dev_r0);
	cudaFree(dev_r1);
	cudaFree(dev_v1);
	cudaFree(dev_v2);
	cudaFree(dev_lw);
	cudaFree(dev_dt);


	printf("11. finish program");

	return 0;
}