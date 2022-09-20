#ifndef SUAVIZADO_H
#define SUAVIZADO_H

void launch_kernel(float *dst, float *src, int height, int width, int max_iter, float lambda);
int div_up(int size, int block_size);

#endif // SUAVIZADO_H

// __global__ void Suavizado_Jacobi(float* fnext, float* img, float* fprev, float lmd, int imageH, int imageW);
// void CUDA_Suavizado_Image(float *fnext,float *img,float *fprev,float lmd,int imageW,int imageH,dim3 grid,dim3 threads);
