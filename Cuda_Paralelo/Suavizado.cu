#include "Suavizado.h"

__global__
void Suavizado_Jacobi(float* fnext, float* img, float* fprev, float lmd, int imageH, int imageW) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;
    int index;
    index = idy * imageW + idx;

    if( idx < imageW - 1 && idx > 0  && idy < imageH - 1 && idy > 0) {


      sum = fprev[(idy+1) * imageW + idx] + fprev[(idy-1) * imageW + idx] +
      fprev[idy * imageW + (idx + 1)] + fprev[idy * imageW + (idx-1)];

      fnext[index] = (float) (img[index] + lmd* sum) / (1 + 4*lmd);

    }
    else if ( idx < imageW  && idy < imageH ) {

        if ( idy == 0 ) { // top row
            if ( idx == 0 ) {
                sum = fprev[(idy+1) * imageW + idx] + fprev[idy * imageW + (idx+1)]; // ok
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
            }
            else if ( idx == imageW - 1 )  {
                sum = fprev[(idy+1) * imageW + idx] + fprev[idy * imageW + (idx-1)]; // Ok
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
            }
            else {
                sum = fprev[(idy+1) * imageW + idx] +fprev[idy * imageW + (idx+1)] +  //checked
                      fprev[idy * imageW + (idx-1)];
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
            }
        }
        else if ( idy == imageH - 1 ) {// botton row
        	if ( idx == 0 ) {
                sum = fprev[(idy-1) * imageW + idx] + fprev[idy * imageW + (idx+1)]; //ok
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
        	}
        	else if ( idx == imageW - 1 ) {
                sum = fprev[(idy-1) * imageW + idx] + fprev[idy * imageW + (idx-1)]; //ok
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
        	}
        	else {
                sum = fprev[(idy-1) * imageW + idx] +
                      fprev[idy * imageW + (idx+1)] + fprev[idy * imageW + (idx-1)];
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
        	}
        }
        else { //left column
        	if ( idx == 0 ) {
                sum = fprev[(idy-1) * imageW + idx] + fprev[(idy+1) * imageW + idx] +
                      fprev[idy * imageW + (idx+1)];
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
        	}
        	else {//right column
                sum = fprev[(idy-1) * imageW + idx] + fprev[(idy+1) * imageW + idx] +
                      fprev[idy * imageW + (idx-1)];
                fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
        	}
        }
    }
}

void launch_kernel(float *dst, float *src, int height, int width, int max_iter, float lambda) {

    float *src_dev, *dst_dev, *f_next, *f_prev;
    size_t size = height * width * sizeof(float);

    cudaMalloc((void **) &src_dev, size);
    cudaMalloc((void **) &dst_dev, size);
    cudaMalloc((void **) &f_next, size);
    cudaMalloc((void **) &f_prev, size);

    cudaMemcpy(src_dev, src, size, cudaMemcpyHostToDevice);
    cudaMemcpy(f_prev, dst_dev, size, cudaMemcpyHostToDevice);

    dim3 threads(32, 32, 1);
    dim3 grid(div_up(width, threads.x), div_up(height, threads.y), 1);

    int i;
    for (i = 0; i < max_iter; ++i) {
        if (i % 2 == 0)
            Suavizado_Jacobi <<< grid, threads >>> (f_next, src_dev, f_prev, lambda, height, width);
        else
            Suavizado_Jacobi <<< grid, threads >>> (f_prev, src_dev, f_next, lambda, height, width);

        //cudaDeviceSynchronize(); //----->  Sugerencia del profesor.
    }
    if ((max_iter - 1) % 2 == 0)
        cudaMemcpy(dst, f_next, size, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(dst, f_prev, size, cudaMemcpyDeviceToHost);

    cudaFree(src_dev);
    cudaFree(dst_dev);
    cudaFree(f_next);
    cudaFree(f_prev);
}

int div_up(int size, int block_size) {

	return size / block_size + (size % block_size == 0 ? 0 : 1);

}


// #include "Suavizado.h"
//
// __global__ void Suavizado_Jacobi(float* fnext, float* img, float* fprev, float lmd, int imageH, int imageW) {
//
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     float sum = 0.0;
//     int index;
//     index = idy * imageW + idx;
//
//     if( idx < imageH - 1 && idx > 0  && idy < imageW - 1 && idy > 0)  {
//
//
//       sum = fprev[(idy+1) * imageW + idx] + fprev[(idy-1) * imageW + idx] +
//       fprev[idy * imageW + (idx + 1)] + fprev[idy * imageW + (idx-1)];
//
//       fnext[index] = (float) (img[index] + lmd* sum) / (1 + 4*lmd);
//
//     }
//     else if ( idx < imageH  && idy < imageW ) {
//
//         if ( idy == 0 ) { // top row
//             if ( idx == 0 ) {
//                 sum = fprev[(idy+1) * imageW + idx] + fprev[idy * imageW + (idx+1)]; // ok
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
//             }
//             else if ( idx == imageW - 1 )  {
//                 sum = fprev[(idy+1) * imageW + idx] + fprev[idy * imageW + (idx-1)]; // Ok
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
//             }
//             else {
//                 sum = fprev[(idy+1) * imageW + idx] +fprev[idy * imageW + (idx+1)] +  //checked
//                       fprev[idy * imageW + (idx-1)];
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
//             }
//         }
//         else if ( idy == imageH - 1 ) {// botton row
//         	if ( idx == 0 ) {
//                 sum = fprev[(idy-1) * imageW + idx] + fprev[idy * imageW + (idx+1)]; //ok
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
//         	}
//         	else if ( idx == imageW - 1 ) {
//                 sum = fprev[(idy-1) * imageW + idx] + fprev[idy * imageW + (idx-1)]; //ok
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 2*lmd );
//         	}
//         	else {
//                 sum = fprev[(idy-1) * imageW + idx] +
//                       fprev[idy * imageW + (idx+1)] + fprev[idy * imageW + (idx-1)];
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
//         	}
//         }
//         else { //left column
//         	if ( idx == 0 ) {
//                 sum = fprev[(idy-1) * imageW + idx] + fprev[(idy+1) * imageW + idx] +
//                       fprev[idy * imageW + (idx+1)];
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
//         	}
//         	else {//right column
//                 sum = fprev[(idy-1) * imageW + idx] + fprev[(idy+1) * imageW + idx] +
//                       fprev[idy * imageW + (idx-1)];
//                 fnext[index] = (float) ( img[index] + lmd* sum ) / ( 1 + 3*lmd );
//         	}
//         }
//     }
//
// }
//
// void CUDA_Suavizado_Image(float *fnext,float *img,float *fprev,float lmd,int imageW,int imageH,dim3 grid,dim3 threads) {
//   Suavizado_Jacobi<<<grid,threads>>>(fnext,img,fprev,lmd,imageH,imageW);
// }
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// //
// //
// //
// //
// // fnext[idx*imageW+idy] =
// // if (idx < imageH)
// // {
// //     float sigma = 0.0;
// //
// //     int idx_Ai = idx*imageW;
// //
// //     for (int j=0; j<imageW; j++)
// //         if (idx != j)
// //             sigma += img[idx_Ai + j] * fprev[j];
// //
// //     fnext[idx] = (b[idx] - sigma) / img[idx_Ai + idx];
// // }
