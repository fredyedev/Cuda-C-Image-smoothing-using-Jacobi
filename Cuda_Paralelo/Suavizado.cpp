#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "Suavizado.h"
using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {

      if (argc < 5) {
          cout << "Usage: " << argv[0] << "image_filename max_iter lambda id" << endl;
          return -1;
      }

      Mat img;

      // Leemos la imagen - B&N
      img = imread(argv[1], IMREAD_GRAYSCALE);
      // Verificamos la lectura
      if (img.empty()) { cout << "Can't open image [" << argv[1] << "]"; return -1; }
      // Convertimos a float con un rango de [0, 1]
      img.convertTo(img, CV_32FC1, 1.f / 255.f);
      // Creamos la imagen de salida
      Mat img_out(img.rows, img.cols, CV_32FC1);

      // Secuencial
      // End

  	  double secs;
  	  clock_t begin, end;

      // CUDA
  	  begin = clock();

      float *img_host, *img_out_host;
      size_t size = img.rows * img.cols * sizeof(float);

      img_host = (float *)malloc(size);
      img_out_host = (float *)malloc(size);

  	  memcpy(img_host, img.data, size);
      // memcpy(img_out_host, img_out.data, size);

      launch_kernel(img_out_host, img_host, img.rows, img.cols, atoi(argv[2]), atof(argv[3]));

      memcpy(img_out.data, img_out_host, size);

      double min, max;
      minMaxIdx(img_out, &min, &max);
      if (min != max) img_out.convertTo(img_out, CV_8U, 255. / (max - min), -255. * min / (max - min));

  	 char filename[100];
  	 sprintf(filename,"SV_%s_%s_%d.png", argv[2], argv[3], atoi(argv[4]));
     imwrite(filename, img_out);

  	 free(img_host);
  	 free(img_out_host);

  	 end = clock();
  	 secs = (end - begin) / (float)CLOCKS_PER_SEC;

  	 cout << "Parallel: " << secs << endl;
      // End

      return 0;
}



// #include <stdio.h>
// #include <cuda_runtime.h>
// #include <opencv2/highgui/highgui.hpp>
// #include "Suavizado.h"
// using namespace std;
// using namespace cv;
//
// int iDivUp(int a,int  b) {
// 	int idup=( (a%b)!=0 )? ( a/b + 1 ) : ( a/b );
// 	return idup;
// }
//
// int main(int argc,char **argv){
//   int mxitr;
//   float lambda;
//   int idimg;
//   mxitr = atoi(argv[1]);
//   lambda = atof(argv[2]);
//   idimg = atoi(argv[4]);
//
// 	cudaSetDevice(0);
// 	//host variables
// 	int imageW,imageH;
//  	Mat img_h;
//  	Mat imgEs_h;
// 	float gpu_elapsed_time_ms;
// 	cudaEvent_t start, stop;
//
// 	//device varaibles
//   float *img_dev,*imgEs_dev,imgEprev_dev,*imgEnext_dev;
//   // float *imgEs_h,*imgEprev_h;
//   //float3
// 	char name_imag[500];
//
// 	sprintf(name_imag,argv[3]);
// 	img_h = imread(name_imag,0);
//
// 	//printf("step1");
// 	imageW = img_h.cols;
// 	imageH = img_h.rows;
// 	size_t sizef = imageW*imageH*sizeof(float);
//
// 	imgEs_h.create(imageH,imageW,CV_32FC(1));
// 	imgEprev_h.create(imageH,imageW,CV_32FC(1));
// //  imgEnext_h.create(imageH,imageW,CV_32FC(1));
//
// 	//create device memory
// 	cudaMalloc((void **)&img_dev,sizef);
// 	cudaMalloc((void **)&imgEs_dev,sizef);
// 	cudaMalloc((void **)&imgEprev_dev,sizef);
//
// 	//Copy Memory (hst - dev)
// 	cudaMemcpy(img_dev,img_h.data,size,cudaMemcpyHostToDevice);
//   cudaMemcpy(imgEprev_dev,img_h.data,size,cudaMemcpyHostToDevice);
// 	//printf("step4");
// 	dim3 threads(32,32,1);
// 	dim3 grid(iDivup(imageW,threads.x),iDivup(imageH,threads.y),1);
//
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);
// 	cudaEventRecord(start, 0);
//   //launch the kernel
//   for (int i=0; i < mxitr; i++) {
//       if(k%2)
//           CUDA_Suavizado_Image(imgEs_dev,img_dev,imgEprev_dev,lambda,imageW,imageH,grid,threads);
//       else
//           CUDA_Suavizado_Image(imgEprev_dev,img_dev,imgEs_dev,lambda,imageW,imageH,grid,threads);
//   }
//
// 	cudaEventRecord(stop, 0);
// 	cudaEventSynchronize(stop);
// 	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
// 	printf("\n Time on GPU: %f\n\n", gpu_elapsed_time_ms);
//
//   if  ((mxitr-1)%2 == 0 ) {
//       cudaMemcpy(imgEs_h.data,imgEs_dev,sizef,cudaMemcpyDeviceToHost);
//   }
//   else {
//       cudaMemcpy(imgEs_h.data,imgEprev_dev,sizef,cudaMemcpyDeviceToHost);
//   }
//   sprintf(name_imag,"%sSV%d.png",argv[3],idimg);
// 	imwrite(name_imag,imgEs_h);
// 	//destroyWindows("Original Frame");
// 	imgEs_h.release();
//
// 	cudaFree(img_dev);
// 	cudaFree(imgEprev_h);
// 	cudaFree(imgEs_h);
//   free(imgEs_h);
//   free(imgEprev_h);
//     return 0;
//  }
