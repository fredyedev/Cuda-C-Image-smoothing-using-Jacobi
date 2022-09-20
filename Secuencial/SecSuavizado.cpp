#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <ctime>
using namespace std;
using namespace cv;

void Suavizado(float* fnext, float* img, float* fprev, float lmd, int imageH, int imageW) {
      float sum = 0.0;
      int index;
      for (int idy=0; idy < imageH; idy++ ) {
          for(int idx=0 ; idx < imageW; idx++) {

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

            //**************************************
          }//for row
      } // for col
}
int main(int argc, char *argv[])  {

    int max_iter;
    float lambda;
    clock_t begin, end;
    double secs;
    max_iter = atoi(argv[2]);
    lambda = atof(argv[3]);
    Mat img;
    float *imgtmp;
    Mat img1;
    // Leemos la imagen - B&N
    img = imread(argv[1], IMREAD_GRAYSCALE);
    // Verificamos la lectura
    if (img.empty()) { cout << "Can't open image [" << argv[1] << "]"; return -1; }
    // Convertimos a float con un rango de [0, 1]
    img1.create(img.rows,img.cols,CV_32FC1);
    img.convertTo(img1, CV_32FC1, 1.f / 255.f);

    // Creamos la imagen de salida
    Mat img_out(img.rows, img.cols, CV_32FC1);
    float *f_next, *f_prev;
    int height = img.rows, width=img.cols;
    size_t size = height * width*sizeof(float);
    imgtmp = new float[size];
    f_next = new float[size];
    f_prev = new float[size];
    memcpy(imgtmp, img1.data, size);
    memcpy(f_prev,img1.data,size);
    begin = clock();
      for (int i = 0; i < max_iter; ++i) {
          if (i % 2 == 0)
              Suavizado(f_next, imgtmp , f_prev, lambda, height, width);
          else
              Suavizado(f_prev, imgtmp , f_next, lambda, height, width);
      }
      end = clock();
      secs = (end - begin) / (float)CLOCKS_PER_SEC;
      cout << "Serial: " << secs << endl;

      if ((max_iter - 1) % 2 == 0)
          memcpy(img_out.data, f_next, size);
      else
          memcpy(img_out.data, f_prev, size);


      double min, max;
      minMaxIdx(img_out, &min, &max);
      if (min != max) img_out.convertTo(img_out, CV_8U, 255. / (max - min), -255. * min / (max - min));
      char filename[100];
      sprintf(filename,"SV_%s_%s_%d.png", argv[2], argv[3], atoi(argv[4]));
      imwrite(filename, img_out);

      delete [] f_prev;
      delete [] f_next;

      return 0;
}//main
