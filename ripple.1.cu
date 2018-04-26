/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<memory>
#include "cuda.h"
#include <stdio.h>
#include<iostream>

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel( unsigned char *ptr, int ticks ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf( fx * fx + fy * fy );
    //unsigned char grey = (unsigned char)(x);
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/10.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));    
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}

int main( void ) {
    cv::Mat_<cv::Vec3b> img(DIM, DIM);
  
    unsigned char ptrs[4*DIM*DIM];

    unsigned char* dev_bitmap;
    cudaMalloc( (void**)&dev_bitmap, 4*DIM*DIM* sizeof(unsigned char) ) ;

    dim3 blocks(DIM/16,DIM/16);
    dim3 threads(16, 16);

    clock_t begin_ = clock();
    for(int time=0; time<100; time++)
    {
        kernel<<<blocks, threads>>>(dev_bitmap, time);
        cudaMemcpy(ptrs, dev_bitmap, 4*DIM*DIM*sizeof(unsigned char), cudaMemcpyDeviceToHost);

        for(int i=0; i< img.rows; i++){
            for(int j=0; j<img.cols; j++){
                for(int ch=0; ch<3; ch++)
                    img.at<cv::Vec3b>(i,j)[ch]=ptrs[ 4*(j*DIM+i) + ch];
            }
        }
        cv::imshow("test", img);
        cv::waitKey(1);
    }
    cudaFree(dev_bitmap);

    clock_t end_ = clock();
    std::cout<<"elapsed: "<<end_ - begin_<<std::endl;
    return 0;
}
