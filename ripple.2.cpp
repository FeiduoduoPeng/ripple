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
#include<ctime>
#include<iostream>

#define DIM 1024
#define PI 3.1415926535897932f

void kernel( unsigned char *ptr, int ticks ) {
    // map from threadIdx/BlockIdx to pixel position
    for(int i=0; i<DIM*DIM; i++){
        int x = i%DIM;
        int y = i/DIM;

        // now calculate the value at that position
        float fx = x - DIM/2;
        float fy = y - DIM/2;
        float d = sqrtf( fx * fx + fy * fy );
        //unsigned char grey = (unsigned char)(x);
        unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                            cos(d/10.0f - ticks/7.0f) /
                                            (d/10.0f + 1.0f));
        ptr[i*4 + 0] = grey;
        ptr[i*4 + 1] = grey;
        ptr[i*4 + 2] = grey;
        ptr[i*4 + 3] = 255;
    }
}

int main( void ) {
    cv::Mat_<cv::Vec3b> img(DIM, DIM);

    unsigned char ptrs[4*DIM*DIM];

    clock_t begin_ = clock();
    for(int time=0; time<100; time++)
    {
        kernel(ptrs, time);
        for(int i=0; i< img.rows; i++){
            for(int j=0; j<img.cols; j++){
                for(int ch=0; ch<3; ch++)
                    img.at<cv::Vec3b>(i,j)[ch]=ptrs[ 4*(j*DIM+i) + ch];
            }
        }
        cv::imshow("test", img);
        cv::waitKey(1);
    }

    clock_t end_ = clock();

    std::cout<<"elapsed: "<<end_ - begin_<<std::endl;
    return 0;
}
