/*
 * File:   image_blurring.cu
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 * Created on September 8th, 2018, 01:13 PM
 */

#include <iostream>
#include <cstdio>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../common/common.h"
#include <cuda_runtime.h>

using namespace std;

// input - input image one dimensional array
// ouput - output image one dimensional array
// width, height - width and height of the images
// colorWidthStep - number of color bytes (cols * colors)
// grayWidthStep - number of gray bytes 
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep)
{
	//pixel margin for blur matrix
	const unsigned int marginSize = 2;

	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Location of colored output pixel
		int output_tid = yIndex * colorWidthStep + (3 * xIndex);

		//Output pixels
		float out_blue = 0;
		float out_green = 0;
		float out_red = 0;

		//If pixel is inside the margins, blur it
		if ((xIndex >= marginSize) && (yIndex >= marginSize) && (xIndex < width - marginSize) && (yIndex < height - marginSize))
		{
			int index = 0;
			
			//Average pixel color calculation
			for (int i = xIndex - marginSize; i <= xIndex + marginSize; i++)
			{
				for (int j = yIndex - marginSize; j <= yIndex + marginSize; j++)
				{
					index = j * colorWidthStep + (3 * i);
					out_blue += input[index];
					out_green += input[index + 1];
					out_red += input[index + 2];
				}
			}
			out_blue /= 25;
			out_green /= 25;
			out_red /= 25;
		}
		//If pixel is out of margin, copy original image color
		else
		{
			//Location of colored input pixel
			int input_tid = yIndex * colorWidthStep + (3 * xIndex);
			out_blue = input[input_tid];
			out_green = input[input_tid + 1];
			out_red = input[input_tid + 2];
		}

		output[output_tid] = static_cast<unsigned char>(out_blue);
		output[output_tid+1] = static_cast<unsigned char>(out_green);
		output[output_tid+2] = static_cast<unsigned char>(out_red);
	}
}

void blur_image(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors	
	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const im3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
	
	// Launch the color conversion kernel
	blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), static_cast<int>(output.step));

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Save resultant image
	cv::imwrite("Altered_Image.jpg", output);

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[])
{
	string imagePath;
	
	if(argc < 2)
		imagePath = "new_image.jpg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	cv::Mat output
	(input.rows, input.cols, input.type());

	//Call the wrapper function
	blur_image(input, output);

	/* ********* DISPLAY IMAGES **********/
	//Allow the windows to resize
	//namedWindow("Input", cv::WINDOW_NORMAL);
	//namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	//imshow("Input", input);
	//imshow("Output", output);

	//Wait for key press
	//cv::waitKey();

	return 0;
}