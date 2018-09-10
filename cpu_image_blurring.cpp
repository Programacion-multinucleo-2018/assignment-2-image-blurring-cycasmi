/*
 * File:   cpu_image_blurring.cpp
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 * Created on September 9th, 2018, 01:33 PM
 */

//g++ cpu_image_blurring.cpp `pkg-config --cflags --libs opencv`

#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

 using namespace std;

void OMP_blur_image(const cv::Mat& M_input, cv::Mat& M_output)
{
	int colorWidthStep = static_cast<int>(M_input.step);
	size_t inputBytes = M_input.step*M_input.rows;
	unsigned char *input, *output;
	output = input = (unsigned char *) malloc(inputBytes*sizeof(unsigned char));

	memcpy(input, M_input.ptr(), inputBytes*sizeof(unsigned char));

	//pixel margin for blur matrix
	const unsigned int marginSize = 2;

	//Output pixels
	float out_blue = 0;
	float out_green = 0;
	float out_red = 0;

	int index, out_index;
	for (int i = 0; i < M_input.cols; i++)
	{
		out_blue = 0;
		out_green = 0;
		out_red = 0;
		for (int j = 0; j < M_input.rows; j++)
		{

			if ((i >= marginSize) && (j >= marginSize) && (i < M_input.cols - marginSize) && (j < M_input.rows - marginSize))
			{
				index = 0;
				#pragma omp parallel for collapse(2) default(shared) reduction (+:out_blue, out_green, out_red)
				//Average pixel color calculation
				for (int m_i = i - marginSize; m_i <= i + marginSize; m_i++)
				{
					for (int m_j = j - marginSize; m_j <= j + marginSize; m_j++)
					{
						index = m_j * colorWidthStep + (3 * m_i);
						out_blue = out_blue + input[index];
						out_green = out_green + input[index + 1];
						out_red = out_red + input[index + 2];
					}
				}
				out_blue /= 25;
				out_green /= 25;
				out_red /= 25;
			}
			else
			{
				index = j * colorWidthStep + (3 * i);
				out_blue = input[index];
				out_green = input[index + 1];
				out_red = input[index + 2];
			}
			out_index = j * colorWidthStep + (3 * i);
			output[out_index] = static_cast<unsigned char>(out_blue);
			output[out_index+1] = static_cast<unsigned char>(out_green);
			output[out_index+2] = static_cast<unsigned char>(out_red);
		}
	}

	memcpy(M_output.ptr(), output, inputBytes*sizeof(unsigned char));

	//Save resultant image
	cv::imwrite("OMP_Altered_Image.jpg", M_output);
}

void CPU_blur_image(const cv::Mat& M_input, cv::Mat& M_output)
{
	int colorWidthStep = static_cast<int>(M_input.step);
	size_t inputBytes = M_input.step*M_input.rows;
	unsigned char *input, *output;
	output = input = (unsigned char *) malloc(inputBytes*sizeof(unsigned char));

	memcpy(input, M_input.ptr(), inputBytes*sizeof(unsigned char));

	//pixel margin for blur matrix
	const unsigned int marginSize = 2;

	//Output pixels
	float out_blue = 0;
	float out_green = 0;
	float out_red = 0;

	int index, out_index;
	for (int i = 0; i < M_input.cols; i++)
	{
		out_blue = 0;
		out_green = 0;
		out_red = 0;
		for (int j = 0; j < M_input.rows; j++)
		{

			if ((i >= marginSize) && (j >= marginSize) && (i < M_input.cols - marginSize) && (j < M_input.rows - marginSize))
			{
				index = 0;
				//Average pixel color calculation
				for (int m_i = i - marginSize; m_i <= i + marginSize; m_i++)
				{
					for (int m_j = j - marginSize; m_j <= j + marginSize; m_j++)
					{
						index = m_j * colorWidthStep + (3 * m_i);
						out_blue += input[index];
						out_green += input[index + 1];
						out_red += input[index + 2];
					}
				}
				out_blue /= 25;
				out_green /= 25;
				out_red /= 25;
			}
			else
			{
				index = j * colorWidthStep + (3 * i);
				out_blue = input[index];
				out_green = input[index + 1];
				out_red = input[index + 2];
			}
			out_index = j * colorWidthStep + (3 * i);
			output[out_index] = static_cast<unsigned char>(out_blue);
			output[out_index+1] = static_cast<unsigned char>(out_green);
			output[out_index+2] = static_cast<unsigned char>(out_red);
		}
	}

	memcpy(M_output.ptr(), output, inputBytes*sizeof(unsigned char));

	//Save resultant image
	cv::imwrite("CPU_Altered_Image.jpg", M_output);
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

	cv::Mat output(input.rows, input.cols, input.type());

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	// NO THREADS CPU TEST
	chrono::duration<float, std::milli> duration_ms = chrono::high_resolution_clock::duration::zero();
	auto start =  chrono::high_resolution_clock::now();
	
	CPU_blur_image(input, output);

	auto end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("CPU image blurring elapsed %f ms\n\n", duration_ms.count());

	/* ********* DISPLAY IMAGES **********/
	/*//Allow the windows to resize
	namedWindow("CPU INPUT", cv::WINDOW_NORMAL);
	namedWindow("CPU OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("CPU INPUT", input);
	imshow("CPU OUTPUT", output);

	//Wait for key press
	cv::waitKey();
	*/

	// OMP CPU TEST
	duration_ms = chrono::high_resolution_clock::duration::zero();
	start =  chrono::high_resolution_clock::now();
	
	OMP_blur_image(input, output);

	end =  chrono::high_resolution_clock::now();
	duration_ms = end - start;
	printf("OMP image blurring elapsed %f ms\n\n", duration_ms.count());

	/* ********* DISPLAY IMAGES **********/
	/*//Allow the windows to resize
	namedWindow("OMP INPUT", cv::WINDOW_NORMAL);
	namedWindow("OMP OUTPUT", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("OMP INPUT", input);
	imshow("OMP OUTPUT", output);

	//Wait for key press
	cv::waitKey();
	*/

	return 0;	
}