#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

void print_help() {
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -f : input image file (default: test.pgm)" << endl;
	cerr << "  -h : print this message" << endl;
}
//This code was developed using Tutorial 2 as a foundation and was appropriately edited and built upon to carry out this task.
//The first step was to implement a histogram kernel. The hist_simple kernel from tutorial 3 was used as a template. The size of H was predefined as 256 as the pixel values always range from 0 to 255, and the buffer was initialised to 0 using enqueueFillBuffer.
//A histogram is a simple example of the scatter pattern which writes data into output locations indicated by an index array.
//The first purpose of plotting this histogram was to identify the distribution of pixel values in the image. Since the majority of pixels in the image fell between the same small range of values, the image is of very low contrast.
//The histogram implemented uses global memory and atomic operators, which deal with race conditions but serialise the access to global memory and are therefore slow. However, they were appropriate for this task as the amount of data is relatively small, so the execution times remained fast.
//The next step was to create a cumulative histogram of this which plots the total number of pixels in the image against the pixel values. So, by 255, all pixels in the image have been counted. This was done by using the scan_add_atomic kernel from Tutorial 3 and adapting it appropriately.
//This is an example of the Scan pattern which computes all partial reductions of a collection so that every element of the output is a reduction of all the elements of the input up to the position of that output element.
//Then the cumulative histogram needed to be normalised to create a lookup table (LUT) of new values for the pixels to increase the contrast of the image. This was done by creating a kernel to multiply each value in the cumulative histogram by 255/total pixels. This resulted in values ranging from 0-255
//This is an example of the map pattern. Map applies ‘elemental function’ to every element of data – the result is a new collection of the same shape as the input. The result does not depend on the order in which various instances of the function are executed.
//The final kernel was also an example of the map pattern. This time, the kernel simply changes each pixel in the input image to the values defined in the LUT.
//The execution time and memory transfer of each kernel is also logged and printed with each resulting histogram.
//The result of this program is an output image with a higher, more balanced contrast than the input image. The code works on greyscale .pgm images of varying sizes.
//By Gabriella Di Gregorio DIG15624188

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platformID = 0;
	int deviceID = 0;
	string imageName = "test_large.pgm"; //Change this to change the input image. Images available: test.pgm, test_large.pgm, Einstein.pgm, cat.pgm

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { imageName = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> inputImage(imageName.c_str());
		CImgDisplay displayInput(inputImage,"input");

		//a 3x3 convolution mask implementing an averaging filter
		vector<float> convolution = { 1.f / 9, 1.f / 9, 1.f / 9,
									  1.f / 9, 1.f / 9, 1.f / 9,
									  1.f / 9, 1.f / 9, 1.f / 9 };

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platformID, deviceID);

		//display the selected device
		cout << "Runing on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input

		typedef int mytype;
		vector<mytype> H(256);
		size_t histsize = H.size() * sizeof(mytype);

		//device - buffers
		cl::Buffer devInputImage(context, CL_MEM_READ_ONLY, inputImage.size());
		cl::Buffer devOutputHistogram(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer devOutputCHistogram(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer devLUT(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer devOutputImage(context, CL_MEM_READ_WRITE, inputImage.size()); //should be the same as input image

		//Part 4 - device operations

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(devInputImage, CL_TRUE, 0, inputImage.size(), &inputImage.data()[0]);
		queue.enqueueFillBuffer(devOutputHistogram, 0, 0, histsize);

		//4.2 Setup and execute the kernel (i.e. device code)

		//The first kernel call plots a histogram of the frequency of each pixel value (0-255) in the picture
		cl::Kernel kernelSimpleHist = cl::Kernel(program, "simpleHist");
		kernelSimpleHist.setArg(0, devInputImage);
		kernelSimpleHist.setArg(1, devOutputHistogram);

		cl::Event profEvent;

		queue.enqueueNDRangeKernel(kernelSimpleHist, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &profEvent);
		queue.enqueueReadBuffer(devOutputHistogram, CL_TRUE, 0, histsize, &H[0]);

		vector<mytype> CH(256);

		queue.enqueueFillBuffer(devOutputCHistogram, 0, 0, histsize);

		//The second kernel call plots a cumulative histogram of the total pixels in the picture across pixel values 0-255, so by 255, all pixels have been counted
		cl::Kernel kernelCHist = cl::Kernel(program, "cumulativeHist");
		kernelCHist.setArg(0, devOutputHistogram);
		kernelCHist.setArg(1, devOutputCHistogram);

		cl::Event profEventTwo;

		queue.enqueueNDRangeKernel(kernelCHist, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &profEventTwo);
		queue.enqueueReadBuffer(devOutputCHistogram, CL_TRUE, 0, histsize, &CH[0]);

		vector<mytype> LUT(256);

		queue.enqueueFillBuffer(devLUT, 0, 0, histsize);

		//The third kernel call creates a new histogram that will serve as a look up table of the new pixel vales. It does this by normalising the cumulative histogram, essentially decreasing the value of the pixels to increase the contrast
		cl::Kernel kernelLUT = cl::Kernel(program, "LUT");
		kernelLUT.setArg(0, devOutputCHistogram);
		kernelLUT.setArg(1, devLUT);

		cl::Event profEventThree;

		queue.enqueueNDRangeKernel(kernelLUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &profEventThree);
		queue.enqueueReadBuffer(devLUT, CL_TRUE, 0, histsize, &LUT[0]);

		//The last kernel assigns the new pixel values from the lookup table to the output image, so that the output is of higher contrast than the input
		cl::Kernel kernelReproject = cl::Kernel(program, "ReProject");
		kernelReproject.setArg(0, devInputImage);
		kernelReproject.setArg(1, devLUT);
		kernelReproject.setArg(2, devOutputImage);

		cl::Event profEventFour;

		//The values from each histogram are printed, along with the kernel execution times and memory transfer of each kernel.
		vector<unsigned char> output_buffer(inputImage.size());
		queue.enqueueNDRangeKernel(kernelReproject, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &profEventFour);
		queue.enqueueReadBuffer(devOutputImage, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		cout << endl;
		cout << "Histogram = " << H << endl;
		cout << "Histogram kernel execution time [ns]: " << profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEvent, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << "Cumulative Histogram = " << CH << endl;
		cout << "Cumulative Histogram kernel execution time [ns]: " << profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventTwo.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEventTwo, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << "LUT = " << LUT << endl;
		cout << "LUT kernel execution time [ns]: " << profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventThree.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEventThree, ProfilingResolution::PROF_US) << endl;
		cout << endl;


		cout << "Vector kernel execution time [ns]: " << profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventFour.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEventFour, ProfilingResolution::PROF_US) << endl;

		CImg<unsigned char> output_image(output_buffer.data(), inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());
		CImgDisplay disp_output(output_image,"output");

 		while (!displayInput.is_closed() && !disp_output.is_closed()
			&& !displayInput.is_keyESC() && !disp_output.is_keyESC()) {
		    displayInput.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}
	catch (CImgException& err) {
		cerr << "ERROR: " << err.what() << endl;
	}

	return 0;
}
