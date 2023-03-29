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

int main(int argc, char **argv) {
	// This part handles the command line options
	int platformID = 0;
	int deviceID = 0;
	string imageName = "test_large.pgm"; // Change this to change the input image. Images available: test.pgm, test_large.pgm
	// Menu
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { imageName = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	// This detects exceptions
	try {
		CImg<unsigned char> inputImage(imageName.c_str());
		CImg<unsigned short> inputImage16(imageName.c_str());

		// This normalises 16 bit images down to 8 and allows them to be parsed
		if (inputImage.max() > 256) {
			inputImage = inputImage16.get_normalize(0, 255);
		}

		CImgDisplay displayInput(inputImage,"input");

		// A 3x3 convolution mask implementing an averaging filter
		vector<float> convolution = { 1.f / 9, 1.f / 9, 1.f / 9,
									  1.f / 9, 1.f / 9, 1.f / 9,
									  1.f / 9, 1.f / 9, 1.f / 9 };

		// This handles the host operations
		
		// Select computing devices
		cl::Context context = GetContext(platformID, deviceID);
		cout << "Runing on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << endl;

		// Creates a command queue
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Builds device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels/my_kernels.cl");
		cl::Program program(context, sources);

		// Builds Kernel Code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}

		// Memory allocation

		// This allows for dynamically sized bins
		string input;
		int bins;
		while (true) {

			cout << "Enter number of bins: " << endl;
			cin >> input;

			bins = stoi(input);

			if (bins > 0 && bins < 257) {
				break;
			}

			cout << "Must have between 1 and 256 bins" << endl;
		}

		typedef int mytype;
		vector<mytype> H(bins);
		size_t histsize = H.size() * sizeof(mytype);

		CImg<unsigned char> CB,CR;
		bool isColour = false;

		if (inputImage.spectrum() == 3) {
			inputImage = inputImage.get_RGBtoYCbCr();
			CB = inputImage.get_channel(1);
			CR = inputImage.get_channel(2);
			inputImage = inputImage.get_channel(0);
			isColour = true;
		}

		// Device buffers
		cl::Buffer devInputImage(context, CL_MEM_READ_ONLY, inputImage.size());
		cl::Buffer devOutputHistogram(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer devOutputCHistogram(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer devLUT(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer devOutputImage(context, CL_MEM_READ_WRITE, inputImage.size()); //should be the same as input image

		// Copy images to device memory
		queue.enqueueWriteBuffer(devInputImage, CL_TRUE, 0, inputImage.size(), &inputImage.data()[0]);
		queue.enqueueFillBuffer(devOutputHistogram, 0, 0, histsize);

		// Setup and execute the kernel
		//The first kernel call plots a histogram of the frequency of each pixel value (0-255) in the picture
		cl::Kernel kernelSimpleHist = cl::Kernel(program, "simpleHist");
		kernelSimpleHist.setArg(0, devInputImage);
		kernelSimpleHist.setArg(1, devOutputHistogram);

		cl::Event profEvent;

		queue.enqueueNDRangeKernel(kernelSimpleHist, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &profEvent);
		queue.enqueueReadBuffer(devOutputHistogram, CL_TRUE, 0, histsize, &H[0]);

		vector<mytype> CH(bins);

		queue.enqueueFillBuffer(devOutputCHistogram, 0, 0, histsize);

		// The second kernel call plots a cumulative histogram of the total pixels in the picture across pixel values 0-255, so by 255, all pixels have been counted
		cl::Buffer hsOutput(context, CL_MEM_READ_WRITE, histsize);
		queue.enqueueCopyBuffer(devOutputHistogram, hsOutput, 0, 0, histsize);
		
		cl::Kernel kernelCHist = cl::Kernel(program, "scan_hs");
		kernelCHist.setArg(0, hsOutput);
		kernelCHist.setArg(1, devOutputHistogram);

		cl::Event profEventTwo;

		queue.enqueueNDRangeKernel(kernelCHist, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &profEventTwo);
		queue.enqueueReadBuffer(devOutputHistogram, CL_TRUE, 0, histsize, &CH[0]);

		vector<mytype> LUT(bins);

		queue.enqueueFillBuffer(devLUT, 0, 0, histsize);

		// The third kernel call creates a new histogram that will serve as a look up table of the new pixel vales. It does this by normalising the cumulative histogram, essentially decreasing the value of the pixels to increase the contrast
		cl::Kernel kernelLUT = cl::Kernel(program, "LUT");
		kernelLUT.setArg(0, devOutputHistogram);
		kernelLUT.setArg(1, devLUT);

		cl::Event profEventThree;

		queue.enqueueNDRangeKernel(kernelLUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &profEventThree);
		queue.enqueueReadBuffer(devLUT, CL_TRUE, 0, histsize, &LUT[0]);

		// The last kernel assigns the new pixel values from the lookup table to the output image, so that the output is of higher contrast than the input
		cl::Kernel kernelReproject = cl::Kernel(program, "ReProject");
		kernelReproject.setArg(0, devInputImage);
		kernelReproject.setArg(1, devLUT);
		kernelReproject.setArg(2, devOutputImage);

		cl::Event profEventFour;

		// The values from each histogram are printed, along with the kernel execution times and memory transfer of each kernel.
		vector<unsigned char> outputBuffer(inputImage.size());
		queue.enqueueNDRangeKernel(kernelReproject, cl::NullRange, cl::NDRange(inputImage.size()), cl::NullRange, NULL, &profEventFour);
		queue.enqueueReadBuffer(devOutputImage, CL_TRUE, 0, outputBuffer.size(), &outputBuffer.data()[0]);

		cout << endl;
		cout << "Histogram = " << H << endl;
		cout << "Histogram kernel execution time [ns]: " << profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEvent, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << "Cumulative Histogram = " << CH << endl;
		cout << "Cumulative Histogram kernel execution time [ns]: " << profEventTwo.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventTwo.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEventTwo, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		cout << "LUT = " << LUT << endl;
		cout << "LUT kernel execution time [ns]: " << profEventThree.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventThree.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEventThree, ProfilingResolution::PROF_US) << endl;
		cout << endl;


		cout << "Vector kernel execution time [ns]: " << profEventFour.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEventFour.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(profEventFour, ProfilingResolution::PROF_US) << endl;

		CImg<unsigned char> outputImage(outputBuffer.data(), inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());
		CImg <unsigned char> recombinedImage (inputImage.width(), inputImage.height(), 1, 3);

		if (isColour) {
			cimg_forXY(recombinedImage, x, y) {
				recombinedImage(x, y, 0, 0) = outputImage(x, y);
				recombinedImage(x, y, 0, 1) = CB(x, y);
				recombinedImage(x, y, 0, 2) = CR(x, y);
			}

			outputImage = recombinedImage.YCbCrtoRGB();
		}

		CImgDisplay displayOutput(outputImage,"output");

 		while (!displayInput.is_closed() && !displayOutput.is_closed()
			&& !displayInput.is_keyESC() && !displayOutput.is_keyESC()) {
		    displayInput.wait(1);
		    displayOutput.wait(1);
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
