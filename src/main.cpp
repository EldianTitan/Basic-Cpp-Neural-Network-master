#include "Common.h"
#include "NeuralNet.h"
#include "CPUNeuralNet.h"

#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <cstring>
#include <string>
#include <stdio.h>

byte* openFile(const char* filepath, long int* filesize)
{
	FILE* file = fopen(filepath, "rb");

	if(!file)
	{
		printf("Error: Unable to open file '%s'.", filepath);
		exit(1);
		return nullptr;
	}

	fseek(file, 0, SEEK_END);
	long int size = ftell(file);
	fseek(file, 0, SEEK_SET);

	byte* imageData = new byte[size];
	fread(imageData, 1, size, file);

	fclose(file);

	if(filesize)
	{
		*filesize = size;
	}

	return imageData;
}

inline int32_t flipInt(const int32_t value)
{
	int32_t b0 = (value >> 24) & 0xFF;
	int32_t b1 = (value >> 8) & 0xFF00;
	int32_t b2 = (value << 8) & 0xFF0000;
	int32_t b3 = (value << 24) & 0xFF000000;

	return b0 | b1 | b2 | b3;
}

struct DataSet
{
	byte* imageData;
	byte* labelData;

	int32_t numImages;
	int32_t rows;
	int32_t columns;

	int32_t numLabels;

	int32_t labelHeaderSize;
	int32_t imageHeaderSize;
};

DataSet loadDataSet(const std::string& imageFile, const std::string& labelFile)
{
	DataSet dataSet;

	byte* imageData = openFile(imageFile.c_str(), nullptr);
	byte* labelData = openFile(labelFile.c_str(), nullptr);

	{
		struct 
		{
			int32_t magicNumber;
			int32_t numItems;
		} *header = (decltype(header))labelData;

		if(flipInt(header->magicNumber) != 2049)
		{
			printf("Value %d found instead of magic number!\n", flipInt(header->magicNumber));
			exit(1);
		}

		dataSet.numLabels = flipInt(header->numItems);
		dataSet.labelHeaderSize = sizeof(*header);
	}

	struct
	{
		int32_t magicNumber;
		int32_t numImages;
		int32_t rows;
		int32_t columns;
	} *header = (decltype(header))imageData;

	if(flipInt(header->magicNumber) != 2051)
	{
		printf("Value %d found instead of magic number!\n", flipInt(header->magicNumber));
		exit(1);
	}
	
	dataSet.imageHeaderSize = sizeof(*header);

	dataSet.numImages = flipInt(header->numImages);
	dataSet.rows = flipInt(header->rows);
	dataSet.columns = flipInt(header->columns);

	dataSet.imageData = imageData;
	dataSet.labelData = labelData;

	return dataSet;
}

//int main()
int main()
{
	std::string root = "<Path_to_project>/res/";

	DataSet trainingSet = loadDataSet(root + "train-images.idx3-ubyte", root + "train-labels.idx1-ubyte");

	int layerSizes[4] = { trainingSet.columns * trainingSet.rows, 16, 16, 10 };
	NeuralNet* neuralNet = new CPUNeuralNet(layerSizes, sizeof(layerSizes) / sizeof(layerSizes[0]));

	{
		neuralNet->loadImageData(&trainingSet.imageData[trainingSet.imageHeaderSize], trainingSet.columns, trainingSet.rows, trainingSet.numImages);
		neuralNet->loadLabelData(&trainingSet.labelData[trainingSet.labelHeaderSize], trainingSet.numLabels);
		neuralNet->train(400, 30, 0.0005f);

		delete[] trainingSet.labelData;
		delete[] trainingSet.imageData;
	}

	std::cout << std::endl;

	DataSet testSet = loadDataSet(root + "t10k-images.idx3-ubyte", root + "t10k-labels.idx1-ubyte");
	float accuracy = 0;

	int numTests = testSet.numImages;
	for(int i = 0; i < numTests; ++i)
	{
		int size = testSet.rows * testSet.columns;
		int y = (int)testSet.labelData[testSet.labelHeaderSize + i];
		int a = neuralNet->test(&testSet.imageData[testSet.imageHeaderSize + i * size]);

		std::cout << "Testing image of " << y << ": " << a << std::endl;

		if (y == a)
		{
			accuracy++;
		}
	}
	accuracy /= numTests;

	std::cout << "Accuracy: " << (accuracy * 100) << "%\n";

	delete neuralNet;

	return 0;
}