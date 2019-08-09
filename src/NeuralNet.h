#pragma once

#include "Common.h"

class NeuralNet
{
public:
	NeuralNet() {}
	virtual ~NeuralNet() {}

	virtual void loadImageData(byte* imageData, int width, int height, int numImage) = 0;
	virtual void loadLabelData(byte* labelData, int numLabels) = 0;

	virtual void train(unsigned int numIteration, unsigned int miniBatchSize = 100, float trainingRate = 0.005) = 0;
	virtual int test(byte* imageData) const = 0;
};