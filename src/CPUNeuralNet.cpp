#include "CPUNeuralNet.h"

#include <math.h>
#include <cstring>
#include <cstdio>
#include <random>
#include <ctime>
#include <iostream>

float sigmoid(float value)
{
	return (float)(1.0 / (1.0 + exp(-value)));
}

float relu(float value)
{
	return std::max(0.0f, value);
}

float sigmoid_der(float value)
{
	float e = expf(-value);
	return e / ((1 + e) * (1 + e));
}

float relu_der(float value)
{
	if(value >= 0)
	{
		return 1.0f;
	}
	else
	{
		return 0.001f * value;
	}
}

/**
 * Each column is an example's activations
*/
Matrix NetworkLayer::calculateAcitvations(const Matrix& previousActivations, Matrix* weightedSumStore) const
{
	Matrix weightedSum = m_weights.dot(previousActivations) + m_biases;
	
	if(weightedSumStore)
	{
		*weightedSumStore = weightedSum;
	}

	if(m_functionType == FUNC_SIGMOID)
	{
		return weightedSum.applyCopy(sigmoid);
	}
	else
	{
		return weightedSum.applyCopy(relu);
	}
}

void NetworkLayer::initWeightsAndBiases(float weightRangeStart, float weightRangeEnd, float biasRangeStart, float biasRangeEnd)
{
	srand((unsigned int)time(0));

	m_weights.apply([=](float) -> float {
		float randomValue = static_cast<float>(rand()) / static_cast<float>(RAND_MAX + 1);
		return randomValue * (weightRangeEnd - weightRangeStart) + weightRangeStart;
	});

#if 0
	m_biases.apply([=](float) -> float {
		float randomValue = static_cast<float>(rand()) / static_cast<float>(RAND_MAX + 1);
		return randomValue * (biasRangeEnd - biasRangeStart) + biasRangeStart;
	});
#else
	m_biases.initValue(0);
#endif
}

//activationDerviatives -> (m_layerSize, numExamples)
Matrix NetworkLayer::gradientDescent(const Matrix& activationDerivatives, const Matrix& weightedSums, const Matrix& previousLayerActivations, float learningRate)
{
	Matrix dZ(1, 1);

	if(m_functionType == FUNC_SIGMOID)
	{
		dZ = weightedSums.applyCopy(sigmoid_der) * activationDerivatives;
	}
	else
	{
		dZ = weightedSums.applyCopy(relu_der) * activationDerivatives;
	}

	int m = dZ.getColumns();
	Matrix dB = 1.0f / m * dZ.sumAcross(AXIS_HORIZONTAL);
	Matrix dW = 1.0f / m * dZ.dot(previousLayerActivations.transpose());

	Matrix dA = m_weights.transpose().dot(dZ);

	m_weights = m_weights - learningRate * dW;
	m_biases = m_biases - learningRate * dB;

	return dA;
}

CPUNeuralNet::CPUNeuralNet(int* layerSizes, int numLayers)// : m_layers(numLayers)
{
	for(int i = 0; i < numLayers; ++i) 
	{
		if(i == 0)
		{
			NetworkLayer layer(layerSizes[i], 0, FunctionType::FUNC_RELU);
			layer.initWeightsAndBiases(-0.001f, 0.001f, -1, 1);
			
			m_layers.push_back(layer);
		}
		else
		{
			NetworkLayer layer(layerSizes[i], layerSizes[i - 1], (i < numLayers - 1) ? FunctionType::FUNC_RELU : FunctionType::FUNC_SIGMOID);
			layer.initWeightsAndBiases(-0.001f, 0.001f, -1, 1);
			
			m_layers.push_back(layer);
		}
	}
}

CPUNeuralNet::~CPUNeuralNet()
{
	if(m_imageData)
	{
		delete[] m_imageData;
	}

	if(m_labelData)
	{
		delete[] m_labelData;
	}
}

void CPUNeuralNet::loadImageData(byte* imageData, int width, int height, int numImages)
{
	m_imageWidth = width;
	m_imageHeight = height;
	m_numImages = numImages;

	int size = width * height;
	m_imageData = new float[numImages * size];

	for(int i = 0; i < numImages; ++i)
	{
		for(int j = 0; j < size; ++j)
		{
			int index = i * size + j;
			m_imageData[index] = (float)(255 -  imageData[index]) / 255.0f;
		}
	}
}

void CPUNeuralNet::loadLabelData(byte* labelData, int numLabels)
{
	m_labelData = new byte[numLabels * sizeof(byte)];

	memcpy(m_labelData, labelData, numLabels * sizeof(byte));
	m_numLabels = numLabels;
}

void CPUNeuralNet::train(unsigned int numIterations, unsigned int miniBatchSize, float trainingRate)
{
	srand((unsigned int)time(0));

	for(unsigned int iteration = 0; iteration < numIterations; ++iteration)
	{
		for(unsigned int i = 0; i < m_numImages; i += miniBatchSize)
		{
			unsigned int remaining = std::min(m_numImages - i, miniBatchSize);
			Matrix inputLayerData(m_layers[0].getLayerSize(), remaining);

			//Set input layer data
			for(unsigned int j = 0; j < remaining; ++j)
			{
				unsigned int imageIndex = i + j;
				unsigned int size = m_imageWidth * m_imageHeight;

				for(unsigned int k = 0; k < size; ++k)
				{
					inputLayerData.setValue(k, j, m_imageData[imageIndex * size + k]);
				}
			}

			std::vector<Matrix> weightedSums;
			weightedSums.reserve(m_layers.size());
			weightedSums.push_back(Matrix(1, 1).initValue(0));

			std::vector<Matrix> layerActivations;
			layerActivations.reserve(m_layers.size());
			layerActivations.push_back(inputLayerData);

			Matrix previousActivations = inputLayerData;

			//Forward propagation
			for(int j = 1; j < m_layers.size(); ++j)
			{
				Matrix weightedSumStore(1, 1);
				previousActivations = m_layers[j].calculateAcitvations(previousActivations, &weightedSumStore);

				layerActivations.push_back(previousActivations);
				weightedSums.push_back(weightedSumStore);
			}

			//Compute cross-entropy loss
			Matrix groundTruthData(m_layers[m_layers.size() - 1].getLayerSize(), remaining);
			groundTruthData.apply([&](float value, int row, int column, int numRows, int numColumns) -> float {
				return (m_labelData[i + column] == row) ? 1.0f : 0.0f;
			});

			Matrix outputLosses = -1.0f / remaining * Matrix::sumAcross(groundTruthData * previousActivations.applyCopy(std::log10f) + (1 - groundTruthData) * (1 - previousActivations).applyCopy(std::log10f), AXIS_HORIZONTAL);
			float totalCost = Matrix::sumAcross(outputLosses, AXIS_VERTICAL).getValue(0, 0);

			if ((m_numImages - i) <= miniBatchSize)
			{
				std::cout << "Total Cost[" << iteration << "]: " << totalCost << std::endl;
			}

			//Backpropagation
			Matrix outputDerivatives = -(groundTruthData / previousActivations - (1 - groundTruthData) / (1 - previousActivations));
			
			for(size_t j = m_layers.size() - 1; j >= 1; --j)
			{
				outputDerivatives = m_layers[j].gradientDescent(outputDerivatives, weightedSums[j], layerActivations[j - 1], trainingRate);
			}
		}
	}
}

int CPUNeuralNet::test(byte* imageData) const
{
	Matrix inputLayerData(m_layers[0].getLayerSize(), 1);

	//Set input layer data
	unsigned int size = m_imageWidth * m_imageHeight;
	for (unsigned int i = 0; i < size; ++i)
	{
		inputLayerData.setValue(i, 0, (float)(255 -  imageData[i]) / 255.0f);
	}

	Matrix previousActivations = inputLayerData;

	//Forward propagation
	for (size_t i = 1; i < m_layers.size(); ++i)
	{
		previousActivations = m_layers[i].calculateAcitvations(previousActivations, nullptr);
	}

	//Get the output value of the network
	int maxIndex = 0;
	float maxValue = 0.0f;
	for (unsigned int i = 0; i < previousActivations.getRows(); ++i)
	{
		float value = previousActivations.getValue(i, 0);
		if (value > maxValue)
		{
			maxValue = value;
			maxIndex = i;
		}
	}

	return maxIndex;
}