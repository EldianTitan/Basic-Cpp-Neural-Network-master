#pragma once

#include "NeuralNet.h"
#include "Matrix.h"

#include <vector>
#include <algorithm>
#include <memory>

enum FunctionType
{
	FUNC_RELU,
	FUNC_SIGMOID
};

class NetworkLayer
{
private:
	Matrix m_weights;
	Matrix m_biases;

	int m_layerSize = 0;
	int m_previousLayerSize = 0;

	FunctionType m_functionType;
public:
	NetworkLayer(int layerSize, int previousLayerSize, FunctionType functionType) :
		m_layerSize(layerSize), m_previousLayerSize(previousLayerSize),
		m_weights(layerSize, std::max(1, previousLayerSize)), m_biases(layerSize, 1),
		m_functionType(functionType)
	{ }

	NetworkLayer(const NetworkLayer& other) :
		m_layerSize(other.m_layerSize), m_previousLayerSize(other.m_previousLayerSize),
		m_weights(other.m_weights), m_biases(other.m_biases),
		m_functionType(other.m_functionType)
	{}

	~NetworkLayer() {}

	void initWeightsAndBiases(float weightRangeStart, float weightRangeEnd, float biasRangeStart, float biasRangeEnd);
	Matrix calculateAcitvations(const Matrix& previousActivations, Matrix* weightedSumStore) const;
	Matrix gradientDescent(const Matrix& activationDerivatives, const Matrix& weightedSums, const Matrix& previousLayerActivations, float learningRate);

	inline int getLayerSize() const { return m_layerSize; }

	inline Matrix& getWeights() { return m_weights; }
	inline const Matrix& getWeights() const { return m_weights; }

	inline Matrix& getBiases() { return m_biases; }
	inline const Matrix& getBiases() const { return m_biases; }
};

class CPUNeuralNet : public NeuralNet
{
private:
	float* m_imageData = nullptr;
	byte* m_labelData = nullptr;

	unsigned int m_imageWidth = 0;
	unsigned int m_imageHeight = 0;
	unsigned int m_numImages = 0;
	int m_numLabels = 0;

	std::vector<NetworkLayer> m_layers;
private:
	
public:
	CPUNeuralNet(int* layerSizes, int numLayers);
	~CPUNeuralNet();

	void loadImageData(byte* imageData, int width, int height, int numImage) override;
	void loadLabelData(byte* labelData, int numLabels) override;

	void train(unsigned int numIteration, unsigned int miniBatchSize, float trainingRate) override;

	int test(byte* imageData) const override;
};