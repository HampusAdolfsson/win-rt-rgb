#pragma once

#include "WaveHandler.h"
#include "RingBuffer.h"
#include <functional>

class WaveToIntensityConverter : public WaveHandler {
public:
	WaveToIntensityConverter(std::function<void(const float&)> callback);
	~WaveToIntensityConverter() override;
	void receiveBuffer(float* samples, unsigned int nSamples) override;

private:
	float sum;
	float maxSum;
	RingBuffer<float> meanPrevVals;

	std::function<void(const float&)> callback;
};