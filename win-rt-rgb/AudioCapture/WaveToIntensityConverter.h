#pragma once

#include "WaveHandler.h"
#include "RingBuffer.h"
#include <functional>

class WaveToIntensityConverter : public WaveHandler {
public:
	WaveToIntensityConverter(unsigned int fps, std::function<void(const float&)> callback);
	~WaveToIntensityConverter() override;

protected:
	void handleWaveData(float* buffer, unsigned int nFrames) override;

private:
	float sum;
	float maxSum;
	RingBuffer<float> meanPrevVals;

	std::function<void(const float&)> callback;
};