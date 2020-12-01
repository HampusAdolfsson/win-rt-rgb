#pragma once
#include "AudioHandler.h"
#include "RingBuffer.h"
#include <functional>

/**
 *	Transforms audio into a 1d energy/intensity value.
 *	Energy values are relayed using a callback function.
 */
class EnergyAudioHandler : public AudioHandler
{
public:
	EnergyAudioHandler(std::function<void(float)> callback);
	~EnergyAudioHandler() override;

	void handleWaveData(float* buffer, unsigned int nFrames) override;

private:
	float sum;
	float maxSum;
	RingBuffer<float> meanPrevVals;

	std::function<void(float)> callback;
};