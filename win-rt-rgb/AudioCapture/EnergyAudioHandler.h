#pragma once
#include "AudioHandler.h"
#include "dsp/ExpFilter.h"
#include "fftw3.h"
#include <functional>

/**
 *	Transforms audio into a 1d energy/intensity value.
 *	Energy values are relayed using a callback function.
 */
class EnergyAudioHandler : public AudioHandler
{
public:
	EnergyAudioHandler(size_t bufferSize, unsigned int nChannels, std::function<void(float)> callback);
	~EnergyAudioHandler() override;

	void handleWaveData(float* buffer) override;

	EnergyAudioHandler(EnergyAudioHandler const&) = delete;
	EnergyAudioHandler(EnergyAudioHandler &&) = delete;
	EnergyAudioHandler operator=(EnergyAudioHandler const&) = delete;

private:
	unsigned int nChannels;
	std::vector<float> singleChannelWave;
	std::vector<float> dftResults;
	fftwf_plan plan;

	float* melFilter;
	ExpFilter<double> melGain;
	ExpFilter<double> melSmoothing;
	ExpFilter<double> commonMode;
	ExpFilter<double> outputFilter;

	std::function<void(float)> callback;
};

class EnergyAudioHandlerFactory : public AudioHandlerFactory
{
public:
	EnergyAudioHandlerFactory(std::function<void(float)> callback);
	std::unique_ptr<AudioHandler> createAudioHandler(size_t bufferSize, unsigned int nChannels) override;

private:
	std::function<void(float)> callback;
};