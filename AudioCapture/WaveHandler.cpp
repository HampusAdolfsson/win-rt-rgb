#include "WaveHandler.h"

WaveHandler::WaveHandler()
: sampleRate(0),
nChannels(0)
{
}

void WaveHandler::setFormat(const unsigned int& samplesPerSec, const unsigned int& nChannels)
{
	this->sampleRate = samplesPerSec;
	this->nChannels = nChannels;
}