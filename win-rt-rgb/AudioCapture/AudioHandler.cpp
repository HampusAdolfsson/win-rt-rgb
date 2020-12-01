#include "AudioHandler.h"

void AudioHandler::setFormat(unsigned int nChannels, unsigned int sampleRate)
{
    this->nChannels = nChannels;
    this->sampleRate = sampleRate;
}