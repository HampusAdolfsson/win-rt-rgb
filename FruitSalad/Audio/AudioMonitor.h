#pragma once
#include <thread>
#include <vector>
#include <functional>
#include "OverrideColorClient.h"
#include "WavetoIntensityStrategy.h"

#define NUM_BUFFERS 5

/**
*	Captures audio from an input device (preferably a loopback device) and monitors the audio intensity.
*/
class AudioMonitor
{
	HWAVEIN				waveInHandle;
	WAVEHDR				waveHeaders[NUM_BUFFERS];
	std::vector<char>	buffer;
	std::thread			handlerThread;
	bool				isRunning;

    std::function<void(uint8_t)> callback;

	WaveToIntensityStrategy waveStrategy;
	DWORD				deviceId;
	WAVEFORMATEX		pwfx;

	void handleWaveMessages();
	bool openDevice();
public:
	/**
	*	Starts an audio monitor for the given device, recording with the given format.
	*	For performance reasons, parts of the class is written specifically for 16-bit samples,
	*	so other sample sizes may not work. Also, it could be a good idea to use mono format, since
	*	the output is one-dimensional anyway.
	*	@param devId ID of the device to record from, as given by waveInGetDevCaps
	*	@param format the format to record audio in
    *	@param callback to call when a new intensity value is generated
	*/
	AudioMonitor(const DWORD &devId, const WAVEFORMATEX &format, std::function<void(uint8_t)> callback);
	~AudioMonitor();

	/**
	*	Initializes the monitor, readying it to receive audio. Should be called ONCE before
	*	calling any other methods.
	*	@return true on success
	*/
	bool initialize();

	/**
	*	Starts recording audio, if not already running.
	*	@return true on success
	*/
	bool start();

	/**
	*	Stops recording audio, if it's running. It can then be started again by calling start()
	*	@return true on success
	*/
	bool stop();

};