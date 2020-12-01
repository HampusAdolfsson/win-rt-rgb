#pragma once
#include <thread>
#include <vector>
#include <functional>
#include <regex>
#include <Windows.h>
#include <audioclient.h>
#include "AudioSink.h"

#define NUM_BUFFERS 5

/**
*	Captures audio from an output device and monitors the audio intensity.
*/
class AudioMonitor
{
	IAudioClient*			audioClient;
	IAudioCaptureClient*	captureClient;
	REFERENCE_TIME			hnsRequestedDuration;
	REFERENCE_TIME			hnsActualDuration;

	std::thread				handlerThread;
	bool					isRunning;
	AudioSink				sink;

	void handleWaveMessages();
	bool openDevice();
public:
	/**
	*	Starts an audio monitor for the default output device, capturing its output.
    *	@param handler to pass all captured audio onto
	*/
	AudioMonitor(AudioSink sink);
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