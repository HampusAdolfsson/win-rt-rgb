#pragma once
#include <Windows.h>
#include <thread>
#include <vector>

#define NUM_BUFFERS 5

/**
*	Captures audio from an input device (preferably a loopback device) and visualizes the audio using a fruitypi server.
*	Audio is visualized on-dimensionally, based simply on loudness, no fourier transforms involved.
*/
class AudioVisualizer
{
	HWAVEIN				waveInHandle;
	WAVEHDR				waveHeaders[NUM_BUFFERS];
	std::vector<char>	buffer;
	std::thread			handlerThread;
	bool				isRunning;

	DWORD				deviceId;
	WAVEFORMATEX		pwfx;

	void handleWaveMessages();
	bool openDevice();
public:
	/**
	*	Starts an audio visualizer for the given device, recording with the given format
	*	For performance reasons, parts of the class is written specifically for 16-bit samples,
	*	so other sample sizes may not work. Also, it could be a good idea to use mono format, since
	*	the output is one-dimensional anyway.
	*	@param devId ID of the device to record from, as given by waveInGetDevCaps
	*	@param format the format to record audio in
	*/
	AudioVisualizer(const DWORD &devId, const WAVEFORMATEX &format);
	~AudioVisualizer();

	/**
	*	Initializes the visualizer, readying it to receive audio. Should be called ONCE before
	*	calling any other methods.
	*	@return true on success
	*/
	bool initialize();

	/**
	*	Starts the visualizer, if not already running.
	*	@return true on success
	*/
	bool start();

	/**
	*	Stops the visualizer, if it's running. It can then be started again by calling start()
	*	@return true on success
	*/
	bool stop();
};