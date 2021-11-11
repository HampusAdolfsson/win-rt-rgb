#pragma once
#include <thread>
#include <vector>
#include <functional>
#include <regex>
#include <Windows.h>
#include <audioclient.h>
#include "AudioSink.h"

#define NUM_BUFFERS 5

namespace AudioCapture
{
	/**
	*	Captures audio from an output device and monitors the audio intensity.
	*/
	class AudioMonitor
	{
		IAudioClient*			audioClient;
		IAudioCaptureClient*	captureClient;
		WAVEFORMATEX*			pwfx;
		REFERENCE_TIME			hnsRequestedDuration;
		REFERENCE_TIME			hnsActualDuration;

		std::thread				handlerThread;
		bool					isRunning;
		std::vector<AudioSink>	sinks;

		void handleWaveMessages();
		bool openDevice();
	public:
		/**
		*	Starts an audio monitor for the default output device, capturing its output.
		*/
		AudioMonitor();
		~AudioMonitor();

		/**
		*	Initializes the monitor, readying it to receive audio. Should be called ONCE before
		*	calling start.
		*	@return true on success
		*/
		bool initialize();

		/**
		*	Adds an audio sink which will receive all captured audio.
		*	@param sink The audio sink to add. The audio monitor takes ownership of the sink.
		*/
		void addAudioSink(AudioSink sink);

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
}
