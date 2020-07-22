#pragma once
#include <array>

template<typename T>
class ExpFilter
{
public:
	ExpFilter(const T &smoothing) : smoothing(smoothing), output(0) {}

	inline void put(const T &value)
	{
		output = smoothing * value + (1 - smoothing) * output;
	}

	inline const T &getOutput() { return output; }
private:
	const float smoothing;
	float output;
};

