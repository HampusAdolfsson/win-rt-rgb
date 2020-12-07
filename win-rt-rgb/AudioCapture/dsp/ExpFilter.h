#pragma once
#include <array>

template<typename T>
class ExpFilter
{
public:
	ExpFilter(const T &rise, const T &fall) : rise(rise), fall(fall), output(0) {}

	inline void put(const T &value)
	{
		T smoothing = value < output ? fall : rise;
		output = smoothing * value + (1 - smoothing) * output;
	}

	inline const T &getOutput() { return output; }
private:
	const T rise;
	const T fall;
	float output;
};

