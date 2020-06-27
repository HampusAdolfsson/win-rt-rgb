#pragma once
#include <array>

class ExpFilter
{
public:
	ExpFilter(const float &smoothing);

	void put(const float &value);

	inline const float &getOutput() { return output; }
private:
	const float smoothing;
	float output;
};

