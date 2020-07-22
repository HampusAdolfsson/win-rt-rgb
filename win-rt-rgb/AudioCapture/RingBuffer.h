#pragma once
#include <vector>

template<typename T>
class RingBuffer
{
public:
	RingBuffer(size_t size, T initialData) : index(0), size(size), data(size, initialData) {}

	inline T putAndGet(const T& item)
	{
		T oldVal = data[index];
		data[index] = item;
		index++;
		if (index >= size) index = 0;
		return oldVal;
	}
private:
	size_t index;
	size_t size;
	std::vector<T> data;
};
