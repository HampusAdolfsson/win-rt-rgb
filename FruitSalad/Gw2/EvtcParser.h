#pragma once
#include <Windows.h>
#include <fstream>

/**
*	Data resulting from a parsed dps log,
*	i.e. which boss was attempted and what health it reached at the end 
*	(0-100, 0 meaning it was killed)
*/
typedef struct
{
	int bossId;
	int finalHealthPercentage;
} BossFightInfo;

/**
*	Parses an open log file and returns some info about the fight
*	If the function encounters an error, the bossId field of the
*	returned struct will be -1.
*/
BossFightInfo parseEvtc(std::ifstream& is);
