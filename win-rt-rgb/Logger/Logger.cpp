#include <cstdarg>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "Logger.h"

#define BUF_SIZ 1024

Logger::Logger(LogLevel consoleLvl, LogLevel fileLvl)
	: consoleLvl(consoleLvl),
	fileLvl(fileLvl),
	fileName("") { }

Logger& Logger::Instance()
{
	static Logger instance(INFO, INFO);
	return instance;
}

void Logger::setConsoleLogLevel(const LogLevel& lvl)
{
	consoleLvl = lvl;
}

void Logger::setFileLogLevel(const LogLevel& lvl)
{
	fileLvl = lvl;
}

void Logger::setLogFile(const std::string& fname)
{
	fileName = fname;
}

void Logger::log(const LogLevel& lvl, const char *format, ...)
{
	if (lvl == NONE) return;
	const char *timeFmt = "%d/%m/%y %H:%M";
	time_t t = std::time(nullptr);
	std::tm time;
	localtime_s(&time, &t);
	const char *lvlStr = lvl == SEVERE ? "SEVERE" :
						 lvl == WARNING ? "WARN" :
						 "INFO";
	char buffer[BUF_SIZ];
	va_list args;
	va_start(args, format);
	vsnprintf(buffer, BUF_SIZ, format, args);
	va_end(args);
	if (lvl >= consoleLvl)
	{
		std::cout << "[" << std::put_time(&time, timeFmt) << "] " << lvlStr << ": " << buffer << std::endl;
	}
	if (lvl >= fileLvl && !fileName.empty())
	{
		std::ofstream os;
		os.open(fileName, std::ios::out | std::ios::app);
		os << "[" << std::put_time(&time, timeFmt) << "] " << lvlStr << ": " << buffer << std::endl;
	}
}