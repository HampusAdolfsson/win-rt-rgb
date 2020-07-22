#pragma once
#include <string>
#define LOGINFO(str, ...) Logger::Instance().log(INFO, str, __VA_ARGS__);
#define LOGWARNING(str, ...) Logger::Instance().log(WARNING, str, __VA_ARGS__);
#define LOGSEVERE(str, ...) Logger::Instance().log(SEVERE, str, __VA_ARGS__);

typedef enum
{
	NONE, INFO, WARNING, SEVERE
} LogLevel;

class Logger
{
	LogLevel consoleLvl;
	LogLevel fileLvl;
	std::string fileName;
	Logger(LogLevel consoleLvl, LogLevel fileLvl);
public:
	Logger(const Logger&) = delete;
	void operator=(const Logger&) = delete;
	/**
	*	Global instance
	*/
	static Logger& Instance();
	/**
	*	Sets the lowest level of log messages to be output to the console.
	*	Default is INFO.
	*/
	void setConsoleLogLevel(const LogLevel& lvl);
	/**
	*	Sets the lowest level of log messages to be written to the log file.
	*	Default is INFO.
	*/
	void setFileLogLevel(const LogLevel& lvl);
	/**
	*	Sets the name of the log file to write to. Can be empty in which case nothing is written to file.
	*	By default, the log file is not used, and has to be specified using this function.
	*	@param fname Relative or absolute path to the log file to be used.
	*/
	void setLogFile(const std::string& fname);
	/**
	*	Write a message to the log.
	*	@param lvl the severity of the message, if this is NONE the message will not be logged
	*	@param format printf-style format specifier for the message
	*	@param ... arguments for the format string
	*/
	void log(const LogLevel& lvl, const char *format, ...);
};
