#include <cstdio>
#include <thread>
#include <locale>
#include <codecvt>
#include "Logger.h"
#include "ResponseCodes.h"
#include "EvtcParser.h"
#include "Gw2BossNotifier.h"

#define ARC_LOG_DIR "D:\\Documents\\Guild Wars 2\\addons\\arcdps\\arcdps.cbtlogs\\"
#define DIR_CHANGE_BUF_SIZE 256
#define FILE_OPEN_TRIES 4

#define EFFECT_REPETITIONS 5
#define EFFECT_LENGTH_NS 500000000
const std::vector<Color> successColors = { {0x09, 0xff, 0x00}, {0x73, 0xba, 0x30}, {0x09, 0xff, 0x00} }; // Colors on boss killed
// These colors are blended based on actual health percentage
const std::vector<Color> failureColors100 = { {0xff, 0x00, 0x00}, {0xbd, 0x11, 0x36}, {0xff, 0x00, 0x00} }; // Colors on boss failed at 100%
const std::vector<Color> failureColors0 = { {0xff, 0xc4, 0x00}, {0xbd, 0x89, 0x11}, {0xff, 0xc4, 0x00} }; // Colors on boss failed at 0%

Gw2BossNotifier::Gw2BossNotifier(RequestClient& cl)
	: reqClient(cl)
{
	hDir = CreateFile(ARC_LOG_DIR, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
	if (INVALID_HANDLE_VALUE == hDir)
	{
		LOGSEVERE("Error setting up directory listener: %d\n", GetLastError());
	}
	directoryListener = std::thread(&Gw2BossNotifier::listenToChanges, this);
}

Gw2BossNotifier::~Gw2BossNotifier()
{
	directoryListener.detach(); // Memory leak
	//CloseHandle(hDir);
}

void Gw2BossNotifier::listenToChanges()
{
	while (true)
	{
		char buf[DIR_CHANGE_BUF_SIZE];
		DWORD bytesRead = 0;
		bool res = ReadDirectoryChangesW(hDir, buf, DIR_CHANGE_BUF_SIZE - 2, // -2 to make space for null termination
											true,
											FILE_NOTIFY_CHANGE_FILE_NAME, 
											&bytesRead,
											nullptr, nullptr);

		if (!res || bytesRead == 0)
		{
			LOGSEVERE("Failed reading directory changes: %d\n", GetLastError());
		}
		else
		{
			PFILE_NOTIFY_INFORMATION changes = reinterpret_cast<PFILE_NOTIFY_INFORMATION>(buf);
			if (FILE_ACTION_ADDED == changes->Action)
			{
				buf[bytesRead] = buf[bytesRead + 1] = '\0';
				std::string fname = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(changes->FileName);
				LOGINFO("New file detected: %s", fname.c_str());
				fname = ARC_LOG_DIR + fname;
				onNewFileDetected(fname);
			}

		}
	}
}


std::vector<Color> getColorsFromBossHp(const int& bossHp);

void Gw2BossNotifier::onNewFileDetected(const std::string& fname)
{
	if (fname.compare(fname.length() - 5, 5, ".evtc") == 0)
	{
		std::ifstream createdFile;
		int tries = 0;
		do
		{
			Sleep(1000);
			createdFile.open(fname, std::ios::binary);
		} while (!createdFile.is_open() && tries < FILE_OPEN_TRIES);
		if (createdFile.is_open())
		{
			BossFightInfo bossInfo = parseEvtc(createdFile);
			if (bossInfo.bossId != -1)
			{
				const std::vector<Color> colors = getColorsFromBossHp(bossInfo.finalHealthPercentage);
				for (int i = 0; i < EFFECT_REPETITIONS; i++)
				{
					unsigned char res = reqClient.sendLightEffect(LightEffect(EFFECT_LENGTH_NS, Flashing, colors), false);
					if (res != SUCCESS)
					{
						LOGWARNING("Couldn't play lighteffect, server returned: %d", res);
					}
				}
			}
			createdFile.close();
		}
		else
		{
			LOGSEVERE("Unable to open file: %s\n", fname.c_str());
		}
	}
}

std::vector<Color> getColorsFromBossHp(const int &bossHp)
{
	if (bossHp == 0)
	{
		return successColors;
	}
	else
	{
		float blendProgress = bossHp / 100.0;
		std::vector<Color> blended;
		for (size_t i = 0; i < failureColors0.size(); i++)
		{
			blended.push_back(blendColors(failureColors0[i], failureColors100[i], blendProgress));
		}
		return blended;
	}
}
