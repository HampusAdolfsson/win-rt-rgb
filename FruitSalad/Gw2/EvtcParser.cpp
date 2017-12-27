#include <cinttypes>
#include <cmath>
#include "EvtcParser.h"

#define ASSERT_NO_FAIL(is) do { \
							if(!is.good()) { \
							fprintf(stderr, "Unexpected eof %d\n", __LINE__); \
							return { -1, 0 }; \
							} \
							} while(0)

#define AGENT_INFO_SIZE 30
#define SKILL_INFO_SIZE 68

#define HEALTH_UPDATE_ID 8
#define REWARD_ID 17

#pragma pack(push, 1)
struct EvtcHeader
{
	char tag[4];
	uint64_t version;
	char padding;
	uint16_t instance_id;
	char padding2;
};
#pragma pack(pop)
#pragma pack(push, 1)
struct AgentData
{
	uint64_t agent;
	uint32_t profession;
	uint32_t elite;
	uint32_t toughness;
	uint32_t healing;
	uint32_t condi;
	char name[68];
};
#pragma pack(pop)
#pragma pack(push, 1)
typedef struct {
	uint64_t time; /* timegettime() at time of event */
	uint64_t src_agent; /* unique identifier */
	uint64_t dst_agent; /* unique identifier */
	int32_t value; /* event-specific */
	int32_t buff_dmg; /* estimated buff damage. zero on application event */
	uint16_t overstack_value; /* estimated overwritten stack duration for buff application */
	uint16_t skillid; /* skill id */
	uint16_t src_instid; /* agent map instance id */
	uint16_t dst_instid; /* agent map instance id */
	uint16_t src_master_instid; /* master source agent map instance id if source is a minion/pet */
	uint8_t iss_offset; /* internal tracking. garbage */
	uint8_t iss_offset_target; /* internal tracking. garbage */
	uint8_t iss_bd_offset; /* internal tracking. garbage */
	uint8_t iss_bd_offset_target; /* internal tracking. garbage */
	uint8_t iss_alt_offset; /* internal tracking. garbage */
	uint8_t iss_alt_offset_target; /* internal tracking. garbage */
	uint8_t skar; /* internal tracking. garbage */
	uint8_t skar_alt; /* internal tracking. garbage */
	uint8_t skar_use_alt; /* internal tracking. garbage */
	uint8_t iff; /* from iff enum */
	uint8_t buff; /* buff application, removal, or damage event */
	uint8_t result; /* from cbtresult enum */
	uint8_t is_activation; /* from cbtactivation enum */
	uint8_t is_buffremove; /* buff removed. src=relevant, dst=caused it (for strips/cleanses). from cbtr enum */
	uint8_t is_ninety; /* source agent health was over 90% */
	uint8_t is_fifty; /* target agent health was under 50% */
	uint8_t is_moving; /* source agent was moving */
	uint8_t is_statechange; /* from cbtstatechange enum */
	uint8_t is_flanking; /* target agent was not facing source */
	uint8_t is_shields; /* all or part damage was vs barrier/shield */
	uint8_t result_local; /* internal tracking. garbage */
	uint8_t ident_local; /* internal tracking. garbage */
} CbtEvent;
#pragma pack(pop)

BossFightInfo parseEvtc(std::ifstream& is)
{
	BossFightInfo bossInfo = { -1, 0 };

	struct EvtcHeader header;
	is.read(reinterpret_cast<char*>(&header), sizeof(header));
	ASSERT_NO_FAIL(is);

	uint32_t agentCount;
	is.read(reinterpret_cast<char*>(&agentCount), sizeof(agentCount));
	ASSERT_NO_FAIL(is);
	AgentData agent;
	is.read(reinterpret_cast<char*>(&agent), sizeof(agent));
	ASSERT_NO_FAIL(is);
	bossInfo.bossId = agent.agent;
	agentCount--;
	for (int i = 0; i < agentCount; i++)
	{
		AgentData agent;
		is.read(reinterpret_cast<char*>(&agent), sizeof(agent));
		ASSERT_NO_FAIL(is);
	}

	uint32_t skillCount;
	is.read(reinterpret_cast<char*>(&skillCount), sizeof(skillCount));
	ASSERT_NO_FAIL(is);
	is.ignore(skillCount * SKILL_INFO_SIZE);
	ASSERT_NO_FAIL(is);

	while (true)
	{
		CbtEvent cev;
		is.read(reinterpret_cast<char*>(&cev), sizeof(cev));
		if (is.eof()) return bossInfo;
		if (cev.is_statechange == HEALTH_UPDATE_ID)
		{
			bossInfo.finalHealthPercentage = ceil(cev.dst_agent / 100.0);
		}
		if (cev.is_statechange == REWARD_ID)
		{
			bossInfo.finalHealthPercentage = 0;
		}
	}
	return bossInfo;
}
