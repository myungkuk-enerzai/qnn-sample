#include "QnnUtils.h"
#include <cstring>
#include <cstdlib>

extern uint32_t input_shape;
extern uint32_t output_shape;

void parse_arg(int argc, char **argv)
{
	if (argc < 2)
		return;

	for (int i = 0; i < argc; i++)
	{
		if (!strcmp(argv[i], "--shape"))
		{
			if (i + 2 < argc)
			{
				input_shape = static_cast<uint32_t>(atoi(argv[i + 1]));
				output_shape = static_cast<uint32_t>(atoi(argv[i + 2]));
			}
		}
	}
}

uint16_t fp32_to_fp16(float f)
{
	uint32_t x = *(uint32_t *)&f;
	uint16_t h = ((x >> 16) & 0x8000) |
				 ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
				 ((x >> 13) & 0x03ff);
	return h;
}

float fp16_to_fp32(uint16_t h)
{
	uint32_t sign = (h & 0x8000) << 16;
	uint32_t exp = (h & 0x7c00) >> 10;
	uint32_t mant = (h & 0x03ff);

	uint32_t f;
	if (exp == 0)
		f = sign;
	else
		f = sign | ((exp + 112) << 23) | (mant << 13);

	float out;
	memcpy(&out, &f, sizeof(out));
	return out;
}
