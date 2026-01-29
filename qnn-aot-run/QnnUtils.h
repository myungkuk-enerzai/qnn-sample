#ifndef __QNN_UTILS_H__
#define __QNN_UTILS_H__

#include <cstdint>

void parse_arg(int argc, char** argv);

uint16_t fp32_to_fp16(float f);
float fp16_to_fp32(uint16_t h);

#endif