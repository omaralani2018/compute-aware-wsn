#include <stdio.h>
#include <stdint.h>
#include <string.h>
static uint32_t fmix(uint32_t h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
static uint32_t chk(uint32_t s, uint32_t a, uint32_t c, uint32_t h) { return fmix(fmix(fmix(fmix(s) ^ a) ^ c) ^ h); }
int main(void){ uint8_t d[7]={0x13,0x2c,0x01,0x00,0x78,0x89,0x46}; uint32_t s=d[0]>>4,a=d[0]&15,r=d[1]|((uint32_t)d[2]<<8),v; memcpy(&v,d+3,4);
 uint32_t h=fmix(((s<<28)|(a<<24)|r)^fmix(v)); printf("%08x %08x\n",h,chk(s,a,1,h)); return 0;}
