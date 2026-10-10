#include <stdint.h>
extern uint32_t _estack, _etext, _sdata, _edata, _sbss, _ebss;
int main(void);
void Reset_Handler(void) {
  uint32_t *s = &_etext, *d = &_sdata;
  while (d < &_edata) *d++ = *s++;
  for (d = &_sbss; d < &_ebss; ) *d++ = 0;
  main();
  __asm volatile (".hword 0xDF01");      /* simulator exit */
  for (;;) ;
}
__attribute__((section(".isr_vector"))) const void *vectors[2] = { &_estack, Reset_Handler };
/* minimal syscalls so newlib-nano links */
int _write(int f, char *p, int n){ for(int i=0;i<n;i++) *(volatile uint32_t*)0xE0000000 = p[i]; return n; }
void *_sbrk(int i){ extern char end; static char *h=&end; char *p=h; h+=i; return p; }
int _close(int f){return -1;} int _fstat(int f, void *s){return 0;} int _isatty(int f){return 1;}
int _lseek(int f,int p,int d){return 0;} int _read(int f,char*p,int n){return 0;}
void _exit(int s){ __asm volatile (".hword 0xDF01"); for(;;); }
int _kill(int p,int s){return -1;} int _getpid(void){return 1;}
