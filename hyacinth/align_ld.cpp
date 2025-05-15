
#include <hyacinth.h>

int32_t align_up(int32_t ld, int32_t align) {
  return (ld + align - 1) & ~(align - 1);
}

int32_t align_c_fp32(int32_t ld) {
  return align_up(ld, 8);
}

int32_t align_c_fp64(int32_t ld) {
  return align_up(ld, 4);
}

int32_t align_c_i8(int32_t ld) {
  return align_up(ld, 32);
}
