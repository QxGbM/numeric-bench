
#include <hyacinth.h>

int32_t align_up(int32_t ld, int32_t align) {
  return (ld + align - 1) & ~(align - 1);
}
