#include <nvtx3/nvToolsExt.h>

extern "C" void warp34_profile_push(const char* label) { nvtxRangePushA(label); }
extern "C" void warp34_profile_pop() { nvtxRangePop(); }
