#include <nvtx3/nvToolsExt.h>

extern "C" void w34_profile_push(const char* label) { nvtxRangePushA(label); }
extern "C" void w34_profile_pop() { nvtxRangePop(); }
