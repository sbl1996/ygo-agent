#include "ygoenv/ygopro0/ygopro.h"
#include "ygoenv/core/py_envpool.h"

using YGOPro0EnvSpec = PyEnvSpec<ygopro0::YGOProEnvSpec>;
using YGOPro0EnvPool = PyEnvPool<ygopro0::YGOProEnvPool>;

PYBIND11_MODULE(ygopro0_ygoenv, m) {
  REGISTER(m, YGOPro0EnvSpec, YGOPro0EnvPool)

  m.def("init_module", &ygopro0::init_module);
}
