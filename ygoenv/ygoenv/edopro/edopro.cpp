#include "ygoenv/edopro/edopro.h"
#include "ygoenv/core/py_envpool.h"

using EDOProEnvSpec = PyEnvSpec<edopro::EDOProEnvSpec>;
using EDOProEnvPool = PyEnvPool<edopro::EDOProEnvPool>;

PYBIND11_MODULE(edopro_ygoenv, m) {
  REGISTER(m, EDOProEnvSpec, EDOProEnvPool)

  m.def("init_module", &edopro::init_module);
}
