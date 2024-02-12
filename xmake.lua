add_rules("mode.debug", "mode.release")

add_repositories("my-repo repo")

add_requires(
    "ygopro-core", "pybind11 2.10.*", "fmt 10.2.*", "glog 0.6.0",
    "concurrentqueue 1.0.4", "sqlitecpp 3.2.1", "unordered_dense 4.4.*")


target("dummy_ygopro")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/dummy/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue")
    set_languages("c++17")
    set_policy("build.optimization.lto", true)
    add_includedirs("ygoenv")
    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/dummy"
        os.mv(target:targetfile(), install_target)
        print("move target to " .. install_target)
    end)


target("ygopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core")
    set_languages("c++17")
    add_cxxflags("-flto=auto -fno-fat-lto-objects -fvisibility=hidden -march=native")
    add_includedirs("ygoenv")

    -- for _, header in ipairs(os.files("ygoenv/ygoenv/core/*.h")) do
    --     set_pcxxheader(header)
    -- end

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro"
        os.mv(target:targetfile(), install_target)
        print("Move target to " .. install_target)
    end)
