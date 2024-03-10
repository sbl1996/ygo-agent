package("edopro-core")

    set_homepage("https://github.com/edo9300/ygopro-core")

    set_urls("https://github.com/edo9300/ygopro-core.git")

    -- set_sourcedir(path.join(os.scriptdir(), "edopro-core"))
    -- set_policy("package.install_always", true)

    add_deps("lua")

    on_install("linux", function (package)
        io.writefile("xmake.lua", [[
            add_rules("mode.debug", "mode.release")
            add_requires("lua")
            target("edopro-core")
                set_kind("static")
                set_languages("c++17")
                add_files("*.cpp")
                add_headerfiles("*.h")
                add_headerfiles("RNG/*.hpp")
                add_packages("lua")
        ]])

        local check_and_insert = function(file, line, insert)
            local lines = table.to_array(io.lines(file))
            if lines[line] ~= insert then
                table.insert(lines, line, insert)
                io.writefile(file, table.concat(lines, "\n"))
            end
        end

        check_and_insert("interpreter.h", 12, "extern \"C\" {")
        check_and_insert("interpreter.h", 14, "}")

        check_and_insert("interpreter.h", 16, "extern \"C\" {")
        check_and_insert("interpreter.h", 19, "}")

        local configs = {}
        if package:config("shared") then
            configs.kind = "shared"
        end
        import("package.tools.xmake").install(package)
        os.cp("*.h", package:installdir("include", "edopro-core"))
        os.cp("RNG", package:installdir("include", "edopro-core"))
    end)
package_end()