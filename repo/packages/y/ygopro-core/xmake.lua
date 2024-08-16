package("ygopro-core")

    set_homepage("https://github.com/Fluorohydride/ygopro-core")

    add_urls("https://github.com/Fluorohydride/ygopro-core.git")
    add_versions("0.0.1", "6ed45241ab9360fd832dbc5fe913aa0017f577fc")
    add_versions("0.0.2", "f96929650ff8685b82fd48670126eae406366734")

    add_deps("lua")

    on_install("linux", function (package)
        io.writefile("xmake.lua", [[
            add_rules("mode.debug", "mode.release")
            add_requires("lua")
            target("ygopro-core")
                set_kind("static")
                add_files("*.cpp")
                add_headerfiles("*.h")
                add_packages("lua")
        ]])

        local check_and_insert = function(file, line, insert)
            local lines = table.to_array(io.lines(file))
            if lines[line] ~= insert then
                table.insert(lines, line, insert)
                io.writefile(file, table.concat(lines, "\n"))
            end
        end

        check_and_insert("field.h", 14, "#include <cstring>")
        check_and_insert("interpreter.h", 11, "extern \"C\" {")
        check_and_insert("interpreter.h", 15, "}")
        local configs = {}
        if package:config("shared") then
            configs.kind = "shared"
        end
        import("package.tools.xmake").install(package)
        os.cp("*.h", package:installdir("include", "ygopro-core"))
    end)
package_end()