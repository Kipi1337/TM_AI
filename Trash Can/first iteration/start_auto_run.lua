-- filename: tminterface_commands.lua
function start_auto_run()
    print("Auto run started!")

    -- yksinkertainen testiajo: 100 framea kaasu päällä
    for i = 1, 100 do
        tm.setInput("Accelerate", true)
        tm.setInput("Brake", false)
        tm.setInput("Left", false)
        tm.setInput("Right", false)
    end

    print("Auto run finished")
end
