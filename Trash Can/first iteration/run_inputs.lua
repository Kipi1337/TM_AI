local inputs = {}
local currentFrame = 0
local finished = false

-- === CSV LUKU ===
function loadInputs(path)
    local file = io.open(path, "r")
    if not file then
        error("input.csv ei löytynyt")
    end

    local first = true
    for line in file:lines() do
        if first then
            first = false -- ohitetaan header
        else
            local frame, acc, brake, left, right =
                line:match("(%d+),(%d+),(%d+),(%d+),(%d+)")
            table.insert(inputs, {
                acc = acc == "1",
                brake = brake == "1",
                left = left == "1",
                right = right == "1"
            })
        end
    end

    file:close()
end

-- === AJO ALKAA ===
function OnStart()
    loadInputs("input.csv")
    currentFrame = 1
    finished = false
    tm.resetRun()
end

-- === FRAME-LOOPPI ===
function OnRunStep()
    if finished then
        return
    end

    local input = inputs[currentFrame]
    if not input then
        finished = true
        tm.finishRun()
        return
    end

    tm.setInput("Accelerate", input.acc)
    tm.setInput("Brake", input.brake)
    tm.setInput("Left", input.left)
    tm.setInput("Right", input.right)

    currentFrame = currentFrame + 1
end
