require 'lfs'
require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

--------------------------------------------------------------------------------
-- Initialization
--------------------------------------------------------------------------------
m = torch.load('umich-stacked-hourglass.t7')   -- Load pre-trained model

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------
for file in lfs.dir(arg[1]) do
    local fullFilePath = arg[1].."/"..file
    local bboxFile = arg[2].."/"..string.sub(file,1,-4).."csv"
    if lfs.attributes(fullFilePath, "mode") == "file" and file:sub(-4) == ".jpg" and lfs.attributes(bboxFile, "mode") == "file" then

        print("Process "..file)

        -- open csv file
        local bboxFh = assert(io.open(bboxFile, "r"))
        local bboxCont = assert(bboxFh:read())
        bboxFh.close()
        local bboxParameters = {}

        -- string split
        for numStr in string.gmatch(bboxCont, "([^,]+)") do
            bboxParameters[#bboxParameters + 1] = tonumber(numStr)
        end

        -- load image file and related parameter
        local im = image.load(fullFilePath)
        local center = {bboxParameters[2], bboxParameters[3]}
        local scale = bboxParameters[1]
        local inp = crop(im, center, scale, 0, 256)

        -- Get network output
        local out = m:forward(inp:view(1, 3, 256, 256):cuda())
        cutorch.synchronize()
        local hm = out[#out][1]:float()
        hm[hm:lt(0)] = 0

        -- Get predictions (hm and img refer to the coordinate space)
        local preds_hm, preds_img = getPreds(hm, center, scale)

        preds_hm:mul(4) -- Change to input scale --> From 64x64 to 256x256
        -- print(preds_img[1][6][2]) -- Y
        -- print(preds_img[1][6][1]) -- X

        if false then
            local dispImg = drawOutput(inp, hm, preds_hm[1])
            w = image.display{image=dispImg, win=w}
            sys.sleep(3)
        end
       
        -- write to file
        local output_pose_file = arg[3].."/"..string.sub(file,1,-4).."csv"
        local outputFh = assert(io.open(output_pose_file, "w"))

        for i = 1, 16 do
            outputFh:write(preds_img[1][i][1]..","..preds_img[1][i][2].."\n")
        end

        outputFh.close()

        collectgarbage()
    end
end
