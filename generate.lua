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

        local bboxFh = assert(io.open(bboxFile, "r"))
        local bboxCont = assert(bboxFh:read())
        local bboxParameters = {}

        for numStr in string.gmatch(cont, "([^,]+)") do
            bboxParameters[#bboxParameters + 1] = tonumber(numStr)
        end

        local im = image.load(fullFilePath)
        local center = {bboxParameters[1], bboxParameters[2]}
        local scale = bboxParameters[0]
        local inp = crop(im, center, scale, 0, 256)

        -- image.save(arg[1].."/"..file..".croped.JPG", inp)

        -- Get network output
        local out = m:forward(inp:view(1, 3, 256, 256):cuda())
        cutorch.synchronize()
        local hm = out[#out][1]:float()
        hm[hm:lt(0)] = 0

        -- Get predictions (hm and img refer to the coordinate space)
        local preds_hm, preds_img = getPreds(hm, center, scale)

        preds_hm:mul(4) -- Change to input scale --> From 64x64 to 256x256
        -- print(preds_img[1][6][1]) -- Y
        -- print(preds_img[1][6][2]) -- X

        if false then
            local dispImg = drawOutput(inp, hm, preds_hm[1])
            w = image.display{image=dispImg, win=w}
            sys.sleep(3)
        end
        
        for i in 1, 16 do
            preds_hm[1][i][1] = preds_hm[1][i][1] - 1
            preds_hm[1][i][2] = preds_hm[1][i][2] - 1
        end

        print(preds_hm[1][1][1])

        collectgarbage()
    end
end
