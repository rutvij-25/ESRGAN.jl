using Images

function load_image(path,img_size)
    img = load(path)
    img = imresize(img,(img_size,img_size))
    img = Float32.(channelview(img)) 
end


function get_batch(path,idxs)
    imgs = readdir(path,join=true)
    imgs = imgs[idxs]
    img_HR = []
    img_LR = []
    for img in imgs
        push!(img_HR,load_image(img,200))
        push!(img_LR,load_image(img,50))
    end
    img_HR = cat(img_HR...,dims=4)
    img_HR = permutedims(img_HR,(2,3,1,4))
    img_LR = cat(img_LR...,dims=4)
    img_LR = permutedims(img_LR,(2,3,1,4))
    return img_HR,img_LR
end


