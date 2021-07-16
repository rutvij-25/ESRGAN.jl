using Flux
using Flux:@functor

function ConvBlock(in,out,k,s,p,use_act)
    return Chain(
        Conv((k,k),in=>out,stride = s,pad = p,bias=true),
        use_act ? x -> leakyrelu.(x,0.2) : x -> x
    )
end

function UpsampleBlock(in,scale = 2)
    return Chain(
        Upsample(:nearest,scale = (scale,scale)),
        Conv((3,3),in=>in,stride = 1,pad = 1,bias=true),
        x -> leakyrelu.(x,0.2)
    )
end

mutable struct DenseResidualBlock
    residual_beta
    blocks
end

@functor DenseResidualBlock

function DenseResidualBlock(in,c = 32,residual_beta = 0.2)
    blocks = []
    for i in 0:4
        push!(blocks,ConvBlock((in + c*i),(i<=3 ? c : in),3,1,1,(i<=3 ? true : false)))
    end
    return DenseResidualBlock(residual_beta,blocks)
end

function (m::DenseResidualBlock)(x) 
    new_inputs = x
    local out,new_inputs
    for block in m.blocks
        out = block(new_inputs)
        new_inputs = cat(new_inputs,out,dims=3)
    end
    return m.residual_beta * out + x
end

mutable struct RRDB
    residual_beta
    rrdb
end

@functor RRDB

function RRDB(in,residual_beta = 0.2)
    rrdb = Chain([DenseResidualBlock(in) for _ in 1:3]...)
    RRDB(residual_beta,rrdb)
end

(m::RRDB)(x) = m.rrdb(x)*m.residual_beta + x

mutable struct Generator
    initial
    residuals
    conv
    upsamples
    final
end

function Generator(in=3,nc=64,nb=23)
    initial = Conv((3,3),in=>nc,stride = 1,pad = 1,bias=true)
    residuals = Chain([RRDB(nc) for _ in 1:nb]...)
    conv = Conv((3,3),nc=>nc,stride = 1,pad = 1)
    upsamples = Chain(UpsampleBlock(nc),UpsampleBlock(nc))
    final = Chain(
        Conv((3,3),nc=>nc,stride = 1,pad = 1,bias = true),
        x -> leakyrelu.(x,0.2),
        Conv((3,3),nc=>ic,stride = 1,pad = 1,bias=true)
    )
    Generator(initial,residuals,conv,upsamples,final)
end

function (m::Generator)(x)
    initial = m.initial(x)
    x = m.conv(m.residuals(initial)) + initial
    x = m.upsamples(x)
    x = m.final(x)
    return x
end

mutable struct Discriminator
    blocks
    classifier
end

function Discriminator(in = 3,features = [64, 64, 128, 128, 256, 256, 512, 512])
    blocks = []
    for (idx,feature) in features
        push!(blocks,ConvBlock(in,feature,3,(idx%2),1,true))
        in = feature
    end
    blocks = Chain(blocks...)
    classifier = Chain(
        AdaptiveMeanPool((6,6)),
        flatten,
        Dense(512 * 6 * 6, 1024),
        x -> x.leakyrelu(x,0.2),
        Linear(1024,1)
    )
    Discriminator(blocks,classifier)
end

(m::Discriminator)(x) = m.classifier(m.blocks(x))