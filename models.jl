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
        push!(blocks,Conv())
    end
end
