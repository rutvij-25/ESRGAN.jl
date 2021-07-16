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

