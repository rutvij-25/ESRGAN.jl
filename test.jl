include("models.jl")

D = Discriminator()
G = Generator()

x = rand(Float32,16,16,3,32)

#println(G(x)|>size) #(64, 64, 3, 32)

y = rand(Float32,64,64,3,32)

println(D(x)|>size)