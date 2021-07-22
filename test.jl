include("models.jl")

D = Discriminator()
G = Generator()

x = rand(Float32,64,64,3,5)

println(G(x)|>size) #(256, 256, 3, 5)

y = G(x)

println(D(y)|>size) #(1, 5)