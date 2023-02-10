#Teste com os 4096 ou 531441 pontos para gerar o posto-TT de rho e delta e graficos

using LinearAlgebra, TensorToolbox
using TestImages, Images, LaTeXStrings
using Optim, Random, LineSearches
using Plots, DataFrames, CSV, Measures
include("methods.jl");
include("core_changes.jl");

Random.seed!(1234)
ρ = [1.0, 0.1, 0.01]
δ = [0.1, 0.01, 0.001]
postoTTmax = 8
sz = [81,81,81]
num_points = prod(sz)

f(x) = sin.(x./4).*cos.(x.^2)
X = LinRange(-4*pi, 4*pi, num_points)

fX = reshape(collect(f.(X)), sz...)
W = weight_tensor(sz, 0.9)
subW = W_c(W, sz, 0.8)
Y = W.*fX;
subY = subW.*fX

graficos= repeat([plot(1)], 6)

println("comecou...")
open("teste17_postoTT_531441.txt", "w") do file
	for k=1:1		
		if k==1
			i=1
			for rho in ρ
				println("rho = $rho")
				global W
				global Y
				global postoTT
				Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, false);
				write(file, "posto-TT para rho = $rho : $rank")

				p = scatter(X, vec(Z), markercolor=:magenta4, markerstrokecolor=:magenta4,ms=1.8, grid=false, label=false, 
    			title = ("ρ = $rho"),
    			titlefontsize=10,
    			xlabel = ("x"),
    			ylabel = ("f(x)")
				)

				graficos[i] = p
				i = i+1
			end
		else
			i=1
			for delta in δ
				println("delta = $delta")
				global W
				global Y
				global postoTT
				Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, delta, 1.0, 0, false, false);
				write(file, "posto-TT para delta = $delta : $rank")

				p = scatter(X, vec(Z), markercolor=:magenta4, markerstrokecolor=:magenta4,ms=1.8, grid=false, label=false, 
    			title = ("δ = $delta"),
    			titlefontsize=10,
    			xlabel = ("x"),
    			ylabel = ("f(x)")
				)

				graficos[i+3] = p
				i = i+1
			end
		end
	end
end
println("ok!")

println("gerando gráficos...")
figura = plot(graficos[1:3]..., layout=(1,3), size=(800,200), margin=5mm)
savefig(figura, "teste17_sinal531441.pdf")
