#TESTE TTWOPT-DU COM A TENSORIZAÇÃO DE UMA MATRIZ - F(X) = cos(x1 + x2)

using LinearAlgebra, TensorToolbox
using TestImages, Images, LaTeXStrings
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

function mapeia(X)
    v = 2*ones(Int, 12)
    new_dim = [v...]
    new_idx = [1 7 2 8 3 9 4 10 5 11 6 12]
    new_X = reshape(X, new_dim...)
    new_X = permutedims(new_X, new_idx)
    new_X = reshape(new_X, 4,4,4,4,4,4)
    return new_X
end

function unmapeia(X)
    v = 2*ones(Int, 12)
    new_dim = [v...]
    old_idx = [1 3 5 7 9 11 2 4 6 8 10 12]
    old_X = reshape(X, new_dim...)
    old_X = permutedims(old_X, old_idx)
    old_X = reshape(old_X, 64, 64)
    return old_X
end

Random.seed!(1234)
ρ = [1.0, 0.1, 0.01]
δ = [0.1, 0.01, 0.001]
postoTTmax = 8
x = LinRange(-pi, pi, 64)
y = LinRange(-pi, pi, 64)
f(x) = cos(3x[1] + 3x[2])
grid = collect(Iterators.product(x, y))
z = f.(grid)
hz = mapeia(z)

erros_rho = []
tempos_rho = []
erros_delta = []
tempos_delta = []
graficos_delta = repeat([plot(1)], 3)
graficos_rho = repeat([plot(1)], 3)
sz = size(hz)
W = weight_tensor(sz, 0.9)
subW = W_c(W, sz, 0.8)
Y = W.*hz;
subY = subW.*hz

println("comecou...")
open("teste14_postoTT.txt", "w") do file
	for k=1:2		
		if k==1
			i=1
			for rho in ρ
				println("rho = $rho")
				global W
				global Y
				global postoTT
				append!(tempos_rho, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, false));
				append!(erros_rho, norm(hz - Z)/norm(hz))
				write(file, "posto-TT para rho = $rho : $rank")

				newZ = unmapeia(Z)

				p = heatmap(newZ, c = cgrad([:blue, :white, :red]),
    			title=("ρ = $rho"),
    			titlefontsize=10,
    			clims=(minimum(newZ),maximum(newZ)),
    			xticks=(range(1,64,5), [L"-\pi", L"\pi/2", L"0", L"\pi/2", L"\pi"]),
    			xtickfontsize=7, 
    			yticks=(range(1,64,5), [L"-\pi", L"\pi/2", L"0", L"\pi/2", L"\pi"]), 
    			ytickfontsize=7)
    			println("deu ruim")
				graficos_rho[i] = p
				i = i+1
				println("aqui")
	
			end
		else
			i=1
			for delta in δ
				println("delta = $delta")
				global W
				global Y
				global postoTT
				append!(tempos_delta, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, delta, 1.0, 0, false, false));
				append!(erros_delta, norm(hz - Z)/norm(hz))
				write(file, "posto-TT para delta = $delta : $rank")

				newZ = unmapeia(Z)

				p = heatmap(newZ, c = cgrad([:blue, :white, :red]),
    			title=("δ = $delta"),
    			clims=(minimum(newZ),maximum(newZ)),
    			titlefontsize=10,
    			xticks=(range(1,64,5), [L"-\pi", L"\pi/2", L"0", L"\pi/2", L"\pi"]),
    			xtickfontsize=7, 
    			yticks=(range(1,64,5), [L"-\pi", L"\pi/2", L"0", L"\pi/2", L"\pi"]), 
    			ytickfontsize=7)
				graficos_delta[i] = p
				i = i+1
			end
		end
	end
end
println("ok!")
df = DataFrame(rho = ρ, rse_rho = erros_rho, tempos_rho = tempos_rho, delta = δ, rse_delta = erros_delta , tempos_delta = tempos_delta)
CSV.write("TTWOPT_DU_matriz_cos.csv",df)

println("gerando gráficos...")
figura1 = plot(graficos_rho..., layout=(1,3), colorbar=true, size=(800,200))
figura2 = plot(graficos_delta..., layout=(1,3), colorbar=true, size=(800,200))
savefig(figura1, "teste14_rho.pdf")
savefig(figura2, "teste14_delta.pdf")
