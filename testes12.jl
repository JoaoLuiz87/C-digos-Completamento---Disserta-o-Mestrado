#TESTE TTWOPT-DU COM OS DADOS SINTÉTICOS DE SISMOLOGIA - f: [0,1]^4 -> R,  f(x) = exp(-norm(x)) 

using LinearAlgebra, TensorToolbox
using TestImages, Images, LaTeXStrings
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

function build_tensor(f::Function, num_points::Int, dim::Int)
    x = LinRange(0,1, num_points)
    A = zeros((num_points for _ in 1:dim)...)
    iter = Iterators.product((x for _ in 1:dim)...)
    grid = [collect(i) for i in iter]
    A = f.(grid)
    
    return A
end

f(x) = exp(-norm(x))
num_points = 20
dim = 4
ρ = [1.0, 0.1, 0.01]
A = build_tensor(f, num_points, dim);
tempos_rho = zeros(3,3)
erros_rho = zeros(3,3)

mr = [0.9, 0.95, 0.99]
Random.seed!(0)
sz = size(A)
num_points = prod(sz)


postoTTmax = 5

open("teste_18_postoTT.txt", "w") do file
	for i in 1:length(mr)
		taxa = mr[i]
		W = weight_tensor(sz, taxa)
		subW = W_c(W, sz, 0.8)
		Y = W.*A;
		subY = subW.*A
		global W
		global Y
		global postoTT

		for j in 1:length(ρ)
			rho = ρ[j]
			println("rho = $rho")
			tempos_rho[i,j] = @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, false);
			erros_rho[i,j] = norm(A - Z)/norm(A)
			write(file, "posto-TT para mr = $taxa e rho = $rho : $rank \n")
		end
	end
end

df = DataFrame(mr = mr, rse_rho1 = erros_rho[:,1], rse_rho2 = erros_rho[:,2], rse_rho3 = erros_rho[:,3], tempos_rho1 = tempos_rho[:,1], tempos_rho2 = tempos_rho[:,2], tempos_rho3 = tempos_rho[:,3])
CSV.write("TTWOPT_DU_sismic.csv",df)
