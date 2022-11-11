using LinearAlgebra, TensorToolbox
using TestImages, Images
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

Random.seed!(0)
δ = [0.01, 0.001, 0.0001]
γ = 1.05
postoTTmax = 16
num_points = 4096
f(x) = sin.(x./4)*cos.(x.^2)
x = LinRange(-4*pi, 4*pi, num_points)
missing_rates = [0.1,0.3,0.5, 0.7, 0.9]
b = length(missing_rates)

erros_dim1 = zeros(b,3)
tempo_dim1 = []
erros_dim2 = zeros(b,3)
tempo_dim2 = []
erros_dim3 = zeros(b,3)
tempo_dim3 = []

println("Começando...")
for teste=1:3
	if teste==1
		println("realizando teste 1...")
		global sz = [16, 16, 16]
		fX = reshape(collect(f.(x)), sz...)
		for i = 1:3
			for j = 1:length(missing_rates)
				println("j = $j")
				W = weight_tensor(sz, missing_rates[j])
				subW = W_c(W, sz, 0.8)
				Y = W.*fX
				subY = subW.*fX

				global W
				global Y
				global postoTT
				append!(tempo_dim1, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, false));
				erros_dim1[j,i] = norm(fX - Z)/norm(fX)
			end
		end

	elseif teste==2
		println("realizando teste 2...")
		sz = [8, 8, 8, 8]
		fX = reshape(collect(f.(x)), sz...)
		for i = 1:3
			for j = 1:length(missing_rates)
				println("j = $j")
				W = weight_tensor(sz, missing_rates[j])
				subW = W_c(W, sz, 0.8)
				Y = W.*fX
				subY = subW.*fX

				global W
				global Y
				global postoTT
				append!(tempo_dim2, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, false));
				erros_dim2[j,i] = norm(fX - Z)/norm(fX)
			end
		end
	else
		sz = [4, 4, 4, 4, 4, 4]
		println("realizando teste 3...")
		fX = reshape(collect(f.(x)), sz...)
		for i = 1:3
			for j = 1:length(missing_rates)
				println("j = $j")
				W = weight_tensor(sz, missing_rates[j])
				subW = W_c(W, sz, 0.8)
				Y = W.*fX
				subY = subW.*fX

				global W
				global Y
				global postoTT
				append!(tempo_dim3, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, false));
				erros_dim3[j,i] = norm(fX - Z)/norm(fX)
			end
		end
	end
end
println("ok!")
df1 = DataFrame(mr = missing_rates, ordem_3 = erros_dim1[:,1], ordem_4 = erros_dim2[:,1], ordem_6 = erros_dim3[:,1])
df2 = DataFrame(mr = missing_rates, ordem_3 = erros_dim1[:,2], ordem_4 = erros_dim2[:,2], ordem_6 = erros_dim3[:,2])
df3 = DataFrame(mr = missing_rates, ordem_3 = erros_dim1[:,3], ordem_4 = erros_dim2[:,3], ordem_6 = erros_dim3[:,3])
df4 = DataFrame(mr = missing_rates, ordem_3 = tempo_dim1[1:5], ordem_4 = tempo_dim2[1:5], ordem_6 = tempo_dim3[1:5])
df5 = DataFrame(mr = missing_rates, ordem_3 = tempo_dim1[6:10], ordem_4 = tempo_dim2[6:10], ordem_6 = tempo_dim3[6:10])
df6 = DataFrame(mr = missing_rates, ordem_3 = tempo_dim1[11:15], ordem_4 = tempo_dim2[11:15], ordem_6 = tempo_dim3[11:15])
CSV.write("erros_delta1_teste7.csv",df1)
CSV.write("erros_delta2_teste7.csv",df2)
CSV.write("erros_delta3_teste7.csv",df3)
CSV.write("tempo_delta1_teste7.csv",df4)
CSV.write("tempo_delta2_teste7.csv",df5)
CSV.write("tempo_delta3_teste7.csv",df6)
