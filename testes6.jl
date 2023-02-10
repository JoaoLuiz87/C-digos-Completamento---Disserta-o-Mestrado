#Teste com sinal gerado em 531441 pont0s
using LinearAlgebra, TensorToolbox
using TestImages, Images
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

Random.seed!(1234)
ρ = [1.0, 0.1, 0.01]
postoTTmax = 8
num_points = 531441
f(x) = sin.(x./4)*cos.(x.^2)
x = LinRange(-4*pi, 4*pi, num_points)
missing_rates = [0.1,0.3,0.5,0.7,0.9]
b = length(missing_rates)

erros_dim1 = zeros(b,3)
tempo_dim1 = []
posto_dim1 = zeros(2,2)
erros_dim2 = zeros(b,3)
tempo_dim2 = []
posto_dim2 = zeros(2,3)
erros_dim3 = zeros(b,3)
tempo_dim3 = []
posto_dim3 = zeros(2,5)
println("Começando...")

open("teste6_postoTT.txt", "w") do file
	for teste=1:3
		if teste==1
			println("realizando teste 1...")
			global sz = [81, 81, 81]
			ordem = 3
			fX = reshape(collect(f.(x)), sz...)
			for i = 1:3
				rho = ρ[i]
				for j = 1:length(missing_rates)
					println("j = $j")
					W = weight_tensor(sz, missing_rates[j])
					subW = W_c(W, sz, 0.8)
					Y = W.*fX
					subY = subW.*fX

					global W
					global Y
					global postoTT
					append!(tempo_dim1, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, false));
					erros_dim1[j,i] = norm(fX - Z)/norm(fX)

					mr = missing_rates[j]
					write(file, "posto TT de ordem=$ordem com mr=$mr e rho=$rho: $rank \n")
				end
			end

		elseif teste==2
			println("realizando teste 2...")
			sz = [27, 27, 27, 27]
			fX = reshape(collect(f.(x)), sz...)
			ordem = 4
			for i = 1:3
				rho = ρ[i]
				for j = 1:length(missing_rates)
					println("j = $j")
					W = weight_tensor(sz, missing_rates[j])
					subW = W_c(W, sz, 0.8)
					Y = W.*fX
					subY = subW.*fX

					global W
					global Y
					global postoTT
					append!(tempo_dim2, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, false));
					erros_dim2[j,i] = norm(fX - Z)/norm(fX)

					mr = missing_rates[j]
					write(file, "posto TT de ordem=$ordem com mr=$mr e rho=$rho: $rank \n")
				end
			end

		else
			sz = [9, 9, 9, 9, 9, 9]
			println("realizando teste 3...")
			fX = reshape(collect(f.(x)), sz...)
			ordem = 6
			for i = 1:3
				rho = ρ[i]
				for j = 1:length(missing_rates)
					println("j = $j")
					W = weight_tensor(sz, missing_rates[j])
					subW = W_c(W, sz, 0.8)
					Y = W.*fX
					subY = subW.*fX

					global W
					global Y
					global postoTT
					append!(tempo_dim3, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, false));
					erros_dim3[j,i] = norm(fX - Z)/norm(fX)

					mr = missing_rates[j]
					write(file, "posto TT de ordem=$ordem com mr=$mr e rho=$rho: $rank \n")
				end
			end
		end
	end
end
println("ok!")
df1 = DataFrame(mr = missing_rates, ordem3_rho1 = erros_dim1[:,1], ordem4_rho1 = erros_dim2[:,1], ordem6_rho1 = erros_dim3[:,1], ordem3_rho2 = erros_dim1[:,2], ordem4_rho2 = erros_dim2[:,2], ordem6_rho2 = erros_dim3[:,2], ordem3_rho3 = erros_dim1[:,3], ordem4_rho3 = erros_dim2[:,3], ordem6_rho3 = erros_dim3[:,3])
df2 = DataFrame(mr = missing_rates, ordem3_rho1 = tempo_dim1[1:5], ordem4_rho1 = tempo_dim2[1:5], ordem6_rho1 = tempo_dim3[1:5], ordem3_rho2 = tempo_dim1[6:10], ordem4_rho2 = tempo_dim2[6:10], ordem6_rho2 = tempo_dim3[6:10], ordem3_rho3 = tempo_dim1[11:15], ordem4_rho3 = tempo_dim2[11:15], ordem6_rho3 = tempo_dim3[11:15])
CSV.write("erros_sinal1_531441.csv",df1)
CSV.write("tempos_sinal1_531441.csv",df2)