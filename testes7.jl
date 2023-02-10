#TESTE DE RECONSTRUÇÃO DE IMAGENS COM TTWOPT-DU
using LinearAlgebra, TensorToolbox
using TestImages, Images
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

Random.seed!(0)

function psnr(X, Z)
	if size(X)!=size(Z)
		println("dimensões diferentes")
		return nothing
	end

	mse = (norm(X-Z)^2)/196608 
	erro_psnr = 10*log10(1/mse)
	return erro_psnr
end

peppers = channelview(load("/home/joao/Mestrado/imagens/peppers.tiff"))
woman = channelview(load("/home/joao/Mestrado/imagens/woman.tiff"))
satelite= channelview(load("/home/joao/Mestrado/imagens/satelite1.tiff"));
textura =  channelview(load("/home/joao/Mestrado/imagens/textura2.tif"));

peppers = permutedims(peppers, (2,3,1));
woman = permutedims(woman, (2,3,1));
satelite = permutedims(satelite, (2,3,1));
textura = permutedims(satelite, (2,3,1));

peppers = mapping(peppers)
woman = mapping(woman)
satelite = mapping(satelite);
textura = mapping(textura);

sz = [4,4,4,4,4,4,4,4,3]
tam = 9
mr = 0.5
ρ = [1.0, 0.1, 0.01]
postoTTmax = 10;
W = weight_tensor(sz, mr)
subW = W_c(W, sz, 0.8)
figs = ["peppers", "woman", "satelite", "textura"]
b = length(ρ)

erros_peppers = zeros(b)
psnr_peppers = zeros(b)
tempo_peppers = []
erros_woman = zeros(b)
psnr_woman = zeros(b)
tempo_woman = []
erros_satelite = zeros(b)
psnr_satelite = zeros(b)
tempo_satelite = []
erros_textura = zeros(b)
psnr_textura = zeros(b)
tempo_textura = []

open("teste7_postoTT_imagens_TTWOPTDU.txt", "w") do file
	println("Iniciando...")
	for imagem in figs
		if imagem == "peppers"
			println("peppers...")
			Y = W.*peppers
			subY = subW.*peppers
			global W
			global Y
			global postoTT
			
			for i=1:b
				rho = ρ[i]
				println(rho)
				append!(tempo_peppers, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, true));
				erros_peppers[i] = norm(peppers - Z)/norm(peppers)
				psnr_peppers[i] = psnr(peppers, Z)
				Z = inverse_mapping(Z)
				save("TTWOPTDU_$imagem$rho.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
				write(file, "Posto-TT da reconstrução de $imagem com rho = $rho: $rank")
			end

		elseif imagem == "woman"
			println("woman...")
			Y = W.*woman
			subY = subW.*woman
			global W
			global Y
			global postoTT
			
			for i=1:b
				rho = ρ[i]
				println(rho)
				append!(tempo_woman, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, true));
				erros_woman[i] = norm(woman - Z)/norm(woman)
				psnr_woman[i] = psnr(woman, Z)
				Z = inverse_mapping(Z)
				save("TTWOPTDU_$imagem$rho.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
				write(file, "Posto-TT da reconstrução de $imagem com rho = $rho: $rank")
			end

		elseif imagem == "satelite"
			println("satelite...")
			Y = W.*satelite
			subY = subW.*satelite
			global W
			global Y
			global postoTT
			
			for i=1:b
				rho = ρ[i]
				println(rho)
				append!(tempo_satelite, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, true));
				erros_satelite[i] = norm(satelite - Z)/norm(satelite)
				psnr_satelite[i] = psnr(satelite, Z)
				Z = inverse_mapping(Z)
				save("TTWOPTDU_$imagem$rho.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
				write(file, "Posto-TT da reconstrução de $imagem com rho = $rho: $rank")
			end

		else
			println("textura...")
			Y = W.*textura
			subY = subW.*textura
			global W
			global Y
			global postoTT
			
			for i=1:b
				rho = ρ[i]
				println(rho)
				append!(tempo_textura, @elapsed Z, rank = TT_WOPT_DU(Y, subY, W, subW, sz, postoTTmax, rho, false, true));
				erros_textura[i] = norm(textura - Z)/norm(textura)
				psnr_textura[i] = psnr(textura, Z)
				Z = inverse_mapping(Z)
				save("TTWOPTDU_$imagem$rho.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
				write(file, "Posto-TT da reconstrução de $imagem com rho = $rho: $rank")
			end
		end
	end
end
println("Criando dataframes...")
df1 = DataFrame(rho = ρ, peppers = erros_peppers, woman = erros_woman, satelite = erros_satelite, textura = erros_textura)
df2 = DataFrame(rho = ρ, peppers = psnr_peppers, woman = psnr_woman, satelite = psnr_satelite, textura = psnr_textura)
df3 = DataFrame(rho = ρ, peppers = tempo_peppers, woman = tempo_woman, satelite = tempo_satelite, textura = tempo_textura)

CSV.write("TTWOPTDU_imagens_RSE.csv",df1)
CSV.write("TTWOPTDU_imagens_PSNR.csv",df2)
CSV.write("TTWOPTDU_imagens_tempos",df3)
