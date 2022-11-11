#TESTE DE RECONSTRUÇÃO DE IMAGENS COM TTWOPT-V2
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
γ = 1
δ = [0.0001]
postoTTmax = 16;
W = weight_tensor(sz, mr)
subW = W_c(W, sz, 0.8)
figs = ["peppers", "woman", "satelite", "textura"]
b = length(δ)

erros_peppers = zeros(b)
psnr_peppers = zeros(b)
lista_posto_peppers = zeros(b, 8)
tempo_peppers = []
erros_woman = zeros(b)
psnr_woman = zeros(b)
tempo_woman = []
lista_posto_woman = zeros(b, 8)
erros_satelite = zeros(b)
psnr_satelite = zeros(b)
tempo_satelite = []
lista_posto_satelite = zeros(b, 8)
erros_textura = zeros(b)
psnr_textura = zeros(b)
tempo_textura = []
lista_posto_textura = zeros(b, 8)

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
			println()
			append!(tempo_peppers, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, true));
			erros_peppers[i] = norm(peppers - Z)/norm(peppers)
			psnr_peppers[i] = psnr(peppers, Z)
			lista_posto_peppers[i,:] = rank
			Z = inverse_mapping(Z)
			save("recover_ttv2_$imagem$i.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
		end

	elseif imagem == "woman"
		println("woman...")
		Y = W.*woman
		subY = subW.*woman
		global W
		global Y
		global postoTT
		
		for i=1:b
			append!(tempo_woman, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, true));
			erros_woman[i] = norm(woman - Z)/norm(woman)
			psnr_woman[i] = psnr(woman, Z)
			lista_posto_woman[i,:] = rank
			Z = inverse_mapping(Z)
			save("recover_ttv2_$imagem$i.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
		end

	elseif imagem == "satelite"
		println("satelite...")
		Y = W.*satelite
		subY = subW.*satelite
		global W
		global Y
		global postoTT
		
		for i=1:b
			append!(tempo_satelite, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, true));
			erros_satelite[i] = norm(satelite - Z)/norm(satelite)
			psnr_satelite[i] = psnr(satelite, Z)
			lista_posto_satelite[i,:] = rank
			Z = inverse_mapping(Z)
			save("recover_ttv2_$imagem$i.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
		end

	else
		println("textura...")
		Y = W.*textura
		subY = subW.*textura
		global W
		global Y
		global postoTT
		
		for i=1:b
			append!(tempo_textura, @elapsed Z, rank = TT_WOPT_v4(Y, subY, W, subW, sz, postoTTmax, δ[i], γ, 2, false, true));
			erros_textura[i] = norm(textura - Z)/norm(textura)
			psnr_textura[i] = psnr(textura, Z)
			lista_posto_textura[i,:] = rank
			Z = inverse_mapping(Z)
			save("recover_ttv2_$imagem$i.pdf",colorview(RGB, permuteddimsview(Z, (3, 1, 2))))
		end
	end
end
 println("Criando dataframes...")
df1 = DataFrame(delta = δ, peppers = erros_peppers, woman = erros_woman, satelite = erros_satelite, textura = erros_textura)
df2 = DataFrame(delta = δ, peppers = psnr_peppers, woman = psnr_woman, satelite = psnr_satelite, textura = psnr_textura)
df3 = DataFrame(delta = δ, peppers = tempo_peppers, woman = tempo_woman, satelite = tempo_satelite, textura = tempo_textura)

CSV.write("erros_teste8_ttv2.csv",df1)
CSV.write("psnr_teste8_ttv2.csv",df2)
CSV.write("tempos_teste8_ttv2.csv",df3)
println("postos-TT:")
println("peppers...")
println(lista_posto_peppers)
println("woman...")
println(lista_posto_woman)
println("satelite...")
println(lista_posto_satelite)
println("textura...")
println(lista_posto_textura)