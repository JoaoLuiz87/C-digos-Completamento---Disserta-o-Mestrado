##ROTINA PARA A GERAÇÃO DOS GRÁFICOS DE RSE E PSNR COM RELAÇÃO ÀS IMAGENS COM KET AUGMENTATION

using LinearAlgebra, TensorToolbox
using TestImages, Images
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

function psnr(X, Z)
	if size(X)!=size(Z)
		println("dimensões diferentes")
		return nothing
	end

	mse = (norm(X-Z)^2)/196608 
	erro_psnr = 10*log10(1/mse)
	return erro_psnr
end

house = channelview(load("/home/joao/Mestrado/imagens/house.tiff"))
monkey = channelview(load("/home/joao/Mestrado/imagens/monkey.tiff"))
peppers = channelview(load("/home/joao/Mestrado/imagens/peppers.tiff"))
woman = channelview(load("/home/joao/Mestrado/imagens/woman.tiff"))

house = permutedims(house, (2,3,1));
monkey = permutedims(monkey, (2,3,1));
peppers = permutedims(peppers, (2,3,1));
woman = permutedims(woman, (2,3,1));

house = mapping(house)
monkey = mapping(monkey)
peppers = mapping(peppers)
woman = mapping(woman)

sz = [4,4,4,4,4,4,4,4,3]
tam = 9
tol = 10^-4
maxiter = 1000

#Parâmetros Heurísticas
r_parafac = 24
r_hosvd = [4, 4, 4, 4, 4, 4, 4, 4, 3]

#Parâmetros SiLRTC
α_silrtc = sz./sum(sz)
β_silrtc = α_silrtc

#Parâmetros SiLRTC-TT
α_silrtctt = pesos_modos(sz)
β_silrtctt = α_silrtctt

#Parâmetros HaLRTC
α_halrtc = sz./sum(sz)
ρ = 0.01

#Parâmetros TMac-TT
λ = 0.395
redo = 1

figs = ["house", "monkey", "peppers", "woman"]
missing_rate = [0.1, 0.3, 0.5, 0.7, 0.9]

for imagem in figs
	t_parafac = []
	t_hosvd = []
	t_silrtc = []
	t_silrtctt = []
	t_halrtc = []
	t_tmac = []
	t_ttwopt = []

	rse_parafac = zeros(5)
	rse_hosvd = zeros(5)
	rse_silrtc = zeros(5)
	rse_silrtctt = zeros(5)
	rse_halrtc = zeros(5)
	rse_tmac = zeros(5)
	rse_ttwopt = zeros(5)

	psnr_parafac = zeros(5)
	psnr_hosvd = zeros(5)
	psnr_silrtc = zeros(5)
	psnr_silrtctt = zeros(5)
	psnr_halrtc = zeros(5)
	psnr_tmac = zeros(5)
	psnr_ttwopt = zeros(5)

	for j = 1:length(missing_rate)
		Random.seed!(0)

		if imagem=="house"
			X = house
		elseif imagem=="monkey"
			X = monkey
		elseif imagem=="peppers"
			X = peppers
		else
			X = woman
		end

	    W1 = weight_tensor(sz, missing_rate[j])
	    Tw = W1.*X
	    normX = norm(X)

	    println("começou")
	    append!(t_parafac, @elapsed X_parafac, erro_parafac, iter_parafac = parafac_completion(Tw, W1, r_parafac, tol, maxiter, true))
	    println("PARAFAC OK")
	    append!(t_hosvd, @elapsed X_hosvd, erro_hosvd, iter_hosvd = hosvd_completion(Tw, W1, r_hosvd, tol, maxiter, true))
	    println("HOSVD OK")
	    append!(t_silrtc, @elapsed X_silrtc, erro_silrtc, iter_silrtc = SiLRTC(Tw, α_silrtc, β_silrtc, tol, maxiter, true))
	    println("SiLRTC OK")
	    append!(t_silrtctt, @elapsed X_silrtctt, erro_silrtctt, iter_silrtctt = SiLRTC_TT(Tw, α_silrtctt, β_silrtctt, tol, maxiter, true))
	    println("SiLRTC-TT OK")
	    append!(t_halrtc, @elapsed X_halrtc, erro_halrtc, iter_halrtc = HaLRTC(Tw, α_halrtc, ρ, tol, maxiter, true))
	    println("HaLRTC OK")
	    append!(t_tmac, @elapsed X_tmac, erro_tmac, iter_tmac = TMac_TT_v2(Tw, λ, tol, maxiter, redo, true))
	    println("Tmac_TT OK")

	    postoTT = [16,16,16,16,16,16,16,16]
	    W = weight_tensor(sz, missing_rate[j])
	    Y = W.*X
	    global W
		global Y
		global postoTT
		println(sz)
	    append!(t_ttwopt, @elapsed(X_ttwopt = TT_WOPT_v3(Y, W, sz, postoTT, true, true)))
	    println("TT_WOPT OK")
	    println(W==W1)
	    
	    rse_parafac[j] = norm(X-X_parafac)/normX
	    rse_hosvd[j] = norm(X-X_hosvd)/normX
	    rse_silrtc[j] = norm(X-X_silrtc)/normX
	    rse_silrtctt[j] = norm(X-X_silrtctt)/normX
	    rse_halrtc[j] = norm(X-X_halrtc)/normX
	    rse_tmac[j] = norm(X-X_tmac)/normX
	    rse_ttwopt[j] = norm(X-X_ttwopt)/normX
		
		psnr_parafac[j] = psnr(X, X_parafac)
		psnr_hosvd[j] = psnr(X, X_hosvd)
		psnr_silrtc[j] = psnr(X, X_silrtc)
		psnr_silrtctt[j] = psnr(X, X_silrtctt)
		psnr_halrtc[j] = psnr(X, X_halrtc)
		psnr_tmac[j] = psnr(X, X_tmac)
		psnr_ttwopt[j] = psnr(X, X_ttwopt)

		X_parafac = inverse_mapping(X_parafac)
		X_hosvd = inverse_mapping(X_hosvd)
		X_silrtc = inverse_mapping(X_silrtc)
		X_silrtctt = inverse_mapping(X_silrtctt)
		X_halrtc = inverse_mapping(X_halrtc)
		X_tmac = inverse_mapping(X_tmac)
		X_ttwopt = inverse_mapping(X_ttwopt)

	    if j== 4 || j==5
	    	save("recover2_ka_parafac_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_parafac, (3, 1, 2))))
	    	save("recover2_ka_hosvd_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_hosvd, (3, 1, 2))))
	    	save("recover2_ka_silrtc_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_silrtc, (3, 1, 2))))
	    	save("recover2_ka_silrtctt_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_silrtctt, (3, 1, 2))))
	    	save("recover2_ka_halrtc_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_halrtc, (3, 1, 2))))
	    	save("recover2_ka_tmac_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_tmac, (3, 1, 2))))
	    	save("recover2_ka_ttwopt_$imagem$j.pdf",colorview(RGB, permuteddimsview(X_ttwopt, (3, 1, 2))))
	    end
	end
	println("plots para $imagem")
	println(length)
	plot(missing_rate, rse_parafac, yticks = ([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]), xlims=(0,1), xticks=0.1:0.2:0.9, title = "RSE com KA - $imagem", marker=:utriangle, ls=:dot, legend=false)
	plot!(missing_rate, rse_hosvd,  marker=:ltriangle, ls=:dot)
	plot!(missing_rate, rse_silrtc,  marker=:star4, ls=:dot)
	plot!(missing_rate, rse_silrtctt,  marker=:circle, ls=:dot)
	plot!(missing_rate, rse_halrtc, marker=:diamond, ls=:dot)
	plot!(missing_rate, rse_tmac, marker=:star5, ls=:dot)
	plot!(missing_rate, rse_ttwopt,  marker=:xcross, ls=:dot)
	println("plot do rse ja foi")
	xlabel!("mr")
	ylabel!("RSE")
	savefig("exp3_rse_$imagem.pdf")

	println("plot do psnr")
	plot(missing_rate, psnr_parafac, xlims=(0,1), xticks=0.1:0.2:0.9, title = "PSNR com KA - $imagem", marker=:utriangle, ls=:dot, legend=false)
	plot!(missing_rate, psnr_hosvd, marker=:ltriangle, ls=:dot)
	plot!(missing_rate, psnr_silrtc, marker=:star4, ls=:dot)
	plot!(missing_rate, psnr_silrtctt, marker=:circle, ls=:dot)
	plot!(missing_rate, psnr_halrtc, marker=:diamond, ls=:dot)
	plot!(missing_rate, psnr_tmac, marker=:star5, ls=:dot)
	plot!(missing_rate, psnr_ttwopt, marker=:xcross, ls=:dot)
	xlabel!("mr")
	ylabel!("PSNR")
	savefig("exp3_psnr_$imagem.pdf")

	df1 = DataFrame(mr = missing_rate, PARAFAC = t_parafac, HOSVD = t_hosvd, SiLRTC = t_silrtc, SiLRTC_TT = t_silrtctt, HaLRTC = t_halrtc, TMac_TT = t_tmac, TT_WOPT = t_ttwopt)
	df2 = DataFrame(mr = missing_rate, PARAFAC = rse_parafac, HOSVD = rse_hosvd, SiLRTC = rse_silrtc, SiLRTC_TT = rse_silrtctt, HaLRTC = rse_halrtc, TMac_TT = rse_tmac, TT_WOPT = rse_ttwopt)
	df3 = DataFrame(mr = missing_rate, PARAFAC = psnr_parafac, HOSVD = psnr_hosvd, SiLRTC = psnr_silrtc, SiLRTC_TT = psnr_silrtctt, HaLRTC = psnr_halrtc, TMac_TT = psnr_tmac, TT_WOPT = psnr_ttwopt)
	println(df1)
	CSV.write("imagem_comKA_time_$imagem.csv",df1)
	CSV.write("imagem_comKA_rse_$imagem.csv",df2)
	CSV.write("imagem_comKA_psnr_$imagem.csv",df3)

end