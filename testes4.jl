##ROTINA PARA A GERAÇÃO DOS GRÁFICOS DE RSE COM RELAÇÃO AOS DADOS SINTÉTICOS BASEADOS NO POSTO DE TUCKER OU MULTILINEAR

using LinearAlgebra, TensorToolbox
using TestImages, Images
using Optim, Random, LineSearches
using Plots, DataFrames, CSV
include("methods.jl");
include("core_changes.jl");

sz = [30,30,30,30]
tam = 4
tol = 10^-4
maxiter = 1000

#Parâmetros Heurísticas
r_parafac = 4

#Parâmetros SiLRTC
α_silrtc = sz./sum(sz)
β_silrtc = 0.01*α_silrtc

#Parâmetros SiLRTC-TT
α_silrtctt = pesos_modos(sz)
β_silrtctt = 0.01*α_silrtctt

#Parâmetros HaLRTC
α_halrtc = sz./sum(sz)
ρ = 0.001

#Parâmetros TMac-TT
λ = 0.415
redo = 1

missing_rate = [0.1, 0.3, 0.5, 0.7, 0.9]
postoTuc = [4,8,12,16]

for r in postoTuc
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

	for j = 1:length(missing_rate)
		Random.seed!(0)

		r_hosvd = (r+3)*ones(Int, tam)
		X = full(randttensor(sz, r_hosvd))
	    W1 = weight_tensor(sz, missing_rate[j])
	    Tw = W1.*X
	    normX = norm(X)

	    println("começou")
	    append!(t_parafac, @elapsed X_parafac, erro_parafac, iter_parafac = parafac_completion(Tw, W1, r_parafac, tol, maxiter, false))
	    println("PARAFAC OK")
	    append!(t_hosvd, @elapsed X_hosvd, erro_hosvd, iter_hosvd = hosvd_completion(Tw, W1, r_hosvd, tol, maxiter, false))
	    println("HOSVD OK")
	    append!(t_silrtc, @elapsed X_silrtc, erro_silrtc, iter_silrtc = SiLRTC(Tw, α_silrtc, β_silrtc, tol, maxiter, false))
	    println("SiLRTC OK")
	    append!(t_silrtctt, @elapsed X_silrtctt, erro_silrtctt, iter_silrtctt = SiLRTC_TT(Tw, α_silrtctt, β_silrtctt, tol, maxiter, false))
	    println("SiLRTC-TT OK")
	    append!(t_halrtc, @elapsed X_halrtc, erro_halrtc, iter_halrtc = HaLRTC(Tw, α_halrtc, ρ, tol, maxiter, false))
	    println("HaLRTC OK")
	    append!(t_tmac, @elapsed X_tmac, erro_tmac, iter_tmac = TMac_TT_v2(Tw, λ, tol, maxiter, redo, false))
	    println("Tmac_TT OK")

	    postoTT = [12,48,12]
	    W = weight_tensor(sz, missing_rate[j])
	    Y = W.*X
	    global W
		global Y
		global postoTT
		println(sz)
	    append!(t_ttwopt, @elapsed(X_ttwopt = TT_WOPT_v3(Y, W, sz, postoTT, true, false)))
	    println("TT_WOPT OK")
	    
	    rse_parafac[j] = norm(X-X_parafac)/normX
	    rse_hosvd[j] = norm(X-X_hosvd)/normX
	    rse_silrtc[j] = norm(X-X_silrtc)/normX
	    rse_silrtctt[j] = norm(X-X_silrtctt)/normX
	    rse_halrtc[j] = norm(X-X_halrtc)/normX
	    rse_tmac[j] = norm(X-X_tmac)/normX
	    rse_ttwopt[j] = norm(X-X_ttwopt)/normX
		
	end
	println("plots para rse: r = $r")
	println(length)
	plot(missing_rate, rse_parafac, yaxis=:log, yticks = ([1e-4,1e-3,1e-2,1e-1,1e0]), xlims=(0,1), xticks=0.1:0.2:0.9, title = "Tucker com r = $r", marker=:utriangle, ls=:dot, legend=false)
	plot!(missing_rate, rse_hosvd, yaxis=:log, marker=:ltriangle, ls=:dot)
	plot!(missing_rate, rse_silrtc,yaxis=:log, marker=:star4, ls=:dot)
	plot!(missing_rate, rse_silrtctt,yaxis=:log, marker=:circle, ls=:dot)
	plot!(missing_rate, rse_halrtc, yaxis=:log, marker=:diamond, ls=:dot)
	plot!(missing_rate, rse_tmac, yaxis=:log, marker=:star5, ls=:dot)
	plot!(missing_rate, rse_ttwopt, yaxis=:log, marker=:xcross, ls=:dot)
	println("plot do rse ja foi")
	xlabel!("mr")
	ylabel!("RSE")
	savefig("exp4_rse_r$r.pdf")

	df1 = DataFrame(mr = missing_rate, PARAFAC = t_parafac, HOSVD = t_hosvd, SiLRTC = t_silrtc, SiLRTC_TT = t_silrtctt, HaLRTC = t_halrtc, TMac_TT = t_tmac, TT_WOPT = t_ttwopt)
	df2 = DataFrame(mr = missing_rate, PARAFAC = rse_parafac, HOSVD = rse_hosvd, SiLRTC = rse_silrtc, SiLRTC_TT = rse_silrtctt, HaLRTC = rse_halrtc, TMac_TT = rse_tmac, TT_WOPT = rse_ttwopt)
	println(df2)
	CSV.write("tucker_time_r$r.csv",df1)
	CSV.write("tucker_rse_r$r.csv",df2)
end