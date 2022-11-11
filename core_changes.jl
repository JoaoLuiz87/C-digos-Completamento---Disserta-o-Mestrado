using LinearAlgebra
using TensorToolbox
using Statistics

function core_to_mat2(G_ten)
    core_list = []
    N = length(G_ten)
    for i = 1:N
        push!(core_list, tenmat(G_ten[i],2))
    end
    return core_list
end

function mat2_to_vec(G_mat)
	N = length(G_mat)
	v = []
	for i = 1:N
		aux = vec(G_mat[i])
		v = vcat(v, aux)
	end
	return v
end

function vec_to_mat2(G_vec, sz, r)
	N = length(sz)
	G_mat = Array{Array{Float64, 2}, 1}(undef, N)
	I = zeros(N)
	for i = 2:N-1
		I[i] = r[i-1]*sz[i]*r[i]
	end
	I[1] = sz[1]*r[1]
	I[N] = sz[N]*r[N-1]

	ind = []
	soma = I[1]

	for i=1:N-1
		append!(ind, soma)
		soma = soma + I[i+1]
	end

	append!(ind, soma)
	ind = vcat(0, ind)
	ind = Int.(ind)

	for i = 1:N
		G_mat[i] = zeros(length(G_vec[ind[i]+1:ind[i+1]]),1)
		G_mat[i][:,1] = G_vec[ind[i]+1:ind[i+1]]
	end
	
	for i=2:N-1
		G_mat[i]=reshape(G_mat[i],tuple([sz[i],r[i-1]*r[i]]...))
	end
	
	G_mat[1]=reshape(G_mat[1],tuple([sz[1],r[1]]...))
	G_mat[N]=reshape(G_mat[N],tuple([sz[N],r[N-1]]...))
	return G_mat
end

function mat2_to_core(G_mat, r)
	N = length(G_mat)
	core = CoreCell(undef, N)
	for i = 2:N-1
		core[i] = reshape(G_mat[i], tuple([size(G_mat[i],1), r[i-1], r[i]]...))
		core[i] = permutedims(core[i], [2,1,3])
	end
	core[1] = reshape(G_mat[1], tuple([size(G_mat[1],1), 1, r[1]]...))
	core[N] = reshape(G_mat[N], tuple([size(G_mat[N],1), r[N-1], 1]...))
	core[1] = permutedims(core[1], [2,1,3])
	core[N] = permutedims(core[N], [2,1,3])

	return core
end

function core_to_tensor(core, sz, postoTT)
	aux = core[1]
	postoTT = [1, postoTT..., 1]
	for i = 1:length(core)-1
		if (length(aux) % postoTT[i+1]) != 0
			println(size(aux))
			println(postoTT[i+1])
			println(i)
		end
		L = reshape(aux, tuple([Int(length(aux)/postoTT[i+1]), postoTT[i+1]]...))	
		R = reshape(core[i+1], tuple([postoTT[i+1], Int(length(core[i+1])/postoTT[i+1])]...))
		aux = L*R
	end
	tensor = reshape(aux, sz...)
	return tensor
end