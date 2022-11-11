using LinearAlgebra
using TensorToolbox
using Statistics
using Optim
using Random
using EllipsisNotation
include("core_changes.jl")

#Gerador do Tensor W
function weight_tensor(sz, missing_rate)
    p = prod(sz)
    Ω = randperm(p)
    Ω = Ω[1:Int(round((1-missing_rate)*p))]
    W = zeros(sz...)
    W[Ω] .= 1
    return W
end

#HEURÍSTICAS
function parafac_completion(T, W, posto, tol, max_iter, fig::Bool)
    list_erro = [1.]
    iter = 1
    X = copy(T)
    normT = norm(T)
    while list_erro[end] > tol && iter<=max_iter
        X = full(cp_als(X, posto))
        X_last = X
        X = convert.(Float32, @. ifelse(W==1, T, X))
        iter = iter + 1
        append!(list_erro, norm(X - X_last)/normT)
    end

    if fig
    X[X.<0] .= 0
    X[X.>1] .= 1
    end
    
    return X, list_erro, iter
end

function hosvd_completion(T, W, posto, tol, max_iter, fig::Bool)
    list_erro = [1.]
    iter = 1
    X = copy(T)
    normT = norm(T)
    while list_erro[end] > tol && iter<=max_iter
        X = full(hosvd(X, reqrank=posto))
        X_last = X
        X = convert.(Float32, @. ifelse(W==1, T, X))
        iter = iter + 1
        append!(list_erro, norm(X - X_last)/normT)
    end

    if fig
    X[X.<0] .= 0
    X[X.>1] .= 1
    end
    
    return X, list_erro, iter
end

#KET AUGMENTATION -><><><><><><><><><><><><><><><><><><><><><><><><><><><-
function ket_augmentation(tensor, tam::Int)
    
    """
    Esta função mapeia uma imagem do tipo 2^n x 2^n x 3 para um tensor de ordem n + 1 com dimensões
    dadas pela variável tam.
    
    Ex: T ∈ ℝ^(16 × 16 × 3), então ket_augmentation(T) ∈ ℝ^(4 × 4 × 4 × 4 × 3)
    """
    dim1, dim2, dim3 = size(tensor)
    n_dim = Int8(log(dim1)/log(tam) + log(dim2)/log(tam)) #dimensão do modo
    new_dim = [] # dimensão do mapeamento
    for _ = 1:n_dim
       append!(new_dim, tam)
    end
    append!(new_dim, dim3)
    
    new_tensor = zeros(new_dim...)
    d = Int(sqrt(tam))    # dimensão do lado do bloco
    
    ind_bloco = reshape(collect(1:4), d, d)    #indexamento do elemento em um certo bloco
    novo_ind = zeros(Int, dim1, dim2, n_dim)    #armazena os índices dos elementos da matriz neste mapeamento
    
    # passagem pelos elementos da matriz
    for i=0:dim1-1
        for j=0:dim2-1
            x = i
            y = j
            
            # determinação dos índices em cada bloco
            for k=n_dim:-1:1
                #verifica qual a posição no bloco [1 2; 3 4]
                indx = Int(x % d)
                indy = Int(y % d)
                novo_ind[i+1, j+1, k] = ind_bloco[indx+1, indy+1]
                
                # divisão exata para prosseguir ao próximo bloco 
                x = x ÷ d
                y = y ÷ d
            end
            lista_ind = vec(novo_ind[i+1, j+1, :])
            new_tensor[lista_ind...,:] = tensor[i+1, j+1, :]
        end
    end
    
    return new_tensor, novo_ind
end

function inverse_ket_augmentation(tensor, novo_ind)
    """
    Realiza o mapeamento inverso do ket augmentation, isto é, leva os elementos do tensor na matriz 
    dada por 2ⁿ × 2ⁿ × 3 elementos.
    """
    tam = size(tensor)[1]
    dim1, dim2, n_dim = size(novo_ind)
    d = Int(sqrt(tam))
    
    new_tensor = zeros(dim1, dim2, 3)
    passo = zeros(n_dim)
    
    for i = 1:n_dim
        passo[i] = d^(n_dim - i)
    end
   
    for i = 1:dim1
        for j = 1:dim2
            ind_tensor = novo_ind[i, j, :]
            x = Int(dot(x_to_matrix(ind_tensor, tam),passo)) + 1
            y = Int(dot(y_to_matrix(ind_tensor, tam),passo)) + 1
            new_tensor[x, y, :] = tensor[ind_tensor...,:]
        end
    end
    return new_tensor
end

function x_to_matrix(vetor, tam)
    """
    Esta função realiza o mapeamento dos índices de um elemento do tensor para
    a i-ésima linha da matriz.
    
    """
    novo_vetor = zeros(length(vetor))
    d = Int(sqrt(tam))
    indx = zeros(Int, d, d)
    
    for i = 0:d-1
        indx[i+1,:] .= i
    end
    
    indx = vec(indx)
    for i = 0:length(vetor)-1
        novo_vetor[i+1] = indx[vetor[i+1]]
    end
    return novo_vetor
end
    
function y_to_matrix(vetor, tam)
    """
    Esta função realiza o mapeamento dos índices de um elemento do tensor para
    a j-ésima coluna da matriz.
    
    """
    novo_vetor = zeros(length(vetor))
    d = Int(sqrt(tam))
    indy = zeros(Int, d, d)

    for i = 0:d-1
        indy[:,i+1] .= i
    end
    indy = vec(indy)

    for i = 0:length(vetor)-1
        novo_vetor[i+1] = indy[vetor[i+1]]
    end
    return novo_vetor
end
#-<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>-

#Solvers dos problemas de completamento -----------------------------------------------

#Função proximal
function 𝒟(X, α)
    U, S, V = svd(X)
    S = Diagonal(max.(S .- α, 0))
    D = U*S*V'
    return D
end

#High Accuracy Low Rank Tensor Completion (HALRTC
function HaLRTC(𝕋, α, ρ, tol, max_iter, fig::Bool)
    num_modos = ndims(𝕋)
    sz = size(𝕋)
    conhecido = @. ifelse(𝕋!=0, 1, 0)
    
    𝕏 = copy(𝕋)
    𝕄 = zeros(sz..., num_modos) # 𝕄ᵢ
    𝕐 = zeros(sz..., num_modos) # 𝕐ᵢ
    normT = norm(𝕋)
    list_erros = [1.]
    iter = 1

    while list_erros[end]>tol && iter <= max_iter
        #Atualização de 𝕄
        for modo = 1:num_modos
            𝕄[..,modo] = matten(𝒟(tenmat(𝕏 + 𝕐[..,modo]./ρ, modo), α[modo]/ρ), modo, [sz...])
        end
        
        #Atualização de 𝕏
        𝕏_last = 𝕏
        𝕏 = sum(i -> 𝕄[..,i] - 𝕐[..,i]/ρ, 1:num_modos)/num_modos
        𝕏 = @. ifelse(conhecido==1, 𝕋, 𝕏)
        
        #Atualização de 𝕐
        for modo = 1:num_modos
            𝕐[.., modo] = 𝕐[.., modo] + ρ*(𝕏 - 𝕄[.., modo])
        end
        erro = norm(𝕏 - 𝕏_last)/normT
        append!(list_erros, erro)
        iter = iter + 1
    end

    if fig
    𝕏[𝕏.<0] .= 0
    𝕏[𝕏.>1] .= 1
    end

    return 𝕏, list_erros, iter
end

#Simple Low Rank Tensor Completion (SiLRTC)
function SiLRTC(𝕋, α, β, tol, max_iter, fig::Bool)
    sz = size(𝕋)
    num_modos = length(sz)
    conhecido = @. ifelse(𝕋!=0, 1, 0)
    sum_beta = sum(β)
    normT = norm(𝕋)
    list_erro = [1.]
    iter = 1
    𝕏 = copy(𝕋)
    
    while list_erro[end]>tol && iter ≤ max_iter
        𝕏_sum = zeros(size(𝕋)...)
        for modo = 1:num_modos
            𝕄 = 𝒟(tenmat(𝕏 , modo), α[modo]/β[modo])
            𝕏_sum = 𝕏_sum + matten(𝕄, modo, [sz...])*β[modo]  
        end
        
        #Atualização de 𝕏
        
        𝕏_last = 𝕏
        𝕏 = 𝕏_sum./sum_beta
        𝕏 = @. ifelse(conhecido==1, 𝕋, 𝕏)
        
        append!(list_erro, norm(𝕏 - 𝕏_last)/normT)
        iter = iter + 1
    end

    if fig
    𝕏[𝕏.<0] .= 0
    𝕏[𝕏.>1] .= 1
    end

    return 𝕏, list_erro, iter
end

#--------------------------------------------------------------------------------------

#Tensor Completion by Parallel Matrix Factorization via Tensor Train (TMac-TT)

#Estima os postos-TT com um limitante λ.

function TTrank_thresholding(𝕏, λ)
    N = length(size(𝕏))
    TTrank = zeros(Int, N-1)
    dim = size(𝕏)
    T = copy(reshape(𝕏, dim[1], prod(dim[2:end])))
    
    for k = 1:N-1
        U, S, V, r = rank_with_thresholding(T, λ)
        T = S*V'
        new_size = tuple(append!([r...], dim[k+1:end])...)
        T = copy(reshape(T, new_size))

        if k <= N-2
            T = copy(reshape(T, r*dim[k+1], prod(dim[k+2:end])))
        end
        
        TTrank[k] = r
    end
    return TTrank
end

function rank_with_thresholding(X, λ) 
    U, S, V = svd(X)
    r = 0
    
    for k=1:length(S)
        if S[k]/S[1] > λ
            r = r + 1
        end
    end
        r = max(r, 2)
        U = U[:,1:r]
        S = Diagonal(S[1:r])
        V = V[:,1:r]
        return U,S,V,r
end

#Cálcula os pesos de cada modo na formulação
function pesos_modos(dim)
    N = length(dim)
    pesos = zeros(N-1)
    modo_esq = dim[1]
    for k = 1:N-1
        modo_dir = prod(dim[k+1:end])
        pesos[k] = min(modo_dir, modo_esq)
        modo_esq = modo_esq*dim[k+1]
    end
    soma = sum(pesos)
    pesos = pesos./soma
    return pesos
end

#Simple Low Rank Tensor Completion via Tensor Train (SiLRTC-TT)
function SiLRTC_TT(𝕋, α, β, tol, max_iter, fig::Bool)
    num_modos = ndims(𝕋)
    sz = size(𝕋)
    conhecido = @. ifelse(𝕋!=0, 1, 0)
    sum_beta = sum(β)
    normT = norm(𝕋)
    list_erro = [1.]
    iter = 1
    
    𝕏 = copy(𝕋)
    while list_erro[end]>tol && iter<=max_iter
        𝕏_sum = zeros(sz...)
        for modo = 1:num_modos-1
            𝕄 = 𝒟(tenmat(𝕏 , row = [1:modo...], col=[modo+1:num_modos...]), α[modo]/β[modo])
            𝕏_sum = 𝕏_sum .+ matten(𝕄, [1:modo...], [modo+1:num_modos...], [sz...]).*β[modo]  
        end
        #Atualização de 𝕏
        
        𝕏_last = 𝕏
        𝕏 = 𝕏_sum./sum_beta
        𝕏 = @. ifelse(conhecido==1, 𝕋, 𝕏)
        append!(list_erro, norm(𝕏 - 𝕏_last)/normT)
        
        iter = iter + 1
    end
    
    if fig
    𝕏[𝕏.<0] .= 0
    𝕏[𝕏.>1] .= 1
    end
    
    𝕏 = convert.(Float64, 𝕏)
    return 𝕏, list_erro, iter
end

#Método Tmac-TT 

function TMac_TT(𝕋, λ, max_iter)
    𝕋_new, tensor_index = ket_augmentation(𝕋, 4)
    dim = tuple([4*ones(Int, 8)...,3]...)
    pesos = pesos_modos(dim)
    num_modos = length(size(𝕋_new))
    conhecido = @. ifelse(𝕋_new!=0, 1, 0)
    normT = norm(𝕋)

    X_approx = copy(𝕋_new)
    X_last = copy(X_approx)
    X = zeros((dim...,num_modos-1))
    
    for k=1:num_modos-1
        X[:,:,:,:,:,:,:,:,:,k] = copy(𝕋_new)
    end
    
    lista_U = []
    lista_V = []
    erro = []

    rankTT = TTrank_thresholding(X, λ)
    
    for k = 1:num_modos-1
        push!(lista_U, 0.1*rand(4^k, rankTT[k]))
        push!(lista_V, 0.1*rand(rankTT[k], 4^(8-k)*3))
        
    end
    
    for iter = 1:max_iter
        
        if mod(iter, 10) == 0
            println(rankTT)
            λ = λ*0.9 #0.7
            rankTT = TTrank_thresholding(X, λ)
            
            lista_U = []
            lista_V = []

            for k = 1:num_modos-1
                push!(lista_U, 0.1*rand(4^k, rankTT[k]))
                push!(lista_V, 0.1*rand(rankTT[k], 4^(8-k)*3))
            end
        end

        for k = 1:num_modos-1

            unfold_X = reshape(X[:,:,:,:,:,:,:,:,:,k], 4^k, 4^(8-k)*3)
            #Atualização
            
            lista_U[k] = unfold_X*permutedims(lista_V[k], [2,1])
            aux = pinv(permutedims(lista_U[k], [2,1])* lista_U[k])
            lista_V[k] = (aux*permutedims(lista_U[k], [2,1]))*unfold_X
            produto = lista_U[k]*lista_V[k]
            X[:,:,:,:,:,:,:,:,:,k] = reshape(produto, dim)
        end
    
        X_approx = sum(i-> X[:,:,:,:,:,:,:,:,:,i]*pesos[i], 1:num_modos-1)
        X_approx = @. ifelse(conhecido==1, 𝕋_new, X_approx)

        push!(erro , norm(X_approx - X_last))
        X_last = copy(X_approx)
        for k = 1: num_modos-1
           X[:,:,:,:,:,:,:,:,:,k] = copy(X_approx)
        end


    end   
    sol = inverse_ket_augmentation(X_approx,tensor_index)

    sol[sol.<0] = 0.0
    sol[sol.>1] = 1.0
    return sol, erro
end

function TMac_TT_v2(𝕋, λ, tol, max_iter, redo, fig::Bool)
    dim = size(𝕋)
    pesos = pesos_modos(dim)
    num_modos = length(dim)
    conhecido = @. ifelse(𝕋!=0, 1, 0)
    normT = norm(𝕋)

    X_approx = copy(𝕋)
    X_last = copy(X_approx)
    
    X = zeros((dim...,num_modos-1))
    
    for k=1:num_modos-1
        X[..,k] = copy(𝕋)
    end

    lista_U = []
    lista_V = []
    list_erro = [1.]
    iter = 1

    rankTT = TTrank_thresholding(X, λ)
    for k = 1:num_modos-1
        push!(lista_U, 0.1*rand(prod(dim[1:k]), rankTT[k]))
        push!(lista_V, 0.1*rand(rankTT[k], prod(dim[k+1:end])))
        
    end
    
    while list_erro[end]>tol && iter <= max_iter
        
        if redo==1
            if iter>=2 && abs(1 - (list_erro[end]/list_erro[end-1]))<0.1
                println(rankTT)
                #λ = λ*0.7 #0.7, 0.8
                #rankTT = TTrank_thresholding(X, λ)
                
                if minimum(rankTT)<30
                    rankTT = rankTT.+ 3
                end

                lista_U = []
                lista_V = []

                for k = 1:num_modos-1
                    push!(lista_U, 0.1*rand(prod(dim[1:k]), rankTT[k]))
                    push!(lista_V, 0.1*rand(rankTT[k], prod(dim[k+1:end])))
                end
            end
        end

        for k = 1:num_modos-1

            unfold_X = reshape(X[..,k], prod(dim[1:k]), prod(dim[k+1:end]))
            #Atualização
            
            lista_U[k] = unfold_X*permutedims(lista_V[k], [2,1])
            aux = pinv(permutedims(lista_U[k], [2,1])* lista_U[k])
            lista_V[k] = (aux*permutedims(lista_U[k], [2,1]))*unfold_X
            produto = lista_U[k]*lista_V[k]
            X[..,k] = reshape(produto, dim)
        end
    
        X_approx = sum(i-> X[..,i]*pesos[i], 1:num_modos-1)
        X_approx = @. ifelse(conhecido==1, 𝕋, X_approx)

        push!(list_erro , norm(X_approx - X_last)/normT)
        X_last = copy(X_approx)
        for k = 1: num_modos-1
           X[..,k] = copy(X_approx)
        end

        iter = iter + 1
    end   
    sol = X_approx

    if fig
    sol[sol.<0] .= 0.0
    sol[sol.>1] .= 1.0
    end

    return sol, list_erro, iter
end
#----------------------------><><><><-------------------Funções do TTWOPT------------------------------

# Divide os núcleos-TT a p partir do k-ésimo modo
function divide_cores(G, k)
    N = length(G)

    if k==1
        G_left = 1
        aux = G[2]
        for i = 2:N-1
            L = reshape(aux, tuple([Int(length(aux)/size(G[i], 3)), size(G[i],3)]...))
            R = reshape(G[i+1], tuple([size(G[i+1],1), Int(length(G[i+1])/size(G[i+1],1))]...))
            aux = L*R
        end
        ind = []
        for i=2:N
            append!(ind, size(G[i],2))
        end
        ind = vcat(size(G[2],1), ind)
        G_right = reshape(aux, tuple(ind...))

    elseif k==N
        G_right = 1
        aux = G[1]
        for i = 1:N-2
            L = reshape(aux, tuple([Int(length(aux)/size(G[i], 3)), size(G[i],3)]...))
            R = reshape(G[i+1], tuple([size(G[i+1],1), Int(length(G[i+1])/size(G[i+1],1))]...))
            aux = L*R
        end
        ind = []
        for i = 1:N-1
            append!(ind, size(G[i],2))
        end
        append!(ind, size(G[N],1))
        G_left = reshape(aux, tuple(ind...))
    
    else
        aux = G[1]
        
        for i=1:k-2
            L = reshape(aux, tuple([Int(length(aux)/size(G[i], 3)), size(G[i],3)]...))
            R = reshape(G[i+1], tuple([size(G[i+1],1), Int(length(G[i+1])/size(G[i+1],1))]...))
            aux = L*R
        end
        ind = []
        for i=1:k-1
            append!(ind, size(G[i],2))
        end
        ind = append!(ind, size(G[k-1],3))
        G_left = reshape(aux, tuple(ind...))

        aux = G[k+1]
        for i=k+1:N-1
            L = reshape(aux, tuple([Int(length(aux)/size(G[i], 3)), size(G[i],3)]...))
            R = reshape(G[i+1], tuple([size(G[i+1],1), Int(length(G[i+1])/size(G[i+1],1))]...))
            aux = L*R
        end
        ind = []
        for i=k+1:N
            append!(ind, size(G[i],2))
        end
        ind = vcat(size(G[k+1],1), ind)
        G_right = reshape(aux, tuple(ind...))
    end

    return G_left, G_right
end

#Inicializa os núcleos-TT
function init_TTcores(size, postoTT)
    N = length(size)
    G = CoreCell(undef, N)
    for i = 2:N-1
         G[i] = randn(postoTT[i-1], size[i], postoTT[i])
    end
    G[1] = randn(1, size[1], postoTT[1])
    G[N] = randn(postoTT[N-1], size[end], 1)

    for i = 1:N
        G[i] = G[i]./maximum(abs.(G[i][:]))
    end
    return G
end

function grad_f(Y, G_vec, postoTT)
    sz = size(Y)
    G = mat2_to_core(vec_to_mat2(G_vec,sz,postoTT), postoTT)
    Z = W.*core_to_tensor(G, sz, postoTT)

    N = length(sz)
    ∇f = Array{Any, 1}(undef, N)
    T = Z - Y
    for k=1:N
        G_left, G_right = divide_cores(G, k)
        
        if k==1
            prod = tenmat(G_right, 1)
        
        elseif k==N
            prod = tenmat(G_left, length(size(G_left)))
        
        else
            prod = tkron(tenmat(G_right, 1), tenmat(G_left, k))
        end

        ∇f[k] = tenmat(T, k)*permutedims(prod, [2,1])
        
    end

    ∇f = convert.(Float64,mat2_to_vec(∇f))
    return ∇f
end


function grad_f2(X,W,G,postoTT)
    sz = size(X)
    Y = W.*X
    Z = W.*core_to_tensor()

    fun = 0.5*norm(Z-Y)^2
    N = length(sz)
    ∇f = Array{Any, 1}(undef, N)

    T = Z - Y
    for k=1:N
        G_left, G_right = divide_cores(G, k)
        
        if k==1
            prod = tenmat(G_right, 1)
        
        elseif k==N
            prod = tenmat(G_left, length(size(G_left)))
        
        else
            prod = tkron(tenmat(G_right, 1), tenmat(G_left, k))
        end

        ∇f[k] = tenmat(T, k)*permutedims(prod, [2,1])
        
    end
    return fun, ∇f
end

function fg!(F, D, G_vec)
    if D!=nothing
        D .= grad_f(Y, G_vec, postoTT)
    end

    if F!=nothing
        F = f(Y, G_vec, sz, postoTT)
    end
end


function TT_WOPT(X, W, postoTT, η, tol, max_iter)
    sz = size(X)
    N = length(sz)
    G = init_TTcores(sz, postoTT)
    iter = 0
    Y = copy(W.*X)
    Z = copy(W.*full(TTtensor(G)))
    erro = []
    list_f = []

    while iter<=max_iter
        f, ∇f = grad_f2(X, W, G, postoTT)
        G_mat = core_to_mat2(G)

        for k = 1:N
            G_mat[k] = G_mat[k] - η.*∇f[k]
        end

        G = mat2_to_core(G_mat, postoTT)
        Z = copy(W.*full(TTtensor(G)))
        append!(list_f, f)
        append!(erro, norm(Y-Z))

        iter = iter + 1
    end
    return Z, list_f, erro
end

function TT_WOPT_v2(X, W, postoTT, η, max_iter)
    sz = size(X)
    Y = W.*X
    N = length(sz)
    core_0 = init_TTcores(sz, postoTT)
    G_vec = mat2_to_vec(core_to_mat2(core_0))
    G_vec = convert.(Float64, G_vec)

    ∇f = grad_f(Y, G_vec, postoTT)

    G = mat2_to_core(vec_to_mat2(G_vec,sz,postoTT), postoTT)
    Z = W.*full(TTtensor(G))
    f = 0.5.*norm(Y-Z)^2
    list_f = [f]
    list_∇f = [norm(∇f)]

    iter = 2
        while (iter<=max_iter)
            G_vec = G_vec .- η*∇f
            ∇f = grad_f(Y, G_vec, postoTT)
            
            G = mat2_to_core(vec_to_mat2(G_vec,sz,postoTT), postoTT)
            Z = W.*full(TTtensor(G))
            f = 0.5*norm(Y-Z)^2

            append!(list_f, f)
            append!(list_∇f, norm(∇f))
            iter = iter + 1
        end

    Z = mat2_to_core(vec_to_mat2(G_vec, sz, postoTT), postoTT)
    Z = core_to_tensor(Z, sz, postoTT)
    Z = @. ifelse(W!=0, Y, Z)
    return Z, list_f, list_∇f, iter
end

function f(Y, G_vec, sz, postoTT)
    aux = mat2_to_core(vec_to_mat2(G_vec, sz, postoTT), postoTT)
    Z = W.*core_to_tensor(aux, sz, postoTT)   
    return 0.5*norm(Y-Z)^2 #conferir a norma
end

function TT_WOPT_v3(X, W, sz, postoTT, show_trace::Bool, fig::Bool)
    N = length(sz)
    core_0 = init_TTcores(sz, postoTT)
    core_0 = mat2_to_vec(core_to_mat2(core_0))
    core_0 = convert.(Float64, core_0)

    res = optimize(Optim.only_fg!(fg!), core_0, ConjugateGradient(linesearch = MoreThuente(gtol = 0.01, x_tol=1e-15, maxfev=20)), Optim.Options(show_trace=show_trace, f_tol=1e-8, x_tol = 1e-4, iterations=1000))

    Z = mat2_to_core(vec_to_mat2(res.minimizer,sz,postoTT),postoTT)
    Z = core_to_tensor(Z, sz, postoTT)

    if fig
    Z[Z.<0] .= 0.0
    Z[Z.>1] .= 1.0
    end

    return Z
end

function TT_WOPT_v4(X, subX, W, subW, sz, postoTTmax, δ, γ, diminuir::Int64, show_trace::Bool, fig::Bool)
    N = length(sz)
    normX = norm(X)
    postoTT = ones(Int, N-1)
    old_cores = reorth(init_TTcores(sz, postoTT))
    Z_last = core_to_tensor(old_cores, sz, postoTT)
    terminou = false

    global postoTT

    cores = copy(old_cores)

    for k = 2:postoTTmax
        for μ = 1:N-1

            ε1 = ε_Ω(subX, Z_last, subW)
            r = postoTT[μ]

            if μ==1 && r == sz[1]
                continue
            end

            if r ≥ postoTTmax
                println("postoTT máximo")
                continue
            else
                

                increase_TTrank!(cores, postoTT, μ)
                tam = size(cores[μ])
                if Int(tam[1]*tam[2]) >= tam[3]
                    println("orthog")
                    cores = reorth(cores)
                end

                cores_vec = mat2_to_vec(core_to_mat2(cores))
                cores_vec = convert.(Float64, cores_vec)

                println(postoTT)
                println("otimizando...")
                res = optimize(Optim.only_fg!(fg!), cores_vec, ConjugateGradient(linesearch = MoreThuente(gtol = 0.01, x_tol=1e-15, maxfev=20)), Optim.Options(show_trace=show_trace, f_tol=1e-8, x_tol = 1e-8, iterations=500))
                
                cores = mat2_to_core(vec_to_mat2(res.minimizer,sz,postoTT),postoTT)

                Z = core_to_tensor(cores, sz, postoTT)
                ε2 = ε_Ω(subX, Z, subW)

                    println("eps1 = $ε1")
                    println("eps2 = $ε2")                

                if ε2 < 0.1 && diminuir==2
                    δ = δ/10
                    println("diminuiu")
                    diminuir = 1

                elseif ε2 < 0.01 && diminuir==1
                    δ = δ/10
                    println("diminuiu")
                    diminuir = 0
                end


                if  (abs(ε1 - ε2)/ε1) < δ || ε2 > γ*ε1
                    println("voltou atrás")
                    postoTT[μ] -= 1
                    cores = copy(old_cores)
                else
                    Z_last = copy(Z)
                    old_cores = copy(cores)
                end

                if ε2 < 1e-06
                    terminou = true
                    break
                end

            end
        end
        if terminou
            println("terminou antes...")
            break
        end
    end
    Z = Z_last
    if fig
    Z[Z.<0] .= 0.0
    Z[Z.>1] .= 1.0
    end

    return Z, postoTT
end

"""
function SR1(X, W, postoTT, η, max_iter)
    r = 10^-8
    sz = size(X)
    Y = W.*X
    N = length(sz)
    core_0 = init_TTcores(sz, postoTT)
    G_vec = mat2_to_vec(core_to_mat2(core_0))
    G_vec = convert.(Float64, G_vec)
    tam = length(G_vec)

    ∇f = grad_f(Y, G_vec, postoTT)
    Hk = Diagonal(ones(tam))

    G = mat2_to_core(vec_to_mat2(G_vec,sz,postoTT), postoTT)
    Z = W.*full(TTtensor(G))
    f = 0.5.*norm(Y-Z)^2
    list_f = [f]
    list_∇f = [norm(∇f)]

    iter = 2
    println("entrou no while")
        while (iter<=max_iter)
            G_vec = G_vec .- η*(Hk*∇f)
            pk = -Hk*∇f

            ∇f = grad_f(Y, G_vec, postoTT)
            G_vec_next = G_vec + η.*pk

            ∇f_next = grad_f(Y, G_vec_next, postoTT)
            yk = ∇f_next - ∇f
            sk = η.*pk

            if abs(((sk - (Hk * yk))' * yk)) >= r * sqrt(sum((sk - (Hk * yk)) .^ 2)) * sqrt(sum(yk .^ 2))
                Hk = Hk + (((sk - (Hk * yk)) * (sk - (Hk * yk))') / ((sk - (Hk * yk))' * yk));
            end

            Y = W.*X
            G = mat2_to_core(vec_to_mat2(G_vec,sz,postoTT), postoTT)
            Z = W.*full(TTtensor(G))
            f = 0.5*norm(Y-Z)^2

            append!(list_f, f)
            append!(list_∇f, norm(∇f))
            iter = iter + 1
        end

        Z = mat2_to_core(vec_to_mat2(G_vec, sz, postoTT), postoTT)
        Z = core_to_tensor(Z, sz, postoTT)
        return Z, list_f, list_∇f, iter
end
"""

function mapping(X)
    v = 2*ones(Int, 16)
    new_dim = [v..., 3]
    new_idx = [1 9 2 10 3 11 4 12 5 13 6 14 7 15 8 16 17]
    new_X = reshape(X, new_dim...)
    new_X = permutedims(new_X, new_idx)
    new_X = reshape(new_X, 4,4,4,4,4,4,4,4,3)
    return new_X
end

function inverse_mapping(X)
    v = 2*ones(Int, 16)
    new_dim = [v..., 3]
    old_idx = [1 3 5 7 9 11 13 15 2 4 6 8 10 12 14 16 17]
    old_X = reshape(X, new_dim...)
    old_X = permutedims(old_X, old_idx)
    old_X = reshape(old_X, 256, 256, 3)
    return old_X
end

function increase_TTrank!(G, postoTT, μ)
    aux1 = G[μ]
    aux2 = G[μ+1]
    r_mu = postoTT[μ]

    if μ == 1
        r_mu1 = postoTT[μ+1]
        L = reshape(aux1, Int(size(aux1, 2)), Int(r_mu))
        R = reshape(aux2, Int(r_mu), Int(size(aux2, 2)*postoTT[μ+1]))

    elseif μ == length(postoTT)
        L = reshape(aux1, Int(postoTT[μ-1]*size(aux1, 2)), Int(r_mu))
        R = reshape(aux2, Int(r_mu), Int(size(aux2, 2)))

    else
        r_mu1 = postoTT[μ+1]
        L = reshape(aux1, Int(postoTT[μ-1]*size(aux1, 2)), Int(r_mu))
        R = reshape(aux2, Int(r_mu), Int(size(aux2, 2)*postoTT[μ+1]))
    end

    szL = size(L)
    szR = size(R)

    newL = zeros(szL[1], szL[2]+1)
    newR = zeros(szR[1]+1, szR[2])

    newL[:, 1:end-1] = L
    newL[:, end] = (1e-04)*randn(szL[1])

    newR[1:end-1, :] = R
    newR[end, :] = (1e-04)*randn(szR[2])


    if μ==1
        G[μ] = reshape(newL, 1, size(aux1, 2), r_mu+1)
        G[μ+1] = reshape(newR, r_mu+1, size(aux2,2), postoTT[μ+1])
        postoTT[μ] +=1
    elseif μ == length(postoTT)
        G[μ] = reshape(newL, postoTT[μ-1], size(aux1, 2), r_mu+1)
        G[μ+1] = reshape(newR, r_mu+1, size(aux2,2), 1)
        postoTT[μ] +=1
    else
        G[μ] = reshape(newL, postoTT[μ-1], size(aux1, 2), r_mu+1)
        G[μ+1] = reshape(newR, r_mu+1, size(aux2,2), postoTT[μ+1])
        postoTT[μ] +=1
    end
end

function W_c(W, sz, taxa)
    subW = weight_tensor(sz, taxa)
    for i = prod(sz)
        if W[i] == 1
            subW[i] = 0
        end
    end
    return subW
end

function ε_Ω(subA, Z, subW)
    return norm(subA - (subW.*Z))/norm(subA)
end








