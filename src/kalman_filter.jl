### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 740dc710-fbaf-11ea-2062-7f44056cbd12
using Distributions, LinearAlgebra

# ╔═╡ 7901f281-a3ff-477c-b292-6121ce12af32
using PlutoUI, Random

# ╔═╡ 83560088-c06d-4f9d-976e-63732ba206e6
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# ╔═╡ c52df260-fc99-11ea-00a2-3f21b9c40f3b
using Reel

# ╔═╡ 5bd86070-194f-11eb-21f9-f57ce27aa5d0
using Test

# ╔═╡ 2cbec03e-fb77-11ea-09a2-634fac25a12a
md"""
# Kalman filters

For code, see [StateEstimation.jl](https://github.com/mossr/StateEstimation.jl)
"""

# ╔═╡ 29e2d71e-fc40-11ea-0c55-f929ddc20588
md"""
# Standard Kalman filter
A special type of filter for continuous state spaces is known as the *Kalman filter*. The mean vector and covariance matrix that define the Gaussian belief are updated using a prediction. 
"""

# ╔═╡ 09fc2050-fc46-11ea-2bc4-257edf069912
md"""
$$\begin{align}
𝛍_b \tag{mean belief vector}\\
𝚺_b \tag{covariance belief matrix}
\end{align}$$
"""

# ╔═╡ 419cda50-fc3b-11ea-2ecf-b521f3f44d38
mutable struct KalmanFilter
	μᵦ # mean vector
	Σᵦ # covariance matrix
end

# ╔═╡ 037674ae-fc41-11ea-025c-8510cc72063b
md"""
A *Kalman filter* assumes that $T$ and $O$ are linear-Gaussian and $b$ is Gaussian:

$$\begin{align}
\hspace{-3cm}T(𝐬^′ \mid 𝐬, 𝐚) &= \mathcal{N}(𝐬^′ \mid 𝐓_s 𝐬 + 𝐓_a 𝐚, 𝚺_s) \tag{linear-Gaussian transition}\\
\hspace{-3cm}O(𝐨 \mid 𝐬^′) &= \mathcal{N}(𝐨 \mid 𝐎_s 𝐬^′, 𝚺_o) \tag{linear-Gaussian observation}\\
\hspace{-3cm}b(𝐬) &= \mathcal{N}(𝐬 \mid 𝛍_b, 𝚺_b) \tag{Gaussian belief}
\end{align}$$
Where $𝚺_s$ is the state transition covariance and $𝚺_o$ is the observation covariance.
"""

# ╔═╡ caef0200-fc3c-11ea-06e8-09595e3e00f3
struct POMDPᵏᶠ Tₛ; Tₐ; Oₛ; Σₛ; Σₒ end

# ╔═╡ 2334f0c0-fc40-11ea-1c88-67467bb5651f
md"## Belief update"

# ╔═╡ 3b261740-fc40-11ea-2253-012e10b5e6e6
md"""
### Kalman prediction
The *predict step* uses the transition dynamics to get a predicted distribution that is parameterized by the following mean and covariance.
"""

# ╔═╡ 80ae2940-fc42-11ea-3db3-bdf6de06f6df
md"""
$$\begin{align}
𝛍_p &← 𝐓_s 𝛍_b + 𝐓_a 𝐚 \tag{predicted mean}\\
𝚺_p &← 𝐓_s 𝚺_b 𝐓_s^\top + 𝚺_s \tag{predicted covariance}
\end{align}$$
"""

# ╔═╡ ca4f7bc0-fc3e-11ea-0588-d9468558d025
function kalman_predict(b::KalmanFilter, 𝒫::POMDPᵏᶠ, a)
	(μᵦ, Σᵦ) = (b.μᵦ, b.Σᵦ)
	(Tₛ, Tₐ, Σₛ) = (𝒫.Tₛ, 𝒫.Tₐ, 𝒫.Σₛ)

	μₚ = Tₛ*μᵦ + Tₐ*a   # predicted mean
	Σₚ = Tₛ*Σᵦ*Tₛ' + Σₛ # predicted covariance

	return (μₚ, Σₚ)
end

# ╔═╡ 469be5f0-fc40-11ea-2a7a-23c9356b4b44
md"""
### Kalman update
The *update step* uses the prediction to update our belief.
"""

# ╔═╡ 2318aed2-fc43-11ea-24e6-19c5342b76a2
md"""
$$\begin{align}
𝐊 &← \frac{𝚺_p 𝐎_s^\top}{𝐎_s 𝚺_p 𝐎_s^\top + 𝚺_o} \tag{Kalman gain}\\
𝛍_b &← 𝛍_p + 𝐊 \biggl(𝐨 - 𝐎_s 𝛍_p \biggr) \tag{updated mean}\\
𝚺_b &← \biggl(𝐈 - 𝐊 𝐎_s \biggr)𝚺_p \tag{updated covariance}
\end{align}$$
Notice the `!` indicates that the belief is modified in-place.
"""

# ╔═╡ f6437792-fc3e-11ea-2941-2ba90b95ecee
function kalman_update!(b::KalmanFilter, 𝒫::POMDPᵏᶠ, o, μₚ, Σₚ)		
	(μᵦ, Σᵦ) = (b.μᵦ, b.Σᵦ)
	(Tₛ, Tₐ, Oₛ) = (𝒫.Tₛ, 𝒫.Tₐ, 𝒫.Oₛ)
	(Σₛ, Σₒ) = (𝒫.Σₛ, 𝒫.Σₒ)
	
	K = Σₚ*Oₛ' / (Oₛ*Σₚ*Oₛ' + Σₒ) # Kalman gain
	μᵦ′ = μₚ + K*(o - Oₛ*μₚ)      # updated mean
	Σᵦ′ = (I - K*Oₛ)*Σₚ           # updated covariance
	
	b.μᵦ = μᵦ′
	b.Σᵦ = Σᵦ′
end

# ╔═╡ 597bd862-fc3b-11ea-2c14-497f8746c4f3
function update!(b::KalmanFilter, 𝒫::POMDPᵏᶠ, a, o)
	(μₚ, Σₚ) = kalman_predict(b, 𝒫, a)
	kalman_update!(b, 𝒫, o, μₚ, Σₚ)
end

# ╔═╡ 038c5510-f8bc-11ea-0fc5-7d765d868496
md"""
# POMDP: Agent definition
Agent randomly walking in a $10\times10$ continuous 2D environment.
"""

# ╔═╡ cbeb9ffc-3267-4d0b-a395-f10844d60dad
DOMAIN = [-5, 5] # xy-domain of the 2D environment

# ╔═╡ 608a4850-f7e8-11ea-2fca-af35a2f0456b
begin
    global 𝒮 = Product(Uniform.([DOMAIN[1], DOMAIN[1]], [DOMAIN[2], DOMAIN[2]]))
	𝒮ₘᵢₙ = minimum.(support.(𝒮.v))
	𝒮ₘₐₓ = maximum.(support.(𝒮.v))

	global 𝒜 = MvNormal([0, 0], [0.25 0; 0 0.25])
	global 𝒪 = MvNormal([0, 0], [1 0; 0 1])
	
	global T = (s,a) -> [clamp.(s .+ a, 𝒮ₘᵢₙ, 𝒮ₘₐₓ)] # deterministic transition
	global O = (s′,a) -> MvNormal(𝒪.μ + s′, 𝒪.Σ*abs.(a)) # observation distribution
end;

# ╔═╡ 4099e950-fb77-11ea-23b7-6d1f7b47c07e
md"# Simulation and testing"

# ╔═╡ 707e9b30-f8a1-11ea-0a6c-ad6756d07bbc
@bind t Slider(0:100, show_value=true, default=10)

# ╔═╡ de37b64a-1e98-416f-99b6-6c746307df93
md"""
## Plotting
"""

# ╔═╡ 4726f4a0-fc50-11ea-12f5-7f19d21d9bcc
function plot_covariance(P; cmap=:Blues, alpha=1) #87bcdc
    varX = range(DOMAIN[1], stop=DOMAIN[2], length=200)
    varY = range(DOMAIN[1], stop=DOMAIN[2], length=200)

    Z = [pdf(P, [x,y]) for y in varY, x in varX] # Note: reverse in X, Y.
	contour!(varX, varY, Z, levels=5, lw=1.5, c=cmap, alpha=alpha)
end

# ╔═╡ a2252160-fc5a-11ea-1c52-4717e186e8ff
md"""
# Extended Kalman filter
Extension to nonlinear dynamics with Gaussian noise:

$$\begin{align}
T(\mathbf s^\prime \mid \mathbf s, \mathbf a) &= \mathcal{N}(\mathbf s^\prime \mid \mathbf f_T(\mathbf s, \mathbf a),\, 𝚺_s)\\
O(\mathbf o \mid \mathbf s^\prime) &= \mathcal{N}(\mathbf o \mid \mathbf f_O(\mathbf s^\prime),\, 𝚺_o)
\end{align}$$

For differentialble functions $\mathbf f_T(\mathbf s, \mathbf a)$ and $\mathbf f_O(\mathbf s^\prime)$.

> **Key**. Uses a local linear approximation of the nonlinear dynamics.
"""

# ╔═╡ 98bc1fe0-0eb4-11eb-016d-1de4734348d3
md"""
## Jacobian
The local linear approximation, or *linearization*, is given by first-order Taylor expansions in the form of Jacobians. The Jacobian of a vector-valued function is a matrix of all partial derivatives.

For a multivariate function $$\mathbf{f}$$ with $n$ inputs and $m$ output, the Jacobian $\mathbf{J}_\mathbf{f}$ is:

$$\mathbf{J}_\mathbf{f} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\\
\vdots & \ddots & \vdots\\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} = (m \times n) \text{ matrix}$$

> See algorithm 19.4 for `ExtendedKalmanFilter` implementation.
"""

# ╔═╡ 5011a010-fc5a-11ea-22b8-df368e66c6cc
md"""
# Unscented Kalman filter 🧼
Derivative free! How clean!
"""

# ╔═╡ 9a55a8f0-fc5b-11ea-15c6-abb241ea8770
md"""
$$\begin{gather}
\mathbf f_T \tag{transition dynamics function}\\
\mathbf f_O \tag{observation dynamics function}
\end{gather}$$
"""

# ╔═╡ 7dfa2370-fc5b-11ea-3d5d-d54349446b89
struct POMDPᵘᵏᶠ fₜ; fₒ; Σₛ; Σₒ end

# ╔═╡ 6feab390-fc5a-11ea-1367-c5f353fadbc7
mutable struct UnscentedKalmanFilter
	μᵦ # mean vector
	Σᵦ # covariance matrix
	λ  # point-spread parameter
end

# ╔═╡ 48e7cb90-fc5d-11ea-0c29-c32610e59625
md"""
## Sigma point samples
$$\begin{align}
𝐬_1 &= 𝛍\\
𝐬_{2i} &= 𝛍 + \left(\sqrt{(n+\lambda)𝚺}\right)_i \quad \text{for } i \text{ in } 1\text{:}n\\
𝐬_{2i+1} &= 𝛍 - \left(\sqrt{(n+\lambda)𝚺}\right)_i \quad \text{for } i \text{ in } 1\text{:}n
\end{align}$$
"""

# ╔═╡ 545eda20-fc5a-11ea-1e32-bfe408c99b35
function sigma_points(μ, Σ, λ)
	n = length(μ)
	Δ = sqrt((n + λ) * Σ)
	S = [μ]
	for i in 1:n
		push!(S, μ + Δ[:,i])
		push!(S, μ - Δ[:,i])
	end
	return S
end

# ╔═╡ d0253f60-0eba-11eb-0283-e5b6e8c14ffe
md"""
Σ₁₁ = $(@bind Σ₁₁ Slider(1:5, default=1, show_value=true)) ... 
Σ₁₂ = $(@bind Σ₁₂ Slider(-5:5, default=0, show_value=true))

Σ₂₁ = $(@bind Σ₂₁ Slider(-5:5, default=0, show_value=true)) ...
Σ₂₂ = $(@bind Σ₂₂ Slider(1:5, default=1, show_value=true)) 

λ = $(@bind λₑₓ Slider(-1:10, default=2, show_value=true))
"""

# ╔═╡ 3e0aab30-0eb8-11eb-1e6c-81a3d89d1cfb
sigma_points([0.0, 0.0], [Σ₁₁ Σ₁₂; Σ₂₁ Σ₂₂], λₑₓ) # example

# ╔═╡ 00c94de0-fc74-11ea-0b52-a9b2938c5117
md"""
### Weights
$$\begin{align}
\lambda &= \text{spread parameter}\\
w_i &= \begin{cases}
\frac{\lambda}{n+\lambda} & \text{for } i=1\\
\frac{1}{2(n+\lambda)} & \text{otherwise}
\end{cases}
\end{align}$$
"""

# ╔═╡ e1d59150-fc73-11ea-2b67-ef871a9d12b5
weights(μ, λ; n=length(μ)) = [λ / (n + λ); fill(1/(2*(n + λ)), 2n)]

# ╔═╡ 999a5c19-f398-4769-9889-99f0597d3511
weights([0, 0], λₑₓ)

# ╔═╡ 84e1f980-0eba-11eb-0c52-878cc6e04534
function plot_sigma_points(μ, Σ, λ; hold=false)
	if !hold
		plot(legend=false, axisratio=:equal)
	end
	S = sigma_points(μ, Σ, λ)
	wₛ = weights(μ, λ)
	for (i,sᵢ) in enumerate(S)
		scatter!([sᵢ[1]], [sᵢ[2]], c="#326fb9", ms=2*5^wₛ[i], msw=0) # sigma points
	end
	xlims!(DOMAIN...)
	ylims!(DOMAIN...)
end

# ╔═╡ 7179e070-fc99-11ea-1f02-511f2215c6a8
function plot_kalman_filter(belief, true_state, iter, action)
	plot(legend=false, axisratio=:equal)
	
	P = MvNormal(belief.μᵦ, Matrix(Hermitian(belief.Σᵦ))) # numerical stability
	plot_covariance(P) # covariance contours

	if hasfield(typeof(belief), :λ)
		plot_sigma_points(belief.μᵦ, belief.Σᵦ, belief.λ; hold=true)
	end

	scatter!([true_state[1]], [true_state[2]], c=:red, alpha=0.8)# true state
    xlims!(DOMAIN...)
    ylims!(DOMAIN...)
    title!("iteration=$iter, action=$(round.(action, digits=2))", titlefontsize=10)
end

# ╔═╡ b1773e60-0eba-11eb-36e6-3f0e9e7cc3b6
plot_sigma_points([0.0, 0.0], [Σ₁₁ Σ₁₂; Σ₂₁ Σ₂₂], λₑₓ)

# ╔═╡ 5564aa00-fc5d-11ea-2a66-b5f9edef3f03
md"""
## Belief update
"""

# ╔═╡ 21998107-2eff-491c-9c23-b7f0d5a87c9a
md"""
##### Reconstruct original mean and covariance
If we wanted to reconstruct our provided mean and covariance using the generated sigma points $\mathbf{s}_i$, then we can use these equations (note, they don't pass the sigma points through the nonlinear function $\mathbf{f}$ like we do in the unscented transform).

$$\begin{align}
𝛍 &= \sum_i w_i 𝐬_i\\
𝚺 &= \sum_i w_i (𝐬_i - 𝛍)(𝐬_i - 𝛍)^\top
\end{align}$$
"""

# ╔═╡ 4f04e39e-fc5d-11ea-0b22-85563521ec7f
md"""
### Unscented transform
Reconstruct updated mean and covariance based on a nonlinear transform $\mathbf{f}$ of the sigma points $\mathbf{s}_i$.
"""

# ╔═╡ 9746c3d0-fc72-11ea-2d9a-5dbe53753813
md"""
$$\begin{align}
𝛍^′ &= \sum_i w_i 𝐟(𝐬_i)\\
𝚺^′ &= \sum_i w_i\bigl(𝐟(𝐬_i) - 𝛍^′\bigr)\bigl(𝐟(𝐬_i) - 𝛍^′\bigr)^\top
\end{align}$$
"""

# ╔═╡ f6aab420-fc5a-11ea-122f-d356a54e953c
function unscented_transform(μ, Σ, f, λ, wₛ)
	S = sigma_points(μ, Σ, λ)
	S′ = f.(S)
	μ′ = sum(w*s for (w,s) in zip(wₛ, S′))
	Σ′ = sum(w*(s - μ′)*(s - μ′)' for (w,s) in zip(wₛ, S′))
	return (μ′, Σ′, S, S′)
end

# ╔═╡ 5ddbdeb0-fc5d-11ea-3600-21920d6bf4a2
md"""
### UKF prediction
Predict where the agent is going based on the nonlinear transition function $\mathbf{f}_T$.
"""

# ╔═╡ d1fc7c70-fc5b-11ea-3c17-7f3e5bc58b44
function ukf_predict(b::UnscentedKalmanFilter, 𝒫::POMDPᵘᵏᶠ, a, wₛ)
	(μᵦ, Σᵦ, λ) = (b.μᵦ, b.Σᵦ, b.λ)
	(μₚ, Σₚ, _, _) = unscented_transform(μᵦ, Σᵦ, s->𝒫.fₜ(s,a), λ, wₛ)
	Σₚ += 𝒫.Σₛ
	return (μₚ, Σₚ)
end

# ╔═╡ 63dd3160-fc5d-11ea-223d-3119cff7630d
md"""
### UKF update
1. Update observation model using predicted mean and covariance.
2. Calculate the _cross covariance matrix_ (measures the variance between two multi-dimensional variables; here it's the transition prediction $\mu_p$ and observation model update $\mu_o$).
3. Update mean and covariance of our belief.
"""

# ╔═╡ befe14ee-ecd5-4f4e-a1af-40fdd0a61da8
function ukf_update_observation(b::UnscentedKalmanFilter, 𝒫::POMDPᵘᵏᶠ, μₚ, Σₚ, wₛ)
	(μₒ, Σₒ, Sₒ, Sₒ′) = unscented_transform(μₚ, Σₚ, 𝒫.fₒ, b.λ, wₛ)
	Σₒ += 𝒫.Σₒ
	return (μₒ, Σₒ, Sₒ, Sₒ′)
end

# ╔═╡ 940dc852-7f01-4add-9854-eac7bb3edfbd
function cross_covariance(μₚ, μₒ, wₛ, Sₒ, Sₒ′)
	return sum(w*(s - μₚ)*(s′ - μₒ)' for (w,s,s′) in zip(wₛ, Sₒ, Sₒ′))
end

# ╔═╡ d9d1a4c0-fc5b-11ea-0d6c-ab55c2b33d19
function ukf_update!(b::UnscentedKalmanFilter, 𝒫::POMDPᵘᵏᶠ, o, μₚ, Σₚ, wₛ)
	# Update observation model
	(μₒ, Σₒ, Sₒ, Sₒ′) = ukf_update_observation(b, 𝒫, μₚ, Σₚ, wₛ)
	
	# Calculate the cross covariance matrix
	Σₚₒ = cross_covariance(μₚ, μₒ, wₛ, Sₒ, Sₒ′)

	# Update belief
	K = Σₚₒ / Σₒ          # Kalman gain
	μᵦ′ = μₚ + K*(o - μₒ) # updated mean
	Σᵦ′ = Σₚ - K*Σₒ*K'    # updated covariance

	b.μᵦ = μᵦ′
	b.Σᵦ = Σᵦ′
end

# ╔═╡ 2f556310-fc5b-11ea-291e-2b953413c453
function update!(b::UnscentedKalmanFilter, 𝒫::POMDPᵘᵏᶠ, a, o)
	(μᵦ, Σᵦ, λ) = (b.μᵦ, b.Σᵦ, b.λ)
	wₛ = weights(μᵦ, λ)
	(μₚ, Σₚ) = ukf_predict(b, 𝒫, a, wₛ)
	ukf_update!(b, 𝒫, o, μₚ, Σₚ, wₛ)
end

# ╔═╡ a89bbc40-fb77-11ea-3a1b-7197afa0c9b0
function step(belief, 𝒫, s, a, o)
    a = rand(𝒜)
	s′ = rand(T(s, a))
	o = rand(O(s′, a))
    update!(belief, 𝒫, a, o)
    return (belief, s′, a, o)
end

# ╔═╡ c447b370-f7eb-11ea-1435-bd549afa0181
with_terminal() do
	Random.seed!(228)
	μᵦ = rand(𝒮)
	Σᵦ = Matrix(0.1I, 2, 2)
	global belief_kf = KalmanFilter(μᵦ, Σᵦ)
	global o = rand(𝒪)
	global s = copy(o)
	global a = missing

	Tₛ = Matrix(1.0I, 2, 2)
	Tₐ = Matrix(0.25I, 2, 2)
	Σₛ = [1/2 1/4; 1/4 1/2]

	Oₛ = Matrix(1.0I, 2, 2)
	Σₒ = [1 1/2; 1/2 2]

	global 𝒫ᵏᶠ = POMDPᵏᶠ(Tₛ, Tₐ, Oₛ, Σₛ, Σₒ)

	for i in 1:t
		(belief_kf, s, a, o) = step(belief_kf, 𝒫ᵏᶠ, s, a, o)
	end
	@show belief_kf.μᵦ
	@show belief_kf.Σᵦ
end

# ╔═╡ c9da23b2-fc49-11ea-16c5-776389af4472
plot_kalman_filter(belief_kf, s, t, a)

# ╔═╡ 70c44350-fc5d-11ea-3331-ef2cf5ab1326
md"## Visualization"

# ╔═╡ 83c7aa00-fc5d-11ea-3b99-e7290109a41b
md"""
Time: $(@bind t_ukf Slider(0:100, show_value=true, default=10))

λ: $(@bind λ_ukf Slider(0:0.1:4, show_value=true, default=2))
"""

# ╔═╡ 7d200530-fc5d-11ea-2ca9-8b81cebf13b0
with_terminal() do
	Random.seed!(228)
	μᵦ = rand(𝒮)
	Σᵦ = Matrix(0.1I, 2, 2)
	global belief_ukf = UnscentedKalmanFilter(μᵦ, Σᵦ, λ_ukf)
	global o_ukf = rand(𝒪)
	global s_ukf = copy(o_ukf)
	global a_ukf = missing

	Tₛ = Matrix(1.0I, 2, 2)
	Tₐ = Matrix(0.25I, 2, 2)
	Σₛ = [1/2 1/4; 1/4 1/2]

	Oₛ = Matrix(1.0I, 2, 2)
	Σₒ = [1 1/2; 1/2 2]

	fₜ = (s,a) -> Tₛ*s + Tₐ*a
	fₒ = s′ -> Oₛ*s′

	global 𝒫ᵘᵏᶠ = POMDPᵘᵏᶠ(fₜ, fₒ, Σₛ, Σₒ)

	for i in 1:t_ukf
		(belief_ukf, s_ukf, a_ukf, o_ukf) = step(belief_ukf,𝒫ᵘᵏᶠ,s_ukf,a_ukf,o_ukf)
	end
	@show belief_ukf.μᵦ
	@show belief_ukf.Σᵦ
end

# ╔═╡ 75b844b0-fc5d-11ea-0cef-4d5652f4cea2
plot_kalman_filter(belief_ukf, s_ukf, t_ukf, a_ukf)

# ╔═╡ 57a7b8c9-9205-42ca-acb7-2b11c665581d
belief_ukf

# ╔═╡ 4eb3bcc0-fc65-11ea-2485-e9211fb0685c
with_terminal() do
	@show s_ukf
end

# ╔═╡ 7d654260-fc9b-11ea-09b3-49b98bdf8aba
md"### Writing GIFs"

# ╔═╡ bd0736b0-0eb8-11eb-12bf-472687d2b830
md"Write GIF? $(@bind write_gif CheckBox())"

# ╔═╡ 68e0a4d0-fc99-11ea-182b-8560cb2714cf
if write_gif
	frames = Frames(MIME("image/png"), fps=2)
	for iter in 1:30
		global frames
        Random.seed!(228)
        μᵦ = rand(𝒮)
		Σᵦ = Matrix(0.1I, 2, 2)
		λ = 2.0
		belief2plot = UnscentedKalmanFilter(μᵦ, Σᵦ, λ)
        local o_ukf = rand(𝒪)
        local s_ukf = copy(o_ukf)
        local a_ukf = missing

		Tₛ = Matrix(1.0I, 2, 2)
		Tₐ = Matrix(0.25I, 2, 2)
		Σₛ = [1/2 1/4; 1/4 1/2]

		Oₛ = Matrix(1.0I, 2, 2)
		Σₒ = [1 1/2; 1/2 2]
	
		fₜ = (s,a) -> Tₛ*s + Tₐ*a
		fₒ = s′ -> Oₛ*s′
	
		local 𝒫ᵘᵏᶠ = POMDPᵘᵏᶠ(fₜ, fₒ, Σₛ, Σₒ)

		if iter == 1
			# X initial frames stationary
			[push!(frames,
				   plot_kalman_filter(belief2plot, s_ukf, iter, [0,0])) for _ in 1:3]
		end
        for i in 1:iter
            (belief2plot, s_ukf, a_ukf, o_ukf) =
				step(belief2plot, 𝒫ᵘᵏᶠ, s_ukf, a_ukf, o_ukf)
        end
		push!(frames, plot_kalman_filter(belief2plot, s_ukf, iter, a_ukf))
	end
	write("../gif/kalman_filter.gif", frames)
end

# ╔═╡ bd0d0580-fdce-4ba0-9d33-c32b46313f13
function show_gif()
	if isfile("../gif/kalman_filter.gif") || write_gif
		LocalResource("../gif/kalman_filter.gif")
	end
end

# ╔═╡ e740513d-7377-45c9-9c28-72a51ef663d5
begin
	import Logging
	stream = IOBuffer(UInt8[])
	logger = Logging.SimpleLogger(stream, Logging.Error)
	gif = Logging.with_logger(logger) do
	   show_gif()
	end
end

# ╔═╡ a51ea4d9-15b0-4a0b-be8c-b920c07502cf
gif

# ╔═╡ d3fbb360-fc51-11ea-1522-3d04a8f3fb5f
md"# Testing"

# ╔═╡ 29206e50-fc3c-11ea-2f8d-8b876eab5bc4
with_terminal() do
	_s  = [-0.75, 1.0]
	_s′ = [-0.25, 0.5]
	_a  = _s′ - _s
	_o = [-0.585, 0.731]

	Tₛ = Matrix(1.0I, 2, 2)
	Tₐ = Matrix(1.0I, 2, 2)
	Σₛ = 0.1*[1.0 0.5; 0.5 1.0]

	Oₛ = Matrix(1.0I, 2, 2)
	Σₒ = 0.05*[1.0 -0.5; -0.5 1.5]

	μᵦ = copy(_s)
	Σᵦ = Matrix(0.1I, 2, 2)
	kf = KalmanFilter(μᵦ, Σᵦ)

	𝒫ᵏᶠ = POMDPᵏᶠ(Tₛ, Tₐ, Oₛ, Σₛ, Σₒ)

	update!(kf, 𝒫ᵏᶠ, _a, _o)
	@show isapprox(norm(kf.μᵦ - [-0.4889, 0.6223]), 0.0, atol=1e-4)
	@show isapprox(norm(kf.Σᵦ - [0.0367 -0.0115; -0.0115 0.0505]), 0.0, atol=1e-4)
end

# ╔═╡ d83c01c0-fb78-11ea-0543-d3a0fdcbadab
function test_filter(belief, s)
    μ_b = belief.μᵦ
    Σ_b = belief.Σᵦ
    belief_error = abs.(μ_b - s)
    @test (μ_b-3σ_b .≤ s .≤ μ_b+3σ_b) || belief_error .≤ 1.0
end

# ╔═╡ 4eafc8d0-0eb3-11eb-2855-b121f3455d40
md"""
---
"""

# ╔═╡ efa90700-0914-11eb-38c3-0783ca0cf6e3
TableOfContents(title="Kalman Filtering")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Reel = "71555da5-176e-5e73-a222-aebc6c6e4f2f"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
Distributions = "~0.25.24"
Plots = "~1.23.5"
PlutoUI = "~0.7.18"
Reel = "~1.3.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0ec322186e078db08ea3e7da5b8b2885c099b393"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.0"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "72dcda9e19f88d09bf21b5f9507a0bb430bce2aa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.24"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "f0c6489b12d28fb4c2103073ec7452f3423bd308"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.1"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "6193c3815f13ba1b78a51ce391db8be016ae9214"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.4"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c8b8775b2f242c80ea85c83714c64ecfa3c53355"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "7dc03c2b145168f5854085a16d054429d612b637"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.23.5"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "57312c7ecad39566319ccf5aa717a20788eb8c1f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.18"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reel]]
deps = ["FFMPEG"]
git-tree-sha1 = "0f600c38899603d9667111176eb6b5b33c80781e"
uuid = "71555da5-176e-5e73-a222-aebc6c6e4f2f"
version = "1.3.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─2cbec03e-fb77-11ea-09a2-634fac25a12a
# ╠═740dc710-fbaf-11ea-2062-7f44056cbd12
# ╟─e740513d-7377-45c9-9c28-72a51ef663d5
# ╟─29e2d71e-fc40-11ea-0c55-f929ddc20588
# ╟─09fc2050-fc46-11ea-2bc4-257edf069912
# ╠═419cda50-fc3b-11ea-2ecf-b521f3f44d38
# ╟─037674ae-fc41-11ea-025c-8510cc72063b
# ╠═caef0200-fc3c-11ea-06e8-09595e3e00f3
# ╟─2334f0c0-fc40-11ea-1c88-67467bb5651f
# ╠═597bd862-fc3b-11ea-2c14-497f8746c4f3
# ╟─3b261740-fc40-11ea-2253-012e10b5e6e6
# ╟─80ae2940-fc42-11ea-3db3-bdf6de06f6df
# ╠═ca4f7bc0-fc3e-11ea-0588-d9468558d025
# ╟─469be5f0-fc40-11ea-2a7a-23c9356b4b44
# ╟─2318aed2-fc43-11ea-24e6-19c5342b76a2
# ╠═f6437792-fc3e-11ea-2941-2ba90b95ecee
# ╟─038c5510-f8bc-11ea-0fc5-7d765d868496
# ╠═cbeb9ffc-3267-4d0b-a395-f10844d60dad
# ╠═608a4850-f7e8-11ea-2fca-af35a2f0456b
# ╟─4099e950-fb77-11ea-23b7-6d1f7b47c07e
# ╠═7901f281-a3ff-477c-b292-6121ce12af32
# ╠═a89bbc40-fb77-11ea-3a1b-7197afa0c9b0
# ╠═c447b370-f7eb-11ea-1435-bd549afa0181
# ╠═707e9b30-f8a1-11ea-0a6c-ad6756d07bbc
# ╠═c9da23b2-fc49-11ea-16c5-776389af4472
# ╟─de37b64a-1e98-416f-99b6-6c746307df93
# ╠═83560088-c06d-4f9d-976e-63732ba206e6
# ╠═7179e070-fc99-11ea-1f02-511f2215c6a8
# ╠═4726f4a0-fc50-11ea-12f5-7f19d21d9bcc
# ╟─a2252160-fc5a-11ea-1c52-4717e186e8ff
# ╟─98bc1fe0-0eb4-11eb-016d-1de4734348d3
# ╟─5011a010-fc5a-11ea-22b8-df368e66c6cc
# ╟─9a55a8f0-fc5b-11ea-15c6-abb241ea8770
# ╠═7dfa2370-fc5b-11ea-3d5d-d54349446b89
# ╠═6feab390-fc5a-11ea-1367-c5f353fadbc7
# ╟─48e7cb90-fc5d-11ea-0c29-c32610e59625
# ╠═545eda20-fc5a-11ea-1e32-bfe408c99b35
# ╟─d0253f60-0eba-11eb-0283-e5b6e8c14ffe
# ╠═b1773e60-0eba-11eb-36e6-3f0e9e7cc3b6
# ╠═999a5c19-f398-4769-9889-99f0597d3511
# ╠═3e0aab30-0eb8-11eb-1e6c-81a3d89d1cfb
# ╠═84e1f980-0eba-11eb-0c52-878cc6e04534
# ╟─00c94de0-fc74-11ea-0b52-a9b2938c5117
# ╠═e1d59150-fc73-11ea-2b67-ef871a9d12b5
# ╟─5564aa00-fc5d-11ea-2a66-b5f9edef3f03
# ╠═2f556310-fc5b-11ea-291e-2b953413c453
# ╟─21998107-2eff-491c-9c23-b7f0d5a87c9a
# ╟─4f04e39e-fc5d-11ea-0b22-85563521ec7f
# ╟─9746c3d0-fc72-11ea-2d9a-5dbe53753813
# ╠═f6aab420-fc5a-11ea-122f-d356a54e953c
# ╟─5ddbdeb0-fc5d-11ea-3600-21920d6bf4a2
# ╠═d1fc7c70-fc5b-11ea-3c17-7f3e5bc58b44
# ╟─63dd3160-fc5d-11ea-223d-3119cff7630d
# ╠═d9d1a4c0-fc5b-11ea-0d6c-ab55c2b33d19
# ╠═befe14ee-ecd5-4f4e-a1af-40fdd0a61da8
# ╠═940dc852-7f01-4add-9854-eac7bb3edfbd
# ╟─70c44350-fc5d-11ea-3331-ef2cf5ab1326
# ╠═7d200530-fc5d-11ea-2ca9-8b81cebf13b0
# ╟─83c7aa00-fc5d-11ea-3b99-e7290109a41b
# ╠═75b844b0-fc5d-11ea-0cef-4d5652f4cea2
# ╠═57a7b8c9-9205-42ca-acb7-2b11c665581d
# ╠═4eb3bcc0-fc65-11ea-2485-e9211fb0685c
# ╟─7d654260-fc9b-11ea-09b3-49b98bdf8aba
# ╠═c52df260-fc99-11ea-00a2-3f21b9c40f3b
# ╟─bd0736b0-0eb8-11eb-12bf-472687d2b830
# ╠═a51ea4d9-15b0-4a0b-be8c-b920c07502cf
# ╠═68e0a4d0-fc99-11ea-182b-8560cb2714cf
# ╟─bd0d0580-fdce-4ba0-9d33-c32b46313f13
# ╟─d3fbb360-fc51-11ea-1522-3d04a8f3fb5f
# ╠═5bd86070-194f-11eb-21f9-f57ce27aa5d0
# ╠═29206e50-fc3c-11ea-2f8d-8b876eab5bc4
# ╠═d83c01c0-fb78-11ea-0543-d3a0fdcbadab
# ╟─4eafc8d0-0eb3-11eb-2855-b121f3455d40
# ╠═efa90700-0914-11eb-38c3-0783ca0cf6e3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
