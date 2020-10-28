### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 04c54992-fc46-11ea-39d5-d18c4392b483
try using AddPackage catch; using Pkg; Pkg.add("AddPackage") end

# ╔═╡ 740dc710-fbaf-11ea-2062-7f44056cbd12
@add using Distributions, LinearAlgebra

# ╔═╡ 3cafb210-f89e-11ea-0cf2-bdf819224cc9
@add using PlutoUI, Random

# ╔═╡ 8439ae70-fc99-11ea-2edb-51fc16909fa9
@add using PyPlot; PyPlot.svg(true)

# ╔═╡ c52df260-fc99-11ea-00a2-3f21b9c40f3b
@add using Reel

# ╔═╡ 85830e20-fb77-11ea-1e9f-d3651f6fe718
@add using Suppressor

# ╔═╡ 5bd86070-194f-11eb-21f9-f57ce27aa5d0
using Test

# ╔═╡ 48e32590-fc3a-11ea-3ff0-a7827e9847f1
include("section_counters.jl")

# ╔═╡ 2cbec03e-fb77-11ea-09a2-634fac25a12a
md"""
# Kalman filters

For code, see [StateEstimation.jl](https://github.com/mossr/StateEstimation.jl)
"""

# ╔═╡ 29e2d71e-fc40-11ea-0c55-f929ddc20588
md"""
## Standard Kalman filter
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
struct POMDPₘ Tₛ; Tₐ; Oₛ; Σₛ; Σₒ end

# ╔═╡ 2334f0c0-fc40-11ea-1c88-67467bb5651f
md"### Belief update"

# ╔═╡ 3b261740-fc40-11ea-2253-012e10b5e6e6
md"""
#### Kalman prediction
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
function kalman_predict(b::KalmanFilter, 𝒫::POMDPₘ, a)
	(μᵦ, Σᵦ) = (b.μᵦ, b.Σᵦ)
	(Tₛ, Tₐ, Σₛ) = (𝒫.Tₛ, 𝒫.Tₐ, 𝒫.Σₛ)

	μₚ = Tₛ*μᵦ + Tₐ*a   # predicted mean
	Σₚ = Tₛ*Σᵦ*Tₛ' + Σₛ # predicted covariance

	return (μₚ, Σₚ)
end

# ╔═╡ 469be5f0-fc40-11ea-2a7a-23c9356b4b44
md"""
#### Kalman update
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
function kalman_update!(b::KalmanFilter, 𝒫::POMDPₘ, o, μₚ, Σₚ)		
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
function update!(b::KalmanFilter, 𝒫::POMDPₘ, a, o)
	(μₚ, Σₚ) = kalman_predict(b, 𝒫, a)
	kalman_update!(b, 𝒫, o, μₚ, Σₚ)
end

# ╔═╡ 038c5510-f8bc-11ea-0fc5-7d765d868496
md"""
## POMDP definition
Agent randomly walking in a $10\times10$ continuous 2D environment.
"""

# ╔═╡ 5d9e4bf0-f7e8-11ea-23d8-2dbd72e46ce6
struct POMDP 𝒮; 𝒜; 𝒪; T; O end

# ╔═╡ 608a4850-f7e8-11ea-2fca-af35a2f0456b
begin
    𝒮 = Product(Uniform.([-10, -10], [10, 10]))
	𝒮ₘᵢₙ = minimum.(support.(𝒮.v))
	𝒮ₘₐₓ = maximum.(support.(𝒮.v))

	𝒜 = MvNormal([0, 0], [1 0; 0 1])
	𝒪 = MvNormal([0, 0], [1 0; 0 1])
	
	transition = (s,a) -> clamp.(s .+ a, 𝒮ₘᵢₙ, 𝒮ₘₐₓ)
    T = (s,a) -> MvNormal(transition(s,a), I*abs.(a))

	observation = (s′,a) -> MvNormal(𝒪.μ + s′, 𝒪.Σ*abs.(a))
    O = (a,s′,o) -> pdf(observation(s′,a), o)
    𝒫 = POMDP(𝒮, 𝒜, 𝒪, T, O)
end;

# ╔═╡ 4099e950-fb77-11ea-23b7-6d1f7b47c07e
md"## Simulation and testing"

# ╔═╡ 707e9b30-f8a1-11ea-0a6c-ad6756d07bbc
md"""
$(@bind t Slider(0:2000, show_value=true, default=10))
"""

# ╔═╡ 4726f4a0-fc50-11ea-12f5-7f19d21d9bcc
function plot_covariance(P, xdomain, ydomain; cmap="Blues", alpha=1)
    varX = range(xdomain[1], stop=xdomain[2], length=100)
    varY = range(ydomain[1], stop=ydomain[2], length=100)

    Z = [pdf(P, [x,y]) for y in varY, x in varX] # Note: reverse in X, Y.
    contour(Z, extent=[xdomain[1], xdomain[2], ydomain[1], ydomain[2]],
		    cmap=cmap, alpha=alpha)
end

# ╔═╡ a2252160-fc5a-11ea-1c52-4717e186e8ff
md"""
## Extended Kalman filter
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
### Jacobian
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
## Unscented Kalman filter 🧼
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
struct POMDPᵤ fₜ; fₒ; Σₛ; Σₒ end

# ╔═╡ 6feab390-fc5a-11ea-1367-c5f353fadbc7
mutable struct UnscentedKalmanFilter
	μᵦ # mean vector
	Σᵦ # covariance matrix
	λ  # point-spread parameter
end

# ╔═╡ 48e7cb90-fc5d-11ea-0c29-c32610e59625
md"""
#### Sigma point samples
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

# ╔═╡ 7179e070-fc99-11ea-1f02-511f2215c6a8
function plot_kalman_filter(belief, true_state, iteration, action)
    clf()
	λ = 2	
	S = sigma_points(belief.μᵦ, belief.Σᵦ, λ)
	for s in S
		plot(s..., "c.") # sigma points
	end
	
	P = MvNormal(belief.μᵦ, Matrix(Hermitian(belief.Σᵦ))) # numerical stability
	xdomain, ydomain = (-10, 10), (-10, 10)
	plot_covariance(P, xdomain, ydomain) # covariance contours
	
	plot(true_state..., "ro") # true state
    xlim([-10, 10])
    ylim([-10, 10])
    title("iteration=$iteration, action=$(round.(action, digits=4))")
    gcf()
end

# ╔═╡ 84e1f980-0eba-11eb-0c52-878cc6e04534
function plot_sigma_points(μ, Σ, λ)
    clf()
	S = sigma_points(μ, Σ, λ)
	for s in S
		plot(s..., "c.") # sigma points
	end
	axis("equal")
	xlim([-10, 10])
	ylim([-10, 10])
	gcf()
end

# ╔═╡ 3e0aab30-0eb8-11eb-1e6c-81a3d89d1cfb
sigma_points([0.0, 0.0], [1 0; 0 1], 2) # example

# ╔═╡ d0253f60-0eba-11eb-0283-e5b6e8c14ffe
md"""
Σ₁₁ = $(@bind Σ₁₁ Slider(1:10, default=1, show_value=true)) ... 
Σ₁₂ = $(@bind Σ₁₂ Slider(0:10, default=0, show_value=true))

Σ₂₁ = $(@bind Σ₂₁ Slider(0:10, default=0, show_value=true)) ...
Σ₂₂ = $(@bind Σ₂₂ Slider(1:10, default=1, show_value=true)) 

λ = $(@bind λₑₓ Slider(-1:10, default=2, show_value=true))
"""

# ╔═╡ b1773e60-0eba-11eb-36e6-3f0e9e7cc3b6
plot_sigma_points([0.0, 0.0], [Σ₁₁ Σ₁₂; Σ₂₁ Σ₂₂], λₑₓ)

# ╔═╡ 00c94de0-fc74-11ea-0b52-a9b2938c5117
md"""
#### Weights
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

# ╔═╡ 4f04e39e-fc5d-11ea-0b22-85563521ec7f
md"#### Unscented transform"

# ╔═╡ 9746c3d0-fc72-11ea-2d9a-5dbe53753813
md"""
$$\begin{align}
𝛍^′ &= \sum_i w_i 𝐬_i\\
𝚺^′ &= \sum_i w_i (𝐬_i - 𝛍^′)(𝐬_i - 𝛍^′)^\top
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

# ╔═╡ 5564aa00-fc5d-11ea-2a66-b5f9edef3f03
md"""
### Belief update
"""

# ╔═╡ 5ddbdeb0-fc5d-11ea-3600-21920d6bf4a2
md"#### Unscented prediction"

# ╔═╡ d1fc7c70-fc5b-11ea-3c17-7f3e5bc58b44
function unscented_predict(b::UnscentedKalmanFilter, 𝒫::POMDPᵤ, a, wₛ)
	(μᵦ, Σᵦ, λ) = (b.μᵦ, b.Σᵦ, b.λ)
	(μₚ, Σₚ, _, _) = unscented_transform(μᵦ, Σᵦ, s->𝒫.fₜ(s,a), λ, wₛ)
	Σₚ += 𝒫.Σₛ
	return (μₚ, Σₚ)
end

# ╔═╡ 63dd3160-fc5d-11ea-223d-3119cff7630d
md"""
#### Unscented update
$$\begin{align}
𝛍^′ &= \sum_i w_i 𝐟(𝐬_i)\\
𝚺^′ &= \sum_i w_i\bigl(𝐟(𝐬_i) - 𝛍^′\bigr)\bigl(𝐟(𝐬_i) - 𝛍^′\bigr)^\top
\end{align}$$
"""

# ╔═╡ d9d1a4c0-fc5b-11ea-0d6c-ab55c2b33d19
function unscented_update!(b::UnscentedKalmanFilter, 𝒫::POMDPᵤ, o, μₚ, Σₚ, wₛ)
	(μᵦ, Σᵦ, λ) = (b.μᵦ, b.Σᵦ, b.λ)
	(μₒ, Σₒ, Sₒ, Sₒ′) = unscented_transform(μₚ, Σₚ, 𝒫.fₒ, λ, wₛ)

	Σₒ += 𝒫.Σₒ
	Σₚₒ = sum(w*(s - μₚ)*(s′ - μₚ)' for (w,s,s′) in zip(wₛ, Sₒ, Sₒ′))
	K = Σₚₒ / Σₒ
	μᵦ′ = μₚ + K*(o - μₒ)
	Σᵦ′ = Σₚ - K*Σₒ*K'

	b.μᵦ = μᵦ′
	b.Σᵦ = Σᵦ′
end

# ╔═╡ 2f556310-fc5b-11ea-291e-2b953413c453
function update!(b::UnscentedKalmanFilter, 𝒫::POMDPᵤ, a, o)
	(μᵦ, Σᵦ, λ) = (b.μᵦ, b.Σᵦ, b.λ)
	wₛ = weights(μᵦ, λ)
	(μₚ, Σₚ) = unscented_predict(b, 𝒫, a, wₛ)
	unscented_update!(b, 𝒫, o, μₚ, Σₚ, wₛ)
end

# ╔═╡ a89bbc40-fb77-11ea-3a1b-7197afa0c9b0
function step(belief, 𝒫, s, a, o)
    a = rand(𝒜)
	s = transition(s, a)
	o = rand(observation(s, a))
    update!(belief, 𝒫, a, o)
    return (belief, s, a, o)
end

# ╔═╡ 70c44350-fc5d-11ea-3331-ef2cf5ab1326
md"### Visualization"

# ╔═╡ 83c7aa00-fc5d-11ea-3b99-e7290109a41b
md"""
$(@bind t_ukf Slider(0:2000, show_value=true, default=10))
"""

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
        o_ukf = rand(𝒪)
        s_ukf = copy(o_ukf)
        a_ukf = missing

		Tₛ = Matrix(1.0I, 2, 2)
		Tₐ = Matrix(1.0I, 2, 2)
		Σₛ = [1.0 0.0; 0.0 0.5]

		Oₛ = Matrix(1.0I, 2, 2)
		Σₒ = [1.0 0.0; 0.0 2.0]
	
		fₜ = (s,a) -> Tₛ*s + Tₐ*a
		fₒ = s′ -> Oₛ*s′
	
		𝒫ᵤ = POMDPᵤ(fₜ, fₒ, Σₛ, Σₒ)

		if iter == 1
			# X initial frames stationary
			[push!(frames,
				   plot_kalman_filter(belief2plot, s_ukf, iter, [0,0])) for _ in 1:3]
		end
        for i in 1:iter
            (belief2plot, s_ukf, a_ukf, o_ukf) =
				step(belief2plot, 𝒫ᵤ, s_ukf, a_ukf, o_ukf)
        end
		push!(frames, plot_kalman_filter(belief2plot, s_ukf, iter, a_ukf))
	end
	write("kalman_filter.gif", frames)
end

# ╔═╡ 802c5e80-f8b2-11ea-310f-6fdbcacb73d0
md"## Helper code"

# ╔═╡ 67ebdf80-f8b2-11ea-2630-d54abc89ad2b
function with_terminal(f)
    local spam_out, spam_err
    @color_output false begin
        spam_out = @capture_out begin
            spam_err = @capture_err begin
                f()
            end
        end
    end
    spam_out, spam_err

    HTML("""
        <style>
        div.vintage_terminal {

        }
        div.vintage_terminal pre {
            color: #ddd;
            background-color: #333;
            border: 5px solid gray;
            font-size: .75rem;
        }

        </style>
    <div class="vintage_terminal">
        <pre>$(Markdown.htmlesc(spam_out))</pre>
    </div>
    """)
end

# ╔═╡ c447b370-f7eb-11ea-1435-bd549afa0181
with_terminal() do
	Random.seed!(228)
	μᵦ = rand(𝒮)
	Σᵦ = Matrix(0.1I, 2, 2)
	global belief = KalmanFilter(μᵦ, Σᵦ)
	global o = rand(𝒪)
	global s = copy(o)
	global a = missing

	Tₛ = Matrix(1.0I, 2, 2)
	Tₐ = Matrix(1.0I, 2, 2)
	Σₛ = [0.5 0; 0 0.5]

	Oₛ = Matrix(1.0I, 2, 2)
	Σₒ = [1 0; 0 2]

	global 𝒫ₘ = POMDPₘ(Tₛ, Tₐ, Oₛ, Σₛ, Σₒ)

	for i in 1:t
		(belief, s, a, o) = step(belief, 𝒫ₘ, s, a, o)
	end
	@show belief.μᵦ
	@show belief.Σᵦ
end

# ╔═╡ c9da23b2-fc49-11ea-16c5-776389af4472
plot_kalman_filter(belief, s, t, a)

# ╔═╡ 7d200530-fc5d-11ea-2ca9-8b81cebf13b0
with_terminal() do
	Random.seed!(228)
	μᵦ = rand(𝒮)
	Σᵦ = Matrix(0.1I, 2, 2)
	λ = 2.0
	global belief_ukf = UnscentedKalmanFilter(μᵦ, Σᵦ, λ)
	global o_ukf = rand(𝒪)
	global s_ukf = copy(o_ukf)
	global a_ukf = missing

	Tₛ = Matrix(1.0I, 2, 2)
	Tₐ = Matrix(1.0I, 2, 2)
	Σₛ = [0.5 0.0; 0.0 0.5]

	Oₛ = Matrix(1.0I, 2, 2)
	Σₒ = [1.0 0.0; 0.0 2.0]

	fₜ = (s,a) -> Tₛ*s + Tₐ*a
	fₒ = s′ -> Oₛ*s′

	global 𝒫ᵤ = POMDPᵤ(fₜ, fₒ, Σₛ, Σₒ)

	for i in 1:t_ukf
		(belief_ukf, s_ukf, a_ukf, o_ukf) = step(belief_ukf, 𝒫ᵤ, s_ukf, a_ukf, o_ukf)
	end
	@show belief_ukf.μᵦ
	@show belief_ukf.Σᵦ
end

# ╔═╡ 75b844b0-fc5d-11ea-0cef-4d5652f4cea2
plot_kalman_filter(belief_ukf, s_ukf, t_ukf, a_ukf)

# ╔═╡ 4eb3bcc0-fc65-11ea-2485-e9211fb0685c
with_terminal() do
	@show s_ukf
end

# ╔═╡ f8ab7310-fc8f-11ea-0af1-f71a83f10460
md"LaTeX-style fonts in `PyPlot`."

# ╔═╡ dfad65e0-fc8e-11ea-2688-2b5004a3f834
begin
	# LaTeX-style fonts in PyPlot
	matplotlib.rc("font", family=["serif"])
	matplotlib.rc("font", serif=["Helvetica"])
	matplotlib.rc("text", usetex=true)
end

# ╔═╡ d3fbb360-fc51-11ea-1522-3d04a8f3fb5f
md"## Testing"

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

	𝒫ₘ = POMDPₘ(Tₛ, Tₐ, Oₛ, Σₛ, Σₒ)

	update!(kf, 𝒫ₘ, _a, _o)
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
try PlutoUI.TableOfContents("Kalman Filtering"); catch end

# ╔═╡ Cell order:
# ╟─2cbec03e-fb77-11ea-09a2-634fac25a12a
# ╠═04c54992-fc46-11ea-39d5-d18c4392b483
# ╠═740dc710-fbaf-11ea-2062-7f44056cbd12
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
# ╠═5d9e4bf0-f7e8-11ea-23d8-2dbd72e46ce6
# ╠═608a4850-f7e8-11ea-2fca-af35a2f0456b
# ╟─4099e950-fb77-11ea-23b7-6d1f7b47c07e
# ╠═3cafb210-f89e-11ea-0cf2-bdf819224cc9
# ╠═a89bbc40-fb77-11ea-3a1b-7197afa0c9b0
# ╠═c447b370-f7eb-11ea-1435-bd549afa0181
# ╟─707e9b30-f8a1-11ea-0a6c-ad6756d07bbc
# ╠═c9da23b2-fc49-11ea-16c5-776389af4472
# ╠═8439ae70-fc99-11ea-2edb-51fc16909fa9
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
# ╠═84e1f980-0eba-11eb-0c52-878cc6e04534
# ╠═3e0aab30-0eb8-11eb-1e6c-81a3d89d1cfb
# ╟─d0253f60-0eba-11eb-0283-e5b6e8c14ffe
# ╠═b1773e60-0eba-11eb-36e6-3f0e9e7cc3b6
# ╟─00c94de0-fc74-11ea-0b52-a9b2938c5117
# ╠═e1d59150-fc73-11ea-2b67-ef871a9d12b5
# ╟─4f04e39e-fc5d-11ea-0b22-85563521ec7f
# ╟─9746c3d0-fc72-11ea-2d9a-5dbe53753813
# ╠═f6aab420-fc5a-11ea-122f-d356a54e953c
# ╟─5564aa00-fc5d-11ea-2a66-b5f9edef3f03
# ╠═2f556310-fc5b-11ea-291e-2b953413c453
# ╟─5ddbdeb0-fc5d-11ea-3600-21920d6bf4a2
# ╠═d1fc7c70-fc5b-11ea-3c17-7f3e5bc58b44
# ╟─63dd3160-fc5d-11ea-223d-3119cff7630d
# ╠═d9d1a4c0-fc5b-11ea-0d6c-ab55c2b33d19
# ╟─70c44350-fc5d-11ea-3331-ef2cf5ab1326
# ╠═7d200530-fc5d-11ea-2ca9-8b81cebf13b0
# ╟─83c7aa00-fc5d-11ea-3b99-e7290109a41b
# ╠═75b844b0-fc5d-11ea-0cef-4d5652f4cea2
# ╠═4eb3bcc0-fc65-11ea-2485-e9211fb0685c
# ╟─7d654260-fc9b-11ea-09b3-49b98bdf8aba
# ╠═c52df260-fc99-11ea-00a2-3f21b9c40f3b
# ╟─bd0736b0-0eb8-11eb-12bf-472687d2b830
# ╠═68e0a4d0-fc99-11ea-182b-8560cb2714cf
# ╟─802c5e80-f8b2-11ea-310f-6fdbcacb73d0
# ╠═85830e20-fb77-11ea-1e9f-d3651f6fe718
# ╟─67ebdf80-f8b2-11ea-2630-d54abc89ad2b
# ╠═48e32590-fc3a-11ea-3ff0-a7827e9847f1
# ╟─f8ab7310-fc8f-11ea-0af1-f71a83f10460
# ╠═dfad65e0-fc8e-11ea-2688-2b5004a3f834
# ╟─d3fbb360-fc51-11ea-1522-3d04a8f3fb5f
# ╠═5bd86070-194f-11eb-21f9-f57ce27aa5d0
# ╠═29206e50-fc3c-11ea-2f8d-8b876eab5bc4
# ╠═d83c01c0-fb78-11ea-0543-d3a0fdcbadab
# ╟─4eafc8d0-0eb3-11eb-2855-b121f3455d40
# ╠═efa90700-0914-11eb-38c3-0783ca0cf6e3
