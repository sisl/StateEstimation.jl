### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ dbeae580-f681-11ea-0605-0da1c23cb448
using Distributions

# ╔═╡ 0e25d640-f687-11ea-1d5d-f73947ddffaa
using LinearAlgebra

# ╔═╡ 3abbe900-f688-11ea-2727-6bc22eb364a8
using Latexify

# ╔═╡ 38451640-f680-11ea-1931-a7d267fbcaf9
md"""
# Beliefs: State Estimation
Section for AA228/CS238: Decision Making Under Uncertainty, Autumn 2020

— Robert Moss (MSCS), TA
"""

# ╔═╡ 5a1edbc0-f680-11ea-08c5-49c22c4b4b97
md"""
- Start with beamer slides
- Example 19.1: landmark belief initialization
"""

# ╔═╡ f31ffd10-f683-11ea-0482-793bf05deee0
md"## Belief Initialization"

# ╔═╡ 010caa10-f682-11ea-3deb-b728b1d405d7
md"**TODO**: Get \"zoomed\" TikZ picture."

# ╔═╡ 27823ed0-f682-11ea-167b-a1523627e265
md"**TODO**: Describe problem"

# ╔═╡ bec045a0-f680-11ea-16da-c77da3ba35e6
md"""
$$\begin{gather}
\hat{r} \sim \mathcal{N}(r, \nu_r) \qquad\qquad \hat{\theta} \sim \mathcal{N}(\theta, \nu_\theta) \qquad\qquad \hat{\phi} \sim \mathcal{U}(0, 2\pi)\\
\hat{x} \leftarrow x + \hat{r}\cos\hat{\phi} \qquad\qquad \hat{y} \leftarrow y + \hat{r}\sin\hat{\phi} \qquad\qquad \hat{\psi} \leftarrow \hat{phi} - \hat{\theta} + \pi
\end{gather}$$
"""

# ╔═╡ e04f5110-f681-11ea-1fff-470db0d13073
md"**TODO**: Get \"circular\" TikZ picture."

# ╔═╡ d86e68f0-f681-11ea-0577-a14d7ac9c74b
md"**TODO**: Defined r, θ, φ distributions"

# ╔═╡ d33c4d20-f681-11ea-0602-093a0453cfa7
md"**TODO**: Define x, y, ψ update functions"

# ╔═╡ 62ae1a60-f682-11ea-0db8-31880fdb771e
struct POMDP
	γ # discount factor
	𝒮 # state space
	𝒜 # action space
	𝒪 # observation space
	T # transition function
	R # reward function
	O # observation function
end

# ╔═╡ c9739230-f681-11ea-0dc5-071d7a96aa1f
md"**TODO**: Belief `update` function with crying baby example" 

# ╔═╡ 27caf5b0-f684-11ea-3592-6303dc526cd9
function update(b::Vector{Float64}, 𝒫, a, o)
	𝒮, T, O = 𝒫.𝒮 ,𝒫.T, 𝒫.O
	b′ = similar(b)
	for (i′, s′) in enumerate(𝒮)
		po = O(a, s′, o)
		b′[i′] = po * sum(T(s, a, s′) * b[i] for (i, s) in enumerate(𝒮))
	end
	if sum(b′) ≈ 0.0
		fill!(b′, 1)
	end
	return normalize!(b′, 1)
end

# ╔═╡ e95b60d0-f683-11ea-3684-cff946175c9a
md"## Crying Baby Problem"

# ╔═╡ 1ed1b52e-f683-11ea-1030-0ff004c644fb
md"**TODO**: Crying baby example from POMDPs.jl notebook (`QuickPOMDP`?)"

# ╔═╡ 5f50d3ae-f684-11ea-326a-173e645a3f15
begin
	@enum State hungry sated
	@enum Action feed sing ignore
	@enum crying quiet
end

# ╔═╡ a25ab130-f684-11ea-04a2-b17aef6e9101
function T(s, a, s′)
	if a == feed
		return s′ == hungry ? 0 : 1
	elseif s == hungry && (a == sing || a == ignore)
		return s′ == hungry ? 1 : 0
	elseif s == sated && (a == sing || a == ignore)
		return s′ == hungry ? 0.1 : 0.9
	end
end

# ╔═╡ 9725edb2-f685-11ea-3282-6fca02b29983
function R(s, a)
	return (s == hungry ? -10 : 0) +
	       (a == feed ? -5 : 0) +
	       (a == sing && s == sated ? +5 : 0) +
	       (a == sing && s == hungry ? -2 : 0)
end

# ╔═╡ c20cbf42-f685-11ea-3854-c7fff3b94bdb
function O(a, s′, o)
	if a == sing # perfect observation
		if s′ == hungry
			return o == crying ? 1 : 0
		elseif s′ == sated
			return o == crying ? 0 : 1
		end
	elseif s′ == hungry
		o == crying ? 0.8 : 0.2
	elseif s′ == sated
		o == crying ? 0.1 : 0.9
	end
end

# ╔═╡ 513846a0-f684-11ea-2766-310f8561add1
𝒫 = POMDP(0.9,
	      (hungry, sated),
	      (feed, sing, ignore),
	      (crying, quiet),
	      T,
	      R,
	      O)

# ╔═╡ e9db37d0-f686-11ea-34bd-4d2e72564f1d
b₀ = [0.5, 0.5]

# ╔═╡ ef6dea30-f686-11ea-0b4e-27cf68080e2a
b₁ = update(b₀, 𝒫, ignore, crying)

# ╔═╡ 465c3f92-f687-11ea-3ac9-d927e67cc870
b₂ = update(b₁, 𝒫, feed, quiet)

# ╔═╡ 5a476e30-f687-11ea-3740-894fb7053372
b₃ = update(b₂, 𝒫, ignore, quiet)

# ╔═╡ 4e01a550-f687-11ea-0e3a-1739be23fec3
b₄ = update(b₃, 𝒫, ignore, quiet)

# ╔═╡ 6da529e0-f687-11ea-39b6-53aee839f504
b₅ = update(b₄, 𝒫, ignore, crying)

# ╔═╡ b1a9bb10-f687-11ea-2f16-736f7b124268
md"## Kalman Filtering"

# ╔═╡ a61e0220-f681-11ea-0d29-ff5c63073bfd
md"**TODO**: Kalman filter?"

# ╔═╡ 9fffffb0-f681-11ea-0d36-2deb4d62ab33
md"**TODO**: Look Ch. 19 at exercises."

# ╔═╡ 5c0e2142-f683-11ea-0a77-bbe7df75a4e0
md"**TODO**: Look Ch. 20 at exercises."

# ╔═╡ a593dbd0-f687-11ea-3270-69f52f78ec87
md"## Particle Filtering"

# ╔═╡ a94d9730-f681-11ea-17fb-f523a92590a2
md"**TODO**: Particle filter from POMDPs.jl notebook"

# ╔═╡ 5db20d40-f683-11ea-34f3-9be2d2d45c00
md"## Exact Belief State Planning"

# ╔═╡ 94fbb780-f681-11ea-0bb5-af29f2c8c0b2
md"### Conditional Plans"

# ╔═╡ 81974510-f681-11ea-0175-b3546e1192b8
md"### Alpha Vectors"

# ╔═╡ 7afa8ad0-f683-11ea-2b99-d1adba1d3fb3
md"**TODO**: Alpha vectors from POMDPs.jl notebook."

# ╔═╡ 7f07997e-f681-11ea-0536-61cb4401fbdf
md"### Pruning"

# ╔═╡ ae90df20-f683-11ea-270e-7321ece2c890
md"### Finding Dominating"

# ╔═╡ 96378eb0-f688-11ea-3a61-57de047a8dca
md"# Appendix"

# ╔═╡ 802e70d0-f682-11ea-3a67-c5d5c9ac34ad
# macro todo_str(str)
# 	Meta.quot(md"$str")
# end

# ╔═╡ 81e840b0-f680-11ea-189f-fd02cf3d1600
# fix spacing in bulleted lists
html"<style>ul li p {margin: 0} ol li p {margin: 0}</style>"

# ╔═╡ Cell order:
# ╟─38451640-f680-11ea-1931-a7d267fbcaf9
# ╟─5a1edbc0-f680-11ea-08c5-49c22c4b4b97
# ╟─f31ffd10-f683-11ea-0482-793bf05deee0
# ╟─010caa10-f682-11ea-3deb-b728b1d405d7
# ╟─27823ed0-f682-11ea-167b-a1523627e265
# ╟─bec045a0-f680-11ea-16da-c77da3ba35e6
# ╟─e04f5110-f681-11ea-1fff-470db0d13073
# ╠═dbeae580-f681-11ea-0605-0da1c23cb448
# ╟─d86e68f0-f681-11ea-0577-a14d7ac9c74b
# ╟─d33c4d20-f681-11ea-0602-093a0453cfa7
# ╠═62ae1a60-f682-11ea-0db8-31880fdb771e
# ╠═c9739230-f681-11ea-0dc5-071d7a96aa1f
# ╠═0e25d640-f687-11ea-1d5d-f73947ddffaa
# ╠═27caf5b0-f684-11ea-3592-6303dc526cd9
# ╟─e95b60d0-f683-11ea-3684-cff946175c9a
# ╠═1ed1b52e-f683-11ea-1030-0ff004c644fb
# ╠═5f50d3ae-f684-11ea-326a-173e645a3f15
# ╠═3abbe900-f688-11ea-2727-6bc22eb364a8
# ╠═a25ab130-f684-11ea-04a2-b17aef6e9101
# ╠═9725edb2-f685-11ea-3282-6fca02b29983
# ╠═c20cbf42-f685-11ea-3854-c7fff3b94bdb
# ╠═513846a0-f684-11ea-2766-310f8561add1
# ╠═e9db37d0-f686-11ea-34bd-4d2e72564f1d
# ╠═ef6dea30-f686-11ea-0b4e-27cf68080e2a
# ╠═465c3f92-f687-11ea-3ac9-d927e67cc870
# ╠═5a476e30-f687-11ea-3740-894fb7053372
# ╠═4e01a550-f687-11ea-0e3a-1739be23fec3
# ╠═6da529e0-f687-11ea-39b6-53aee839f504
# ╟─b1a9bb10-f687-11ea-2f16-736f7b124268
# ╠═a61e0220-f681-11ea-0d29-ff5c63073bfd
# ╠═9fffffb0-f681-11ea-0d36-2deb4d62ab33
# ╠═5c0e2142-f683-11ea-0a77-bbe7df75a4e0
# ╟─a593dbd0-f687-11ea-3270-69f52f78ec87
# ╠═a94d9730-f681-11ea-17fb-f523a92590a2
# ╟─5db20d40-f683-11ea-34f3-9be2d2d45c00
# ╠═94fbb780-f681-11ea-0bb5-af29f2c8c0b2
# ╠═81974510-f681-11ea-0175-b3546e1192b8
# ╠═7afa8ad0-f683-11ea-2b99-d1adba1d3fb3
# ╠═7f07997e-f681-11ea-0536-61cb4401fbdf
# ╠═ae90df20-f683-11ea-270e-7321ece2c890
# ╟─96378eb0-f688-11ea-3a61-57de047a8dca
# ╠═802e70d0-f682-11ea-3a67-c5d5c9ac34ad
# ╠═81e840b0-f680-11ea-189f-fd02cf3d1600
