### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 634b2ed0-0f1a-11eb-30aa-e1092f5f338f
md"""
# Crying Baby Problem
For code, see [StateEstimation.jl](https://github.com/mossr/StateEstimation.jl)

We cannot directly observe whether the baby is hungry or not (i.e. the true states), but we can observe if it is *crying* or *quite* and use that as a noisy observation to update our beliefs about their true state.
"""

# ╔═╡ 7b68d4d0-0f20-11eb-2484-b354c4cff750
md"""
The state, action, and observation spaces are:

$$\begin{align}
	\mathcal{S} &= \{\text{hungry},\, \text{sated}\}\tag{state space}\\
	\mathcal{A} &= \{\text{feed},\, \text{sing},\, \text{ignore}\}\tag{action space}\\
	\mathcal{O} &= \{\text{crying},\, \text{quiet}\}\tag{observation space}
\end{align}$$
"""

# ╔═╡ 70bb6df0-0f1a-11eb-0055-25079b36caaf
begin
	@enum State hungry sated
	@enum Action feed sing ignore
	@enum Observation crying quiet
end

# ╔═╡ a0d7ddc0-0f1a-11eb-291d-59e2b1633f67
md"""
## POMDP definition
"""

# ╔═╡ 105acfb0-195a-11eb-0920-89dfeeedd245
md"""
$$\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, R, O, \gamma \rangle\tag{POMDP 7-tuple}$$
"""

# ╔═╡ a4f86280-0f1a-11eb-104e-8fef3d3303ef
struct POMDP
	𝒮 # state space
	𝒜 # action space
	𝒪 # observation space
	T # transition function
	R # reward function
	O # observation function
	γ # discount factor
end

# ╔═╡ 7b94ae80-0f1a-11eb-1aef-eb26390584d8
md"""
## Transition model
Also called the *transition function*.
"""

# ╔═╡ 928b39a0-0f20-11eb-10e9-d1289faf91f9
md"""
$$T(s^\prime \mid a, s)$$

$$\begin{aligned}
    T(\text{sated}  \ \mid &\>\text{hungry}, \text{feed}) &= 100\% \\
    T(\text{hungry} \ \mid &\>\text{hungry}, \text{sing}) &= 100\% \\
    T(\text{hungry} \ \mid &\>\text{hungry}, \text{ignore}) &= 100\% \\
    T(\text{sated}  \ \mid &\>\text{sated}, \text{feed}) &= 100\% \\
    T(\text{hungry} \ \mid &\>\text{sated}, \text{sing}) &= 10\% \\
    T(\text{hungry} \ \mid &\>\text{sated}, \text{ignore}) &= 10\%
\end{aligned}$$
"""

# ╔═╡ 7569dace-0f1a-11eb-3530-ab7c4a4b0163
function T(s, a, s′)
	if a == feed
		return s′ == hungry ? 0 : 1
	elseif s == hungry && (a == sing || a == ignore)
		return s′ == hungry ? 1 : 0
	elseif s == sated && (a == sing || a == ignore)
		return s′ == hungry ? 0.1 : 0.9
	end
end

# ╔═╡ 80c2f9c0-0f1a-11eb-1cd4-a128ecee865d
md"""
## Reward model
Also called the *reward function*. We assign $-10$ reward if the baby is hungry and $-5$ reward for feeding the baby (which is additive). Singing to a *sated* baby yields $5$ reward, but singing to a *hungry* baby incurs $-2$ reward.

$$R(s,a)$$
"""

# ╔═╡ 77b50350-0f1a-11eb-3518-2f84ee105ca1
function R(s, a)
	return (s == hungry ? -10 : 0) +
	       (a == feed ? -5 : 0) +
	       (a == sing && s == sated ? +5 : 0) +
	       (a == sing && s == hungry ? -2 : 0)
end

# ╔═╡ 8a9c0860-0f1a-11eb-312f-cf748280d3e6
md"""
## Observation model
A *hungry* baby cries $80\%$ of the time, whereas a *sated* baby cries $10\%$ of the time. Singing to the baby yields a perfect observation.

$$O(o \mid a, s^\prime)$$
"""

# ╔═╡ 7a273bd0-0f1a-11eb-3218-8b33c0141bb8
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

# ╔═╡ b22f1b10-0f1a-11eb-229e-ed202606a72c
md"""
## Belief updating
$$\begin{gather}
b^\prime(s^\prime) \propto O(o \mid a, s^\prime) \sum_s T(s^\prime \mid s, a)b(s) \tag{then normalize}
\end{gather}$$
"""

# ╔═╡ b04ebe20-0f21-11eb-2607-0b01c99ac423
import LinearAlgebra: normalize!

# ╔═╡ b4db7662-0f1a-11eb-3fb8-d58feae7e66c
function update(b::Vector{Float64}, 𝒫, a, o)
	𝒮, T, O = 𝒫.𝒮 ,𝒫.T, 𝒫.O
	b′ = similar(b)
	for (i′, s′) in enumerate(𝒮)
		b′[i′] = O(a, s′, o) * sum(T(s, a, s′) * b[i] for (i, s) in enumerate(𝒮))
	end
	if sum(b′) ≈ 0.0
		fill!(b′, 1)
	end
	return normalize!(b′, 1)
end

# ╔═╡ 84db99ce-0f1b-11eb-27a3-678c1c091540
md"""
# Instantiating the crying baby POMDP
"""

# ╔═╡ e35cb6de-1959-11eb-09dc-938c74a5877b
# State, action, and observation spaces (or sets)
begin
	𝒮 = (hungry, sated)
	𝒜 = (feed, sing, ignore)
	𝒪 = (crying, quiet)
end;

# ╔═╡ 90dba720-0f1b-11eb-2d19-f501bb8f3286
𝒫 = POMDP(𝒮,   # state space
	      𝒜,   # action space
	      𝒪,   # observation space
	      T,   # transition model
	      R,   # reward model
	      O,   # observation model
		  0.9) # discount factor

# ╔═╡ b1dc4ab0-0f1b-11eb-3f29-dbfcbd991121
md"""
## Example: updaing beliefs
$$\mathbf b = \begin{bmatrix} p(\text{hungry}) \\ p(\text{sated})\end{bmatrix} = \text{belief vector over states}$$
"""

# ╔═╡ bcbda190-0f1b-11eb-147e-79986a90edef
md"""
We start with an initial uniform belief $b_0$ across the states *hungry* and *sated*.
"""

# ╔═╡ b9fa62e0-0f1b-11eb-0bbd-271d46693d2b
b₀ = [0.5, 0.5]

# ╔═╡ cde25d30-0f1b-11eb-2fa3-65e902a27818
md"""
Then we update our belief if we *ignore* the baby and observe it *crying*.
"""

# ╔═╡ dc904c70-0f1b-11eb-0a8c-bd9770a4a074
b₁ = update(b₀, 𝒫, ignore, crying)

# ╔═╡ dd5a5330-0f1b-11eb-3fe5-6f3ae2246b2a
md"""
Updating again after we *feed* the baby and observe it becomes *quiet*.
"""

# ╔═╡ 03157190-0f1c-11eb-3658-493ade3025d2
b₂ = update(b₁, 𝒫, feed, quiet)

# ╔═╡ 752b1230-0f1c-11eb-1d64-5314eabbd247
md"""
Then we *ignore* the baby and still observe it is *quiet*.
"""

# ╔═╡ 81dfa26e-0f1c-11eb-3753-4713abb1b5cc
b₃ = update(b₂, 𝒫, ignore, quiet)

# ╔═╡ a1ee6bf0-0f1c-11eb-25fb-9b2508e6ceed
md"""
Again we *ignore* the baby and still observe it is *quiet*.
"""

# ╔═╡ 9625c250-0f1c-11eb-11fd-0d7c8cef3ad2
b₄ = update(b₃, 𝒫, ignore, quiet)

# ╔═╡ 869477f0-0f1c-11eb-1a0c-5d59b8c2145d
md"""
Finally, we *ignore* the baby again and observe that it's *crying*.
"""

# ╔═╡ a8b817b0-0f1c-11eb-055b-a34848891544
b₅ = update(b₄, 𝒫, ignore, crying)

# ╔═╡ ac8dfee0-0f1c-11eb-0207-ed93b0d6f9fc
md"""
And recall, this final belief $b_5$ is telling us that we *believe* the baby is **hungry** with probability $0.538$ and that it is **sated** with probability $0.462$. Only given observations and without seeing the true state.
"""

# ╔═╡ Cell order:
# ╟─634b2ed0-0f1a-11eb-30aa-e1092f5f338f
# ╟─7b68d4d0-0f20-11eb-2484-b354c4cff750
# ╠═70bb6df0-0f1a-11eb-0055-25079b36caaf
# ╟─a0d7ddc0-0f1a-11eb-291d-59e2b1633f67
# ╟─105acfb0-195a-11eb-0920-89dfeeedd245
# ╠═a4f86280-0f1a-11eb-104e-8fef3d3303ef
# ╟─7b94ae80-0f1a-11eb-1aef-eb26390584d8
# ╟─928b39a0-0f20-11eb-10e9-d1289faf91f9
# ╠═7569dace-0f1a-11eb-3530-ab7c4a4b0163
# ╟─80c2f9c0-0f1a-11eb-1cd4-a128ecee865d
# ╠═77b50350-0f1a-11eb-3518-2f84ee105ca1
# ╟─8a9c0860-0f1a-11eb-312f-cf748280d3e6
# ╠═7a273bd0-0f1a-11eb-3218-8b33c0141bb8
# ╟─b22f1b10-0f1a-11eb-229e-ed202606a72c
# ╠═b04ebe20-0f21-11eb-2607-0b01c99ac423
# ╠═b4db7662-0f1a-11eb-3fb8-d58feae7e66c
# ╟─84db99ce-0f1b-11eb-27a3-678c1c091540
# ╠═e35cb6de-1959-11eb-09dc-938c74a5877b
# ╠═90dba720-0f1b-11eb-2d19-f501bb8f3286
# ╟─b1dc4ab0-0f1b-11eb-3f29-dbfcbd991121
# ╟─bcbda190-0f1b-11eb-147e-79986a90edef
# ╠═b9fa62e0-0f1b-11eb-0bbd-271d46693d2b
# ╟─cde25d30-0f1b-11eb-2fa3-65e902a27818
# ╠═dc904c70-0f1b-11eb-0a8c-bd9770a4a074
# ╟─dd5a5330-0f1b-11eb-3fe5-6f3ae2246b2a
# ╠═03157190-0f1c-11eb-3658-493ade3025d2
# ╟─752b1230-0f1c-11eb-1d64-5314eabbd247
# ╠═81dfa26e-0f1c-11eb-3753-4713abb1b5cc
# ╟─a1ee6bf0-0f1c-11eb-25fb-9b2508e6ceed
# ╠═9625c250-0f1c-11eb-11fd-0d7c8cef3ad2
# ╟─869477f0-0f1c-11eb-1a0c-5d59b8c2145d
# ╠═a8b817b0-0f1c-11eb-055b-a34848891544
# ╟─ac8dfee0-0f1c-11eb-0207-ed93b0d6f9fc
