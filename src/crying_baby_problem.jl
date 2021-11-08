### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 2675774e-e020-48be-b79e-c8c234e4d57e
using PlutoUI

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
## Example: updating beliefs
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

# ╔═╡ 9e8bf74f-7806-49a9-83ef-d1870f65ad58
md"""
### Belief vector
The belief vector represents a _discrete probability distribution_, therefore it must be strictly non-negative and sum to one:

$$b(s) \ge 0 \text{ for all } s \in \mathcal{S} \qquad\qquad \sum_s b(s) = 1$$
"""

# ╔═╡ eee200df-0b39-4b51-84e6-012f0cfc59da
[all(b .≥ 0) && sum(b) ≈ 1 for b in [b₀, b₁, b₂, b₃, b₄, b₅]]

# ╔═╡ b94886d3-de31-40f0-93d1-d611a9d0d6f2
md"""
---
"""

# ╔═╡ 47790444-a55f-4e12-86cf-08be342a4af5
TableOfContents(title="POMDP")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.18"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0ec322186e078db08ea3e7da5b8b2885c099b393"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "5efcf53d798efede8fee5b2c8b09284be359bf24"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.2"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "57312c7ecad39566319ccf5aa717a20788eb8c1f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.18"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
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
# ╟─9e8bf74f-7806-49a9-83ef-d1870f65ad58
# ╠═eee200df-0b39-4b51-84e6-012f0cfc59da
# ╟─b94886d3-de31-40f0-93d1-d611a9d0d6f2
# ╠═2675774e-e020-48be-b79e-c8c234e4d57e
# ╠═47790444-a55f-4e12-86cf-08be342a4af5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
