### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 2675774e-e020-48be-b79e-c8c234e4d57e
using PlutoUI

# ╔═╡ 3e8bd204-c86b-4f9b-8e0d-4ca8526fd80c
html"<h1>Crying baby POMDP</h1>
For code, see <a href='https://github.com/mossr/StateEstimation.jl'>StateEstimation.jl</a>:
<button onclick='present()' style='
	# margin: auto;
	# display: block;
    background-color: white;
    color: black;
    border-color: black; 
    border-radius: 5px; 
    border-width: 1px;
    min-height: 20px; 
    min-width: 100px;'>Present</button>
<p>
<p>
We cannot directly observe whether the baby is hungry or not (i.e. the true states), but we can observe if it is <it>crying</it> or <it>quite</it> and use that as a noisy observation to update our beliefs about their true state."

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
# POMDP definition
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
# Transition model
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
end # returns likelihood/probability

# ╔═╡ 80c2f9c0-0f1a-11eb-1cd4-a128ecee865d
md"""
# Reward model
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
# Observation model
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
end # returns likelihood/probability

# ╔═╡ b22f1b10-0f1a-11eb-229e-ed202606a72c
md"""
# Belief updating
$$\begin{gather}
b^\prime(s^\prime) \propto O(o \mid a, s^\prime) \sum_s T(s^\prime \mid s, a)b(s) \tag{then normalize}
\end{gather}$$
"""

# ╔═╡ b04ebe20-0f21-11eb-2607-0b01c99ac423
import LinearAlgebra: normalize!

# ╔═╡ b4db7662-0f1a-11eb-3fb8-d58feae7e66c
function update(b::Vector{Float64}, 𝒫; a, o)
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
# Example: Updating beliefs
$$\mathbf b = \begin{bmatrix} p(\text{hungry}) \\ p(\text{sated})\end{bmatrix} = \text{belief vector over states}$$
"""

# ╔═╡ bcbda190-0f1b-11eb-147e-79986a90edef
md"""
We start with an initial uniform belief $b_0$ across the states *hungry* and *sated*.
"""

# ╔═╡ b9fa62e0-0f1b-11eb-0bbd-271d46693d2b
b₀ = [0.5, 0.5]

# ╔═╡ e4a2fb2a-6ef1-4221-861a-a8698a8d62c8
md"""
# Belief updating step 1
"""

# ╔═╡ cde25d30-0f1b-11eb-2fa3-65e902a27818
md"""
Then we update our belief if we *ignore* the baby and observe it *crying*.
"""

# ╔═╡ dc904c70-0f1b-11eb-0a8c-bd9770a4a074
b₁ = update(b₀, 𝒫, a=ignore, o=crying)

# ╔═╡ 4b457bc6-8675-411d-b4b7-bd7b914033d4
md"""
# Belief updating step 2
"""

# ╔═╡ dd5a5330-0f1b-11eb-3fe5-6f3ae2246b2a
md"""
Updating again after we *feed* the baby and observe it becomes *quiet*.
"""

# ╔═╡ 03157190-0f1c-11eb-3658-493ade3025d2
b₂ = update(b₁, 𝒫, a=feed, o=quiet)

# ╔═╡ 1512ac13-5514-4241-9a0e-04e83ea1e375
md"""
# Belief updating step 3
"""

# ╔═╡ 752b1230-0f1c-11eb-1d64-5314eabbd247
md"""
Then we *ignore* the baby and still observe it is *quiet*.
"""

# ╔═╡ 81dfa26e-0f1c-11eb-3753-4713abb1b5cc
b₃ = update(b₂, 𝒫, a=ignore, o=quiet)

# ╔═╡ 97a9d541-137c-4151-9ad3-369f6a1aa6d1
md"""
# Belief updating step 4
"""

# ╔═╡ a1ee6bf0-0f1c-11eb-25fb-9b2508e6ceed
md"""
Again we *ignore* the baby and still observe it is *quiet*.
"""

# ╔═╡ 9625c250-0f1c-11eb-11fd-0d7c8cef3ad2
b₄ = update(b₃, 𝒫, a=ignore, o=quiet)

# ╔═╡ 0dae373b-f284-41b1-bb50-aa5511c937d5
md"""
# Belief updating step 5
"""

# ╔═╡ 869477f0-0f1c-11eb-1a0c-5d59b8c2145d
md"""
Finally, we *ignore* the baby again and observe that it's *crying*.
"""

# ╔═╡ a8b817b0-0f1c-11eb-055b-a34848891544
b₅ = update(b₄, 𝒫, a=ignore, o=crying)

# ╔═╡ ac8dfee0-0f1c-11eb-0207-ed93b0d6f9fc
md"""
And recall, this final belief $b_5$ is telling us that we *believe* the baby is **hungry** with probability $0.538$ and that it is **sated** with probability $0.462$. Only given observations and without seeing the true state.
"""

# ╔═╡ 9e8bf74f-7806-49a9-83ef-d1870f65ad58
md"""
# Belief vector
The belief vector represents a _discrete probability distribution_, therefore it must be strictly non-negative and sum to one:

$$b(s) \ge 0 \text{ for all } s \in \mathcal{S} \qquad\qquad \sum_s b(s) = 1$$
"""

# ╔═╡ eee200df-0b39-4b51-84e6-012f0cfc59da
[all(b .≥ 0) && sum(b) ≈ 1 for b in [b₀, b₁, b₂, b₃, b₄, b₅]]

# ╔═╡ b94886d3-de31-40f0-93d1-d611a9d0d6f2
md"""
# Back matter
"""

# ╔═╡ 47790444-a55f-4e12-86cf-08be342a4af5
TableOfContents(title="POMDP")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.53"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "ae9521c012303ba1f263ceaef6568f74b290b26e"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8ec28846dbf846228a32de5a6912c63e2052e3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.53"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─3e8bd204-c86b-4f9b-8e0d-4ca8526fd80c
# ╟─7b68d4d0-0f20-11eb-2484-b354c4cff750
# ╠═70bb6df0-0f1a-11eb-0055-25079b36caaf
# ╟─a0d7ddc0-0f1a-11eb-291d-59e2b1633f67
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
# ╟─e4a2fb2a-6ef1-4221-861a-a8698a8d62c8
# ╟─cde25d30-0f1b-11eb-2fa3-65e902a27818
# ╠═dc904c70-0f1b-11eb-0a8c-bd9770a4a074
# ╟─4b457bc6-8675-411d-b4b7-bd7b914033d4
# ╟─dd5a5330-0f1b-11eb-3fe5-6f3ae2246b2a
# ╠═03157190-0f1c-11eb-3658-493ade3025d2
# ╟─1512ac13-5514-4241-9a0e-04e83ea1e375
# ╟─752b1230-0f1c-11eb-1d64-5314eabbd247
# ╠═81dfa26e-0f1c-11eb-3753-4713abb1b5cc
# ╟─97a9d541-137c-4151-9ad3-369f6a1aa6d1
# ╟─a1ee6bf0-0f1c-11eb-25fb-9b2508e6ceed
# ╠═9625c250-0f1c-11eb-11fd-0d7c8cef3ad2
# ╟─0dae373b-f284-41b1-bb50-aa5511c937d5
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
