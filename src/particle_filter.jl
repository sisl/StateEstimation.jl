### A Pluto.jl notebook ###
# v0.11.14

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

# ╔═╡ 740dc710-fbaf-11ea-2062-7f44056cbd12
using AddPackage

# ╔═╡ de842650-f7e7-11ea-3f11-5b92ea413bb5
@add using Distributions, LinearAlgebra

# ╔═╡ 3cafb210-f89e-11ea-0cf2-bdf819224cc9
@add using PlutoUI, Test, Random

# ╔═╡ b9b56160-fc95-11ea-18e0-737a0aa29148
using Reel

# ╔═╡ 85830e20-fb77-11ea-1e9f-d3651f6fe718
@add using Suppressor

# ╔═╡ 3145281e-fc3a-11ea-3f49-8590a886aa73
include("section_counters.jl")

# ╔═╡ 2cbec03e-fb77-11ea-09a2-634fac25a12a
md"# Particle filter"

# ╔═╡ 038c5510-f8bc-11ea-0fc5-7d765d868496
md"## POMDP definition"

# ╔═╡ 5d9e4bf0-f7e8-11ea-23d8-2dbd72e46ce6
struct POMDP 𝒮; 𝒜; 𝒪; T; O end

# ╔═╡ 1a06d470-f7e8-11ea-3640-c3964cba9e1f
begin
	function particle_filter(𝐛::Vector, 𝒫::POMDP, a, o)
		(T, O) = (𝒫.T, 𝒫.O)
		𝐬′ = rand.(T.(𝐛, a))
		𝐰 = O.(a, 𝐬′, o)
		D = Categorical(normalize(𝐰, 1))
		return 𝐬′[rand(D, length(𝐬′))]
	end

	function particle_filter(𝐛::Matrix, 𝒫::POMDP, a, o)
		(T, O) = (𝒫.T, 𝒫.O)
		𝐬′ = mapslices(b->rand(T(b, a)), 𝐛; dims=1)
		𝐰 = mapslices(s′->O(a, s′, o), 𝐬′; dims=1)
		𝐰ₙ = mapslices(w->normalize(w, 1), 𝐰; dims=2)
		if isnan(sum(𝐰ₙ))
			fill!(𝐰ₙ, 1/length(𝐰ₙ))
		end
		D = Categorical(vec(𝐰ₙ))
		return 𝐬′[:, rand(D, size(𝐬′, 2))]
	end
end

# ╔═╡ 608a4850-f7e8-11ea-2fca-af35a2f0456b
begin
	𝒮 = -10:10
	𝒜 = Normal(0, 1)
	𝒪 = Uniform(-10, 10)
	transition = (s,a) -> clamp(s+a, minimum(𝒮), maximum(𝒮))
	T = (s,a) -> Normal(transition(s,a), abs(a))
	observation = (s′,a) -> Normal(s′, abs(a))
	O = (a,s′,o) -> pdf(observation(s′,a), o)
	𝒫 = POMDP(𝒮, 𝒜, 𝒪, T, O)
end

# ╔═╡ 4099e950-fb77-11ea-23b7-6d1f7b47c07e
md"## Simulation and testing"

# ╔═╡ d83c01c0-fb78-11ea-0543-d3a0fdcbadab
function test_filter(belief, s)
	μ_b = mean(belief)
	σ_b = std(belief)
	belief_error = abs(μ_b - s)
	@test (μ_b-3σ_b ≤ s ≤ μ_b+3σ_b) || belief_error ≤ 1.0
end

# ╔═╡ 707e9b30-f8a1-11ea-0a6c-ad6756d07bbc
md"""
$(@bind t Slider(0:2000, show_value=true, default=10))
$(@bind stationary CheckBox())
"""

# ╔═╡ a89bbc40-fb77-11ea-3a1b-7197afa0c9b0
function step(𝒫, belief, 𝒜, s, a, o, transition, observation)
	a = rand(𝒜)
	if !stationary
		s = transition(s, a)
		o = rand(observation(s, a))
	end
	belief = particle_filter(belief, 𝒫, a, o)
	return (belief, s, a, o)
end

# ╔═╡ f45355a0-fc65-11ea-26ff-1fd18bdbfdb2
md"## Random walk 2D example"

# ╔═╡ faf88970-fc65-11ea-3283-03df32338623
begin
    𝒮2 = Product(Uniform.([-10, -10], [10, 10]))
	𝒮ₘᵢₙ = minimum.(support.(𝒮2.v))
	𝒮ₘₐₓ = maximum.(support.(𝒮2.v))

	𝒜2 = MvNormal([0, 0], [1 0; 0 1])
	𝒪2 = Product(Uniform.([-10, -10], [10, 10]))

	transition2 = (s,a) -> clamp.(s .+ a, 𝒮ₘᵢₙ, 𝒮ₘₐₓ)
    T2 = (s,a) -> MvNormal(transition2(s,a), I*abs.(a))

	observation2 = (s′,a) -> MvNormal(s′, I*abs.(a))
    O2 = (a,s′,o) -> pdf(observation2(s′,a), o)
    𝒫2 = POMDP(𝒮2, 𝒜2, 𝒪2, T2, O2)
end;

# ╔═╡ a30d19f0-fc66-11ea-211b-87a727b700cb
md"""
$(@bind t2 Slider(0:500, show_value=true, default=7))
"""

# ╔═╡ 7a9c9430-fc95-11ea-3fa7-6bbf6c0aec33
function plot_walk2d(belief, true_state, iteration, action)
	clf()

	scatter(belief[1,:], belief[2,:], 1, alpha=0.25, marker=".", color="black")

	plot(true_state..., "ro")
    xlim([-10, 10])
    ylim([-10, 10])
    title("iteration=$iteration, action=$(round.(action, digits=4))")
    gcf()
end

# ╔═╡ fee6c082-fc9a-11ea-2209-3b9e1a3c9526
md"### Writing GIFs"

# ╔═╡ 71771e20-fc95-11ea-172c-3fa41cc22792
begin
	frames = Frames(MIME("image/png"), fps=2)
	for iter in 1:30
		Random.seed!(0x228)
		global frames
		belief2plot = rand(𝒮2, 1000)
		o2plot = rand(𝒪2)
		s2plot = o2plot
		a2plot = missing
		if iter == 1
			# X initial frames stationary
			[push!(frames,
				   plot_walk2d(belief2plot, s2plot, iter, [0, 0])) for _ in 1:3]
		end
		for i in 1:iter
			(belief2plot, s2plot, a2plot, o2plot) =
				step(𝒫2, belief2plot, 𝒜2, s2plot, a2plot, o2plot,
				     transition2, observation2)
		end
		push!(frames, plot_walk2d(belief2plot, s2plot, iter, a2plot))
	end
	write("particle_filter.gif", frames)
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
	@testset begin
		Random.seed!(228)
		global m = 1000
		global belief = rand(𝒮, m)
		global o = rand(𝒪)
		global s = o
		global a = missing
		for i in 1:t
			(belief, s, a, o) = step(𝒫, belief, 𝒜, s, a, o, transition, observation)
			test_filter(belief, s)
		end
	end
end

# ╔═╡ 43027b00-f7ec-11ea-3354-c15426d5e63f
begin
	@add using PyPlot; PyPlot.svg(true)
	clf()
	hist(belief)
	plot(s, 0, "ro")
	xlim([-10, 10])
	ylim([0, m])
	title("iteration=$t, action=$(round(a, digits=4))")
	gcf()
end

# ╔═╡ 3cb2dc82-fc66-11ea-2772-6307b9e219d9
with_terminal() do
	# @testset begin
		Random.seed!(0x228)
		global m2 = 1000
		global belief2 = rand(𝒮2, m2)
		global o2 = rand(𝒪2)
		global s2 = o2
		global a2 = missing
		for i in 1:t2
			(belief2, s2, a2, o2) = step(𝒫2, belief2, 𝒜2, s2, a2, o2,
			                             transition2, observation2)
			# test_filter(belief, s)
		end
		@show s2
	# end
end

# ╔═╡ 2794d330-fc66-11ea-0a35-f57068b69c0e
plot_walk2d(belief2, s2, t2, a2)

# ╔═╡ 5c8239f0-fc90-11ea-2e1e-9703069d37af
md"LaTeX-style fonts in `PyPlot`."

# ╔═╡ dd875c22-fc8f-11ea-3557-6d3ad934151d
begin
	# LaTeX-style fonts in PyPlot
	matplotlib.rc("font", family=["serif"])
	matplotlib.rc("font", serif=["Helvetica"])
	matplotlib.rc("text", usetex=true)
end

# ╔═╡ Cell order:
# ╟─2cbec03e-fb77-11ea-09a2-634fac25a12a
# ╠═740dc710-fbaf-11ea-2062-7f44056cbd12
# ╠═de842650-f7e7-11ea-3f11-5b92ea413bb5
# ╠═1a06d470-f7e8-11ea-3640-c3964cba9e1f
# ╟─038c5510-f8bc-11ea-0fc5-7d765d868496
# ╠═5d9e4bf0-f7e8-11ea-23d8-2dbd72e46ce6
# ╠═608a4850-f7e8-11ea-2fca-af35a2f0456b
# ╟─4099e950-fb77-11ea-23b7-6d1f7b47c07e
# ╠═3cafb210-f89e-11ea-0cf2-bdf819224cc9
# ╠═a89bbc40-fb77-11ea-3a1b-7197afa0c9b0
# ╠═d83c01c0-fb78-11ea-0543-d3a0fdcbadab
# ╠═c447b370-f7eb-11ea-1435-bd549afa0181
# ╟─707e9b30-f8a1-11ea-0a6c-ad6756d07bbc
# ╠═43027b00-f7ec-11ea-3354-c15426d5e63f
# ╟─f45355a0-fc65-11ea-26ff-1fd18bdbfdb2
# ╠═faf88970-fc65-11ea-3283-03df32338623
# ╟─a30d19f0-fc66-11ea-211b-87a727b700cb
# ╠═3cb2dc82-fc66-11ea-2772-6307b9e219d9
# ╠═7a9c9430-fc95-11ea-3fa7-6bbf6c0aec33
# ╠═2794d330-fc66-11ea-0a35-f57068b69c0e
# ╟─fee6c082-fc9a-11ea-2209-3b9e1a3c9526
# ╠═b9b56160-fc95-11ea-18e0-737a0aa29148
# ╠═71771e20-fc95-11ea-172c-3fa41cc22792
# ╟─802c5e80-f8b2-11ea-310f-6fdbcacb73d0
# ╠═85830e20-fb77-11ea-1e9f-d3651f6fe718
# ╟─67ebdf80-f8b2-11ea-2630-d54abc89ad2b
# ╠═3145281e-fc3a-11ea-3f49-8590a886aa73
# ╟─5c8239f0-fc90-11ea-2e1e-9703069d37af
# ╠═dd875c22-fc8f-11ea-3557-6d3ad934151d
