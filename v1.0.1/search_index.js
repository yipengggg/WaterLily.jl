var documenterSearchIndex = {"docs":
[{"location":"#WaterLily-1","page":"WaterLily","title":"WaterLily","text":"","category":"section"},{"location":"#Introduction-and-Quickstart-1","page":"WaterLily","title":"Introduction and Quickstart","text":"","category":"section"},{"location":"#","page":"WaterLily","title":"WaterLily","text":"See the WaterLily README for an Introduction and Quickstart.","category":"page"},{"location":"#Types-Methods-and-Functions-1","page":"WaterLily","title":"Types Methods and Functions","text":"","category":"section"},{"location":"#","page":"WaterLily","title":"WaterLily","text":"CurrentModule = WaterLily","category":"page"},{"location":"#","page":"WaterLily","title":"WaterLily","text":"","category":"page"},{"location":"#","page":"WaterLily","title":"WaterLily","text":"Modules = [WaterLily]","category":"page"},{"location":"#WaterLily.AbstractBody","page":"WaterLily","title":"WaterLily.AbstractBody","text":"AbstractBody\n\nImmersed body Abstract Type. Any AbstractBody subtype must implement\n\n`d = sdf(body::AbstractBody, x, t=0)` and\n`d,n,V = measure(body::AbstractBody, x, t=0)`\n\nwhere `d` is the signed distance from `x` to the body at time `t`,\nand `n` & `V` are the normal and velocity vectors implied at `x`.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.AbstractPoisson","page":"WaterLily","title":"WaterLily.AbstractPoisson","text":"Poisson{N,M}\n\nComposite type for conservative variable coefficient Poisson equations:\n\n∮ds β ∂x/∂n = σ\n\nThe resulting linear system is\n\nAx = [L+D+L']x = z\n\nwhere A is symmetric, block-tridiagonal and extremely sparse. Moreover,  D[I]=-∑ᵢ(L[I,i]+L'[I,i]). This means matrix storage, multiplication, ect can be easily implemented and optimized without external libraries.\n\nTo help iteratively solve the system above, the Poisson structure holds helper arrays for inv(D), the error ϵ, and residual r=z-Ax. An iterative solution method then estimates the error ϵ=̃A⁻¹r and increments x+=ϵ, r-=Aϵ.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.AutoBody","page":"WaterLily","title":"WaterLily.AutoBody","text":"AutoBody(sdf,map=(x,t)->x; compose=true) <: AbstractBody\n\n- sdf(x::AbstractVector,t::Real)::Real: signed distance function\n- map(x::AbstractVector,t::Real)::AbstractVector: coordinate mapping function\n- compose::Bool=true: Flag for composing sdf=sdf∘map\n\nImplicitly define a geometry by its sdf and optional coordinate map. Note: the map is composed automatically if compose is set to true, ie sdf(x,t) = sdf(map(x,t),t).  Both parameters remain independent otherwise. It can be particularly heplful to set it as  false when adding mulitple bodies together to create a more complexe one.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.Flow","page":"WaterLily","title":"WaterLily.Flow","text":"Flow{D, V, S, F, B, T}\n\nComposite type for a multidimensional immersed boundary flow simulation.\n\nFlow solves the unsteady incompressible Navier-Stokes equations on a Cartesian grid. Solid boundaries are modelled using the Boundary Data Immersion Method. The primary variables are the scalar pressure p (an array of dimension N) and the velocity vector field u (an array of dimension M=N+1).\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.MultiLevelPoisson","page":"WaterLily","title":"WaterLily.MultiLevelPoisson","text":"MultiLevelPoisson{N,M}\n\nComposite type used to solve the pressure Poisson equation with a geometric multigrid method. The only variable is levels, a vector of nested Poisson systems.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.Simulation","page":"WaterLily","title":"WaterLily.Simulation","text":"Simulation(dims::NTuple, u_BC::NTuple, L::Number;\n           U=norm2(u_BC), Δt=0.25, ν=0., ϵ = 1,\n           uλ::Function=(i,x)->u_BC[i],\n           body::AbstractBody=NoBody(),\n           T=Float32, mem = Array)\n\nConstructor for a WaterLily.jl simulation:\n\n`dims`: Simulation domain dimensions.\n`u_BC`: Simulation domain velocity boundary conditions, `u_BC[i]=uᵢ, i=eachindex(dims)`.\n`L`: Simulation length scale.\n`U`: Simulation velocity scale.\n`Δt`: Initial time step.\n`ν`: Scaled viscosity (`Re=UL/ν`).\n`ϵ`: BDIM kernel width.\n`uλ`: Function to generate the initial velocity field.\n`body`: Immersed geometry.\n`T`: Array element type.\n`mem`: memory location. `Array` and `CuArray` run on CPU and CUDA backends, respectively.\n\nSee files in examples folder for examples.\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.BC!","page":"WaterLily","title":"WaterLily.BC!","text":"BC!(a,A,f=1)\n\nApply boundary conditions to the ghost cells of a vector field. A Dirichlet condition a[I,i]=f*A[i] is applied to the vector component normal to the domain boundary. For example aₓ(x)=f*Aₓ ∀ x ∈ minmax(X). A zero Nuemann condition is applied to the tangential components.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.BC!-Tuple{Any}","page":"WaterLily","title":"WaterLily.BC!","text":"BC!(a)\n\nApply zero Nuemann boundary conditions to the ghost cells of a scalar field.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.L₂-Tuple{Any}","page":"WaterLily","title":"WaterLily.L₂","text":"L₂(a)\n\nL₂ norm of array a excluding ghosts.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.apply!-Tuple{Any, Any}","page":"WaterLily","title":"WaterLily.apply!","text":"apply!(f, c)\n\nApply a vector function f(i,x) to the faces of a uniform staggered array c.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.inside-Tuple{AbstractArray}","page":"WaterLily","title":"WaterLily.inside","text":"inside(a)\n\nReturn CartesianIndices range excluding a single layer of cells on all boundaries.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.loc-Tuple{Any, Any}","page":"WaterLily","title":"WaterLily.loc","text":"loc(i,I)\n\nLocation in space of the cell at CartesianIndex I at face i. Using i=0 returns the cell center s.t. loc = I.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.measure!","page":"WaterLily","title":"WaterLily.measure!","text":"measure!(sim::Simulation,t=time(sim))\n\nMeasure a dynamic body to update the flow and pois coefficients.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.measure!-Union{Tuple{N}, Tuple{Flow{N, T} where T, AbstractBody}} where N","page":"WaterLily","title":"WaterLily.measure!","text":"`measure!(flow::Flow, body::AbstractBody; t=0, ϵ=1)`\n\nQueries the body geometry to fill the arrays:\n\n`flow.μ₀`, Zeroth kernel moment\n`flow.μ₁`, First kernel moment scaled by the body normal\n`flow.V`,  Body velocity\n`flow.σᵥ`, Body velocity divergence scaled by `μ₀-1`\n\nat time t using an immersion kernel of size ϵ. See Maertens & Weymouth, https://doi.org/10.1016/j.cma.2014.09.007\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.measure-Tuple{AutoBody, Any, Any}","page":"WaterLily","title":"WaterLily.measure","text":"d,n,V = measure(body::AutoBody,x,t)\n\nDetermine the implicit geometric properties from the sdf and map. The gradient of d=sdf(map(x,t)) is used to improve d for psuedo-sdfs.  The velocity is determined soley from the optional map function.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.mom_step!-Tuple{Flow, AbstractPoisson}","page":"WaterLily","title":"WaterLily.mom_step!","text":"mom_step!(a::Flow,b::AbstractPoisson)\n\nIntegrate the Flow one time step using the Boundary Data Immersion Method and the AbstractPoisson pressure solver to project the velocity onto an incompressible flow.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.mult!-Tuple{Poisson, Any}","page":"WaterLily","title":"WaterLily.mult!","text":"mult!(p::Poisson,x)\n\nEfficient function for Poisson matrix-vector multiplication.  Fills p.z = p.A x with 0 in the ghost cells.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sim_step!-Tuple{Simulation, Any}","page":"WaterLily","title":"WaterLily.sim_step!","text":"sim_step!(sim::Simulation,t_end;remeasure=true,verbose=false)\n\nIntegrate the simulation sim up to dimensionless time t_end. If remeasure=true, the body is remeasured at every time step.  Can be set to false for static geometries to speed up simulation.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sim_time-Tuple{Simulation}","page":"WaterLily","title":"WaterLily.sim_time","text":"sim_time(sim::Simulation)\n\nReturn the current dimensionless time of the simulation tU/L where t=sum(Δt), and U,L are the simulation velocity and length scales.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.solver!-Tuple{Poisson}","page":"WaterLily","title":"WaterLily.solver!","text":"solver!(A::Poisson;log,tol,itmx)\n\nApproximate iterative solver for the Poisson matrix equation Ax=b.\n\n`A`: Poisson matrix with working arrays\n`A.x`: Solution vector. Can start with an initial guess.\n`A.z`: Right-Hand-Side vector. Will be overwritten! \n`A.n[end]`: stores the number of iterations performed.\n`log`: If `true`, this function returns a vector holding the `L₂`-norm of the residual at each iteration.\n`tol`: Convergence tolerance on the `L₂`-norm residual.\n'itmx': Maximum number of iterations\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.δ-Union{Tuple{N}, Tuple{Any, Val{N}}} where N","page":"WaterLily","title":"WaterLily.δ","text":"δ(i,N::Int)\nδ(i,I::CartesianIndex{N}) where {N}\n\nReturn a CartesianIndex of dimension N which is one at index i and zero elsewhere.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.@inside-Tuple{Any}","page":"WaterLily","title":"WaterLily.@inside","text":"@inside <expr>\n\nSimple macro to automate efficient loops over cells excluding ghosts. For example\n\n@inside p[I] = sum(loc(0,I))\n\nbecomes\n\n@loop p[I] = sum(loc(0,I)) over I ∈ inside(p)\n\nSee @loop.\n\n\n\n\n\n","category":"macro"},{"location":"#WaterLily.NoBody","page":"WaterLily","title":"WaterLily.NoBody","text":"NoBody\n\nUse for a simulation without a body\n\n\n\n\n\n","category":"type"},{"location":"#WaterLily.Jacobi!-Tuple{Any}","page":"WaterLily","title":"WaterLily.Jacobi!","text":"Jacobi!(p::Poisson; it=1)\n\nJacobi smoother run it times.  Note: This runs for general backends, but is very slow to converge.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.curl-Tuple{Any, Any, Any}","page":"WaterLily","title":"WaterLily.curl","text":"curl(i,I,u)\n\nCompute component i of ∇×u at the edge of cell I. For example curl(3,CartesianIndex(2,2,2),u) will compute ω₃(x=1.5,y=1.5,z=2) as this edge produces the highest accuracy for this mix of cross derivatives on a staggered grid.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.curvature-Tuple{AbstractMatrix}","page":"WaterLily","title":"WaterLily.curvature","text":"curvature(A::AbstractMatrix)\n\nReturn H,K the mean and Gaussian curvature from A=hessian(sdf). K=tr(minor(A)) in 3D and K=0 in 2D.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.inside_u-Union{Tuple{N}, Tuple{Tuple{Vararg{T, N}} where T, Any}} where N","page":"WaterLily","title":"WaterLily.inside_u","text":"inside_u(dims,j)\n\nReturn CartesianIndices range excluding the ghost-cells on the boundaries of a vector array on face j with size dims.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ke-Union{Tuple{m}, Tuple{CartesianIndex{m}, Any}, Tuple{CartesianIndex{m}, Any, Any}} where m","page":"WaterLily","title":"WaterLily.ke","text":"ke(I::CartesianIndex,u,U=0)\n\nCompute ½|u-U|² at center of cell I where U can be used to subtract a background flow.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.measure_sdf!","page":"WaterLily","title":"WaterLily.measure_sdf!","text":"measure_sdf!(a::AbstractArray, body::AbstractBody, t=0)\n\nUses sdf(body,x,t) to fill a.\n\n\n\n\n\n","category":"function"},{"location":"#WaterLily.pcg!-Tuple{Poisson}","page":"WaterLily","title":"WaterLily.pcg!","text":"pcg!(p::Poisson; it=6)\n\nConjugate-Gradient smoother with Jacobi preditioning. Runs at most it iterations,  but will exit early if the Gram-Smit update parameter |α|<1% or |rD⁻¹r|<1e-8. Note: This runs for general backends and is the default smoother.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.sdf-Tuple{AutoBody, Any, Any}","page":"WaterLily","title":"WaterLily.sdf","text":"d = sdf(body::AutoBody,x,t) = body.sdf(x,t)\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.slice-Union{Tuple{N}, Tuple{Tuple{Vararg{T, N}} where T, Any, Any}, Tuple{Tuple{Vararg{T, N}} where T, Any, Any, Any}} where N","page":"WaterLily","title":"WaterLily.slice","text":"slice(dims,i,j,low=1,trim=0)\n\nReturn CartesianIndices range slicing through an array of size dims in dimension j at index i. low optionally sets the lower extent of the range in the other dimensions.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.λ₂-Tuple{CartesianIndex{3}, Any}","page":"WaterLily","title":"WaterLily.λ₂","text":"λ₂(I::CartesianIndex{3},u)\n\nλ₂ is a deformation tensor metric to identify vortex cores. See https://en.wikipedia.org/wiki/Lambda2_method and Jeong, J., & Hussain, F. doi:10.1017/S0022112095000462\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ω-Tuple{CartesianIndex{3}, Any}","page":"WaterLily","title":"WaterLily.ω","text":"ω(I::CartesianIndex{3},u)\n\nCompute 3-vector ω=∇×u at the center of cell I.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ω_mag-Tuple{CartesianIndex{3}, Any}","page":"WaterLily","title":"WaterLily.ω_mag","text":"ω_mag(I::CartesianIndex{3},u)\n\nCompute |ω| at the center of cell I.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.ω_θ-Tuple{CartesianIndex{3}, Any, Any, Any}","page":"WaterLily","title":"WaterLily.ω_θ","text":"ω_θ(I::CartesianIndex{3},z,center,u)\n\nCompute ω⋅θ at the center of cell I where θ is the azimuth direction around vector z passing through center.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.∂-NTuple{4, Any}","page":"WaterLily","title":"WaterLily.∂","text":"∂(i,j,I,u)\n\nCompute ∂uᵢ/∂xⱼ at center of cell I. Cross terms are computed less accurately than inline terms because of the staggered grid.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.∮nds-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, N}, AbstractArray{T}, AutoBody}, Tuple{AbstractArray{T, N}, AbstractArray{T}, AutoBody, Any}} where {T, N}","page":"WaterLily","title":"WaterLily.∮nds","text":"∮nds(p,body::AutoBody,t=0)\n\nSurface normal integral of field p over the body.\n\n\n\n\n\n","category":"method"},{"location":"#WaterLily.@loop-Tuple","page":"WaterLily","title":"WaterLily.@loop","text":"@loop <expr> over <I ∈ R>\n\nMacro to automate fast CPU or GPU loops using KernelAbstractions.jl. The macro creates a kernel function from the expression <expr> and evaluates that function over the CartesianIndices I ∈ R.\n\nFor example\n\n@loop a[I,i] += sum(loc(i,I)) over I ∈ R\n\nbecomes\n\n@kernel function kern(a,i,@Const(I0))\n    I ∈ @index(Global,Cartesian)+I0\n    a[I,i] += sum(loc(i,I))\nend\nkern(get_backend(a),64)(a,i,R[1]-oneunit(R[1]),ndrange=size(R))\n\nwhere get_backend is used on the first variable in expr (a in this example).\n\n\n\n\n\n","category":"macro"}]
}
