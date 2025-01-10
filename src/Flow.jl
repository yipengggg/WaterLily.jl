@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]        #对于标量场的任意位置的任意方向的梯度计算，标量场可以是：密度，质量，温度，浓度等
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]  #对于速度场的任意位置的任意方向的梯度计算
@inline ϕ(a,I,f) = @inbounds (f[I]+f[I-δ(a,I)])*0.5                                #一种插值方法，用于给扩散项插值，与旁边的点取均值，来求扩散标量的变化
@fastmath quick(u,c,d) = median((5c+2d-u)/6,c,median(10c-9u,c,d))                  #一种插值方法，用于给对流项插值，表示二阶插值，f_i-2,f_i-1,f_i之间的关系，二阶插值相比一阶插值会提高精度，但是导致数据不稳定（振荡）
@fastmath vanLeer(u,c,d) = (c≤min(u,d) || c≥max(u,d)) ? c : c+(d-c)*(c-u)/(d-u)    #一种插值方法，用于给对流项插值，避免数据振荡，即数据上下波动
@inline ϕu(a,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)]) # 对流通量表示为flux：J， J的计算为 ϕ乘u，这里是通过标量和速度相乘来表示 通量
#计算 对流 通量，基于速度u和物理量phi,phi通过插值方式获得       当速度大于0，用后边的表达式表示标量，相乘得到通量               当速度小于0，用后边的表达式表示标量，相乘得到通量 
@inline ϕuP(a,Ip,I,f,u,λ=quick) = @inbounds u>0 ? u*λ(f[Ip],f[I-δ(a,I)],f[I]) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])  #当确定为周期性边界条件时，使用这个函数来计算周期边界的对流通量flux
@inline ϕuL(a,I,f,u,λ=quick) = @inbounds u>0 ? u*ϕ(a,I,f) : u*λ(f[I+δ(a,I)],f[I],f[I-δ(a,I)])   #处理常规情况下左侧边界的对流通量flux计算
@inline ϕuR(a,I,f,u,λ=quick) = @inbounds u<0 ? u*ϕ(a,I,f) : u*λ(f[I-2δ(a,I)],f[I-δ(a,I)],f[I])  ##处理常规情况下右侧边界的对流通量flux计算

#在julia里定义变量的方式类似于fortran，在一个字母后边施加四个点 :: 来解释前边这个字符的含义

@fastmath @inline function div(I::CartesianIndex{m},u) where {m}                    #对任一位置标量的散度计算
    init=zero(eltype(u))                                                            #对任意位置上的点的标量，进行方向上的遍历，然后将不同方向的梯度进行累加，返回的init就是 divergence
    for i in 1:m                        
     init += @inbounds ∂(i,I,u)
    end
    return init
end

@fastmath @inline function μddn(I::CartesianIndex{np1},μ,f) where np1              #计算任意三个数的中间值，之后的插值计算会用到
    s = zero(eltype(f))
    for j ∈ 1:np1-1
        s+= @inbounds μ[I,j]*(f[I+δ(j,I)]-f[I-δ(j,I)])
    end
    return 0.5s
end
function median(a,b,c)
    if a>b
        b>=c && return b
        a>c && return c
    else
        b<=c && return b
        a<c && return c
    end
    return a
end

function conv_diff!(r,u,Φ;ν=0.1,perdir=())                                         #计算标量在模拟网格中从初始网格向外扩张的物理过程，如浓度的扩散，温度的传递，流畅内速度分类的传递
    r .= 0.                                                                        #输出数组，在这里表示对应网格内所有位置的通量flux，这里的r表示的是综合通量变量，不仅仅有对流和扩散项，在下边还会有各种加速度带来的效果
    N,n = size_u(u)                                                                #定义标量场的网格尺寸信心
    for i ∈ 1:n, j ∈ 1:n                                                         #i的循环表示速度的不同分量：u,v,w, j的循环表示对不同速度分量进行通量计算x,y,z方向，这个通量为不同方向上的
        # if it is periodic direction
        tagper = (j in perdir)                                                     #判断j所代表的x,y,z方向是否为周期性方向，是则tagper = true, 不是则tapger = false
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,Φ,ν,i,j,N,Val{tagper}())                                #根据tapger的值，对下边界用neumann边界还是周期性边界进行判断，即流入边界
        # inner cells
        @loop (Φ[I] = ϕu(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν*∂(j,CI(I,i),u);           #计算I点的通量
               #I点储存的通量         #计算对流通量                    #计算扩散通量
               r[I,i] += Φ[I]) over I ∈ inside_u(N,j)                             #将通量的矢量和记录到r的对应位置
        @loop r[I-δ(j,I),i] -= Φ[I] over I ∈ inside_u(N,j)                        #确保通量守恒，通量是从一个格子流入到周围格子的，如果在j方向上上一行代码在I出j方向上加入通量Φ，那么在I沿j方向倒退一步的地方要减去通量Φ，确保通量守恒
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,Φ,ν,i,j,N,Val{tagper}())                                 #根据tapger的值，对上边界用neumann边界还是周期性边界进行判断，即流出边界
    end
end

# Neumann BC Building block                                                         # Neumann边界条件表示边界没渗出
lowerBoundary!(r,u,Φ,ν,i,j,N,::Val{false}) = @loop r[I,i] += ϕuL(j,CI(I,i),u,ϕ(i,CI(I,j),u)) - ν*∂(j,CI(I,i),u) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,::Val{false}) = @loop r[I-δ(j,I),i] += -ϕuR(j,CI(I,i),u,ϕ(i,CI(I,j),u)) + ν*∂(j,CI(I,i),u) over I ∈ slice(N,N[j],j,2)

# Periodic BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,::Val{true}) = @loop (                                 # 周期条件表示上边界速度等于下边界速度
    Φ[I] = ϕuP(j,CIj(j,CI(I,i),N[j]-2),CI(I,i),u,ϕ(i,CI(I,j),u)) -ν*∂(j,CI(I,i),u); r[I,i] += Φ[I]) over I ∈ slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,::Val{true}) = @loop r[I-δ(j,I),i] -= Φ[CIj(j,I,2)] over I ∈ slice(N,N[j],j,2)

using EllipsisNotation                                                                                       #一个 Julia 包，它主要用于高效和简洁地操作多维数组，允许你使用 .. 来表示“所有其他维度” 例如: A[:, :, :, k]

"""
    accelerate!(r,dt,g)

Add a uniform acceleration `gᵢ+dUᵢ/dt` at time `t=sum(dt)` to field `r`.                                          #添加一个多情况函数，表示不同情况下的加速度作用，并将其效果添加到输入数组r
"""
accelerate!(r,dt,g::Function,::Tuple,t=sum(dt)) = for i ∈ 1:last(size(r))                                         #这个情况下的g是被直接定义的函数，将这个直接定义的函数的效果叠加到r上
    r[..,i] .+= g(i,t)
end
accelerate!(r,dt,g::Nothing,U::Function) = accelerate!(r,dt,(i,t)->ForwardDiff.derivative(τ->U(i,τ),t),())        #这个情况下g不存在，即没有直接定义的加速度函数，而有速度场函数，所以通过对速度场函数求导，然后将加速度加入r
accelerate!(r,dt,g::Function,U::Function) = accelerate!(r,dt,(i,t)->g(i,t)+ForwardDiff.derivative(τ->U(i,τ),t),())#同时存在g和u，则在每一个点将所有g和u的导数的效果同时叠加
accelerate!(r,dt,::Nothing,::Tuple) = nothing                                                                     #没有任何的加速度和速度场的情况下，则r直接就是conv-diff的结果，不施加任何额外的操作
"""
    BCTuple(U,dt,N)

Return BC tuple `U(i∈1:N, t=sum(dt))`.
"""
BCTuple(f::Function,dt,N,t=sum(dt))=ntuple(i->f(i,t),N)                                                           #用来计算边界的值并生产一个tuple，边界上flux的值，在我们的情况下
BCTuple(f::Tuple,dt,N)=f                                                                                          #如果已经有了f为定值，则直接用

"""
    Flow{D::Int, T::Float, Sf<:AbstractArray{T,D}, Vf<:AbstractArray{T,D+1}, Tf<:AbstractArray{T,D+2}}

Composite type for a multidimensional immersed boundary flow simulation.

Flow solves the unsteady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) on a Cartesian grid.
Solid boundaries are modelled using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/).
The primary variables are the scalar pressure `p` (an array of dimension `D`)
and the velocity vector field `u` (an array of dimension `D+1`).
"""
struct Flow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}}
#定义参数类型：D，空间维度，T，数据精度，sf，压力场标量， vf，速度矢量场， tf，张量场
    # Fluid fields
    u :: Vf # velocity vector field 当前速度场
    u⁰:: Vf # previous velocity     之前速度场
    f :: Vf # force vector
    p :: Sf # pressure scalar field
    σ :: Sf # divergence scalar。  用来强制不可压缩，（使得散度为零）
    # BDIM fields                用来处理流体和固体边界交互的情况
    V :: Vf # body velocity vector
    μ₀:: Vf # zeroth-moment vector。  描述边界的体积效应，如密度和质量
    μ₁:: Tf # first-moment tensor field。描述边界的动量效应，如力和速度的分布
    # Non-fields
    U :: Union{NTuple{D,Number},Function} # domain boundary values
    Δt:: Vector{T} # time step (stored in CPU memory)
    ν :: T # kinematic viscosity
    g :: Union{Function,Nothing} # (possibly time-varying) uniform acceleration field
    exitBC :: Bool # Convection exit
    perdir :: NTuple # tuple of periodic direction
    function Flow(N::NTuple{D}, U; f=Array, Δt=0.25, ν=0., g=nothing,
                  uλ::Function=(i, x) -> 0., perdir=(), exitBC=false, T=Float64) where D
        Ng = N .+ 2
        Nd = (Ng..., D)
        u = Array{T}(undef, Nd...) |> f; apply!(uλ, u);
        BC!(u,BCTuple(U,0.,D),exitBC,perdir); exitBC!(u,u,BCTuple(U,0.,D),0.)
        u⁰ = copy(u);
        fv, p, σ = zeros(T, Nd) |> f, zeros(T, Ng) |> f, zeros(T, Ng) |> f
        V, μ₀, μ₁ = zeros(T, Nd) |> f, ones(T, Nd) |> f, zeros(T, Ng..., D, D) |> f
        BC!(μ₀,ntuple(zero, D),false,perdir)
        new{D,T,typeof(p),typeof(u),typeof(μ₁)}(u,u⁰,fv,p,σ,V,μ₀,μ₁,U,T[Δt],ν,g,exitBC,perdir)
    end
end

"""
    time(a::Flow)

Current flow time.
"""
time(a::Flow) = sum(@view(a.Δt[1:end-1]))

function BDIM!(a::Flow)                    #更新流体和固体交互边界的速度场
    dt = a.Δt[end]
    @loop a.f[Ii] = a.u⁰[Ii]+dt*a.f[Ii]-a.V[Ii] over Ii in CartesianIndices(a.f)                        #第一个循环，根据 历史速度 和 固体边界影响，修正力场
    @loop a.u[Ii] += μddn(Ii,a.μ₁,a.f)+a.V[Ii]+a.μ₀[Ii]*a.f[Ii] over Ii ∈ inside_u(size(a.p))           #第二个循环, 根据 修正力场， 更新流体速度场
end

function project!(a::Flow{n},b::AbstractPoisson,w=1) where n                        #压力修正，压力柏松方程的求解， 柏松方程即用ns方程求解速度和压强的关系，然后对压强求积分，得到速度和力的关系
    dt = w*a.Δt[end]                                                                #确保速度场imcompressible
    @inside b.z[I] = div(I,a.u); b.x .*= dt # set source term & solution IC
    solver!(b)
    for i ∈ 1:n  # apply solution and unscale to recover pressure
        @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x)
    end
    b.x ./= dt
end

"""
    mom_step!(a::Flow,b::AbstractPoisson)

Integrate the `Flow` one time step using the [Boundary Data Immersion Method](https://eprints.soton.ac.uk/369635/)
and the `AbstractPoisson` pressure solver to project the velocity onto an incompressible flow.
"""
@fastmath function mom_step!(a::Flow{N},b::AbstractPoisson) where N            #分布逼近真实解， 预测阶段估算粗略值，矫正阶段计算精确值
    a.u⁰ .= a.u; scale_u!(a,0); U = BCTuple(a.U,a.Δt,N)
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,ν=a.ν,perdir=a.perdir)
    accelerate!(a.f,@view(a.Δt[1:end-1]),a.g,a.U)
    BDIM!(a); BC!(a.u,U,a.exitBC,a.perdir)
    a.exitBC && exitBC!(a.u,a.u⁰,U,a.Δt[end]) # convective exit
    project!(a,b); BC!(a.u,U,a.exitBC,a.perdir)
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,ν=a.ν,perdir=a.perdir)
    accelerate!(a.f,a.Δt,a.g,a.U)
    BDIM!(a); scale_u!(a,0.5); BC!(a.u,U,a.exitBC,a.perdir)
    project!(a,b,0.5); BC!(a.u,U,a.exitBC,a.perdir)
    push!(a.Δt,CFL(a))
end
scale_u!(a,scale) = @loop a.u[Ii] *= scale over Ii ∈ inside_u(size(a.p))

function CFL(a::Flow;Δt_max=10)                                                #确定delta T的具体大小为多少，得到稳定的时间步长
    @inside a.σ[I] = flux_out(I,a.u)
    min(Δt_max,inv(maximum(a.σ)+5a.ν))
end
@fastmath @inline function flux_out(I::CartesianIndex{d},u) where {d}          #计算任意网格点的流出通量，用来帮忙计算cfl，即步长
    s = zero(eltype(u))
    for i in 1:d
        s += @inbounds(max(0.,u[I+δ(i,I),i])+max(0.,-u[I,i]))
    end
    return s
end
