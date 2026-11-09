using Test, Jutul, LinearAlgebra, SparseArrays
import Pkg

@testset "Jutul without KernelAbstractions" begin
    @test !any(info.name == "KernelAbstractions" for info in values(Pkg.dependencies()))
    @test isnothing(Base.get_extension(Jutul, :JutulKernelAbstractionsExt))
    @test !any(id.name == "KernelAbstractions" for id in keys(Base.loaded_modules))

    KP = Jutul.KAPreconditioners
    A = spdiagm(-1 => fill(-1.0, 31), 0 => fill(3.0, 32), 1 => fill(-1.0, 31))
    b = ones(32)
    matrix = KP.csr_matrix(A)
    for config in (KP.SPAI0(), KP.GaussSeidel(), KP.HybridGaussSeidel(),
            KP.ILU0(), KP.DILU(), KP.ILU0(; ordering = :multicolor))
        state = KP.setup_smoother(matrix, config)
        x = zeros(32)
        KP.apply!(x, state, b)
        @test norm(b - A*x) < norm(b)
        @test KP.update_smoother!(state, KP.csr_matrix(1.1A)) === state
    end
    for T in (Float64, Float32)
        M = T.(A)
        H = KP.setup_amg(M, KP.AMGOptions(; coarse_size = 4))
        x = zeros(T, 32)
        KP.cycle!(x, H, ones(T, 32))
        @test norm(ones(T, 32) - M*x) < sqrt(T(32))
    end
    for pivoting in (true, false)
        F = KP.setup_sparse_lu(matrix; pivoting)
        x = zeros(32)
        ldiv!(x, F, b)
        @test x ≈ A \ b
        @test KP.resetup_sparse_lu!(F, KP.csr_matrix(1.1A)) === F
        ldiv!(x, F, b)
        @test x ≈ (1.1A) \ b
    end
    include("test_systems/scalar.jl")
end
