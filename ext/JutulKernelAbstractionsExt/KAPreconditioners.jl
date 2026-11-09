module KAPreconditionersKernels

using Jutul
using KernelAbstractions
using LinearAlgebra, SparseArrays, StaticArrays
import Adapt, ForwardDiff
import Jutul: factorize_linear_system, refactorize_linear_system!, prepare_host_transfer!, backend_copyto!, transfer_csr_to_backend
import Jutul.KAPreconditioners:
    axpy_kernel!,
    axpy_kernel_at_index!,
    copy_kernel!,
    copy_kernel_at_index!,
    csr_to_dense_kernel!,
    csr_to_dense_kernel_at_index!,
    dilu_factor_level_kernel!,
    dilu_factor_level_kernel_at_index!,
    dilu_lower_level_kernel!,
    dilu_lower_level_kernel_at_index!,
    dilu_smooth_lower_level_kernel!,
    dilu_smooth_lower_level_kernel_at_index!,
    dilu_smooth_upper_level_kernel!,
    dilu_smooth_upper_level_kernel_at_index!,
    dilu_upper_level_kernel!,
    dilu_upper_level_kernel_at_index!,
    fill_kernel!,
    fill_kernel_at_index!,
    galerkin_kernel!,
    galerkin_kernel_at_index!,
    ilu0_factor_level_kernel!,
    ilu0_factor_level_kernel_at_index!,
    ilu0_lower_level_kernel!,
    ilu0_lower_level_kernel_at_index!,
    ilu0_smooth_lower_level_kernel!,
    ilu0_smooth_lower_level_kernel_at_index!,
    ilu0_smooth_upper_level_kernel!,
    ilu0_smooth_upper_level_kernel_at_index!,
    ilu0_upper_level_kernel!,
    ilu0_upper_level_kernel_at_index!,
    ilu_gather_kernel!,
    ilu_gather_kernel_at_index!,
    ilu_scatter_kernel!,
    ilu_scatter_kernel_at_index!,
    interpolation_product_kernel!,
    interpolation_product_kernel_at_index!,
    pivoted_sparse_lu_factor_kernel!,
    pivoted_sparse_lu_factor_kernel_at_index!,
    pivoted_sparse_lu_solve_kernel!,
    pivoted_sparse_lu_solve_kernel_at_index!,
    prolong_kernel!,
    prolong_kernel_at_index!,
    prolong_to_kernel!,
    prolong_to_kernel_at_index!,
    residual_kernel!,
    residual_kernel_at_index!,
    restrict_kernel!,
    restrict_kernel_at_index!,
    spai0_apply_kernel!,
    spai0_apply_kernel_at_index!,
    spai0_residual_step_kernel!,
    spai0_residual_step_kernel_at_index!,
    spai0_setup_kernel!,
    spai0_setup_kernel_at_index!,
    spai0_step_kernel!,
    spai0_step_kernel_at_index!,
    spai0_zero_residual_step_kernel!,
    spai0_zero_residual_step_kernel_at_index!,
    sparse_lu_factor_kernel!,
    sparse_lu_factor_kernel_at_index!,
    sparse_lu_solve_kernel!,
    sparse_lu_solve_kernel_at_index!,
    spmv_kernel!,
    spmv_kernel_at_index!,
    strength_kernel!,
    strength_kernel_at_index!,
    truncate_product_kernel!,
    truncate_product_kernel_at_index!,
    update_interpolation_p_kernel!,
    update_interpolation_p_kernel_at_index!,
    update_partial_extended_p_kernel!,
    update_partial_extended_p_kernel_at_index!,
    zero_spai0_residual_kernel!,
    zero_spai0_residual_kernel_at_index!
using Jutul: StaticSparsityMatrixCSR, KAPreconditioners

@kernel function ka_update_partial_extended_p_kernel!(
        pv, @Const(prp), @Const(pcv), @Const(rows),
        @Const(arp), @Const(acv), @Const(av),
        @Const(cf), @Const(cmap), @Const(strong), rescale, n
    )
    update_partial_extended_p_kernel_at_index!(@index(Global), pv, prp, pcv, rows, arp, acv, av, cf, cmap, strong, rescale, n)
end

@kernel function ka_interpolation_product_kernel!(
        values, @Const(p1), @Const(p2),
        @Const(offsets), @Const(left), @Const(right), n
    )
    interpolation_product_kernel_at_index!(@index(Global), values, p1, p2, offsets, left, right, n)
end

@kernel function ka_truncate_product_kernel!(
        values, @Const(rp), @Const(full_values),
        @Const(full_rp), @Const(retained), rescale, n
    )
    truncate_product_kernel_at_index!(@index(Global), values, rp, full_values, full_rp, retained, rescale, n)
end

@kernel function ka_copy_kernel!(dst, @Const(src), n)
    copy_kernel_at_index!(@index(Global), dst, src, n)
end

@kernel function ka_fill_kernel!(dst, value, n)
    fill_kernel_at_index!(@index(Global), dst, value, n)
end

@kernel function ka_spmv_kernel!(y, @Const(rp), @Const(cv), @Const(av), @Const(x), n)
    spmv_kernel_at_index!(@index(Global), y, rp, cv, av, x, n)
end

@kernel function ka_residual_kernel!(r, @Const(b), @Const(x), @Const(rp), @Const(cv), @Const(av), n)
    residual_kernel_at_index!(@index(Global), r, b, x, rp, cv, av, n)
end

@kernel function ka_strength_kernel!(
        strong, @Const(rp), @Const(cv), @Const(av),
        theta, max_row_sum, strength_mode, n
    )
    strength_kernel_at_index!(@index(Global), strong, rp, cv, av, theta, max_row_sum, strength_mode, n)
end

@kernel function ka_csr_to_dense_kernel!(dense, @Const(rp), @Const(cv), @Const(av), n)
    csr_to_dense_kernel_at_index!(@index(Global), dense, rp, cv, av, n)
end

@kernel function ka_restrict_kernel!(
        bc, @Const(offsets), @Const(rows), @Const(pidx),
        @Const(pval), @Const(r), n
    )
    restrict_kernel_at_index!(@index(Global), bc, offsets, rows, pidx, pval, r, n)
end

@kernel function ka_prolong_kernel!(x, @Const(rp), @Const(cv), @Const(pv), @Const(xc), n)
    prolong_kernel_at_index!(@index(Global), x, rp, cv, pv, xc, n)
end

@kernel function ka_prolong_to_kernel!(
        dst, @Const(src), @Const(rp), @Const(cv),
        @Const(pv), @Const(xc), n
    )
    prolong_to_kernel_at_index!(@index(Global), dst, src, rp, cv, pv, xc, n)
end

@kernel function ka_galerkin_kernel!(
        ac, @Const(a), @Const(p), @Const(offsets),
        @Const(pl), @Const(ai), @Const(pr), n
    )
    galerkin_kernel_at_index!(@index(Global), ac, a, p, offsets, pl, ai, pr, n)
end

@kernel function ka_update_interpolation_p_kernel!(
        pv, @Const(prp), @Const(pcv),
        @Const(arp), @Const(acv), @Const(av),
        @Const(cf), @Const(cmap), @Const(strong),
        extended, rescale, n
    )
    update_interpolation_p_kernel_at_index!(@index(Global), pv, prp, pcv, arp, acv, av, cf, cmap, strong, extended, rescale, n)
end

@kernel function ka_sparse_lu_factor_kernel!(
        factors, source, rowptr, colval, diagonal, source_to_factor,
        status, nnz_source, n
    )
    sparse_lu_factor_kernel_at_index!(@index(Global), factors, source, rowptr, colval, diagonal, source_to_factor, status, nnz_source, n)
end

@kernel function ka_pivoted_sparse_lu_factor_kernel!(
        factors, pivots, status, source, rowptr, colval, n
    )
    pivoted_sparse_lu_factor_kernel_at_index!(@index(Global), factors, pivots, status, source, rowptr, colval, n)
end

@kernel function ka_sparse_lu_solve_kernel!(
        output, rhs, factors, rowptr, colval, diagonal, n
    )
    sparse_lu_solve_kernel_at_index!(@index(Global), output, rhs, factors, rowptr, colval, diagonal, n)
end

@kernel function ka_pivoted_sparse_lu_solve_kernel!(output, rhs, factors, pivots, n)
    pivoted_sparse_lu_solve_kernel_at_index!(@index(Global), output, rhs, factors, pivots, n)
end

@kernel function ka_ilu0_factor_level_kernel!(
        factors, inverse_diagonal,
        @Const(rowptr), @Const(colval),
        @Const(diagonal_positions),
        @Const(rows), first, count
    )
    ilu0_factor_level_kernel_at_index!(@index(Global), factors, inverse_diagonal, rowptr, colval, diagonal_positions, rows, first, count)
end

@kernel function ka_dilu_factor_level_kernel!(
        inverse_diagonal, @Const(values),
        @Const(rowptr), @Const(colval),
        @Const(diagonal_positions),
        @Const(transpose_positions),
        @Const(rows), first, count
    )
    dilu_factor_level_kernel_at_index!(@index(Global), inverse_diagonal, values, rowptr, colval, diagonal_positions, transpose_positions, rows, first, count)
end

@kernel function ka_ilu0_lower_level_kernel!(
        work, @Const(rhs), @Const(factors),
        @Const(rowptr), @Const(colval), @Const(diagonal_positions),
        @Const(rows), first, count
    )
    ilu0_lower_level_kernel_at_index!(@index(Global), work, rhs, factors, rowptr, colval, diagonal_positions, rows, first, count)
end

@kernel function ka_ilu0_upper_level_kernel!(
        x, @Const(work), @Const(factors),
        @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval), @Const(diagonal_positions),
        damping, @Const(rows), first, count
    )
    ilu0_upper_level_kernel_at_index!(@index(Global), x, work, factors, inverse_diagonal, rowptr, colval, diagonal_positions, damping, rows, first, count)
end

@kernel function ka_dilu_lower_level_kernel!(
        work, @Const(rhs), @Const(values),
        @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval), @Const(diagonal_positions),
        @Const(rows), first, count
    )
    dilu_lower_level_kernel_at_index!(@index(Global), work, rhs, values, inverse_diagonal, rowptr, colval, diagonal_positions, rows, first, count)
end

@kernel function ka_dilu_upper_level_kernel!(
        x, @Const(work), @Const(values),
        @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval), @Const(diagonal_positions),
        damping,
        @Const(rows), first, count
    )
    dilu_upper_level_kernel_at_index!(@index(Global), x, work, values, inverse_diagonal, rowptr, colval, diagonal_positions, damping, rows, first, count)
end

@kernel function ka_ilu0_smooth_lower_level_kernel!(
        work, @Const(rhs),
        @Const(x), @Const(matrix_values), @Const(factors),
        @Const(rowptr), @Const(colval), @Const(rows), first, count
    )
    ilu0_smooth_lower_level_kernel_at_index!(@index(Global), work, rhs, x, matrix_values, factors, rowptr, colval, rows, first, count)
end

@kernel function ka_ilu0_smooth_upper_level_kernel!(
        x, correction, @Const(work),
        @Const(factors), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), @Const(diagonal_positions), damping, @Const(rows), first, count
    )
    ilu0_smooth_upper_level_kernel_at_index!(@index(Global), x, correction, work, factors, inverse_diagonal, rowptr, colval, diagonal_positions, damping, rows, first, count)
end

@kernel function ka_dilu_smooth_lower_level_kernel!(
        work, @Const(rhs),
        @Const(x), @Const(values), @Const(inverse_diagonal),
        @Const(rowptr), @Const(colval),
        @Const(rows), first, count
    )
    dilu_smooth_lower_level_kernel_at_index!(@index(Global), work, rhs, x, values, inverse_diagonal, rowptr, colval, rows, first, count)
end

@kernel function ka_dilu_smooth_upper_level_kernel!(
        x, correction, @Const(work),
        @Const(values), @Const(inverse_diagonal), @Const(rowptr),
        @Const(colval), @Const(diagonal_positions), damping,
        @Const(rows), first, count
    )
    dilu_smooth_upper_level_kernel_at_index!(@index(Global), x, correction, work, values, inverse_diagonal, rowptr, colval, diagonal_positions, damping, rows, first, count)
end

@kernel function ka_axpy_kernel!(x, @Const(y), alpha, n)
    axpy_kernel_at_index!(@index(Global), x, y, alpha, n)
end

@kernel function ka_ilu_gather_kernel!(destination, @Const(source), @Const(positions), n)
    ilu_gather_kernel_at_index!(@index(Global), destination, source, positions, n)
end

@kernel function ka_ilu_scatter_kernel!(destination, @Const(source), @Const(positions), n)
    ilu_scatter_kernel_at_index!(@index(Global), destination, source, positions, n)
end

@kernel function ka_spai0_setup_kernel!(d, @Const(rp), @Const(cv), @Const(av), damping, n)
    spai0_setup_kernel_at_index!(@index(Global), d, rp, cv, av, damping, n)
end

@kernel function ka_spai0_step_kernel!(
        dst, @Const(src), @Const(b), @Const(d),
        @Const(rp), @Const(cv), @Const(av), n
    )
    spai0_step_kernel_at_index!(@index(Global), dst, src, b, d, rp, cv, av, n)
end

@kernel function ka_spai0_apply_kernel!(x, @Const(b), @Const(d), n)
    spai0_apply_kernel_at_index!(@index(Global), x, b, d, n)
end

@kernel function ka_spai0_residual_step_kernel!(
        dst, @Const(src), @Const(residual),
        @Const(d), n
    )
    spai0_residual_step_kernel_at_index!(@index(Global), dst, src, residual, d, n)
end

@kernel function ka_spai0_zero_residual_step_kernel!(
        dst, @Const(residual),
        @Const(d), n
    )
    spai0_zero_residual_step_kernel_at_index!(@index(Global), dst, residual, d, n)
end

@kernel function ka_zero_spai0_residual_kernel!(
        x, residual, @Const(b), @Const(d),
        @Const(rp), @Const(cv),
        @Const(av), n
    )
    zero_spai0_residual_kernel_at_index!(@index(Global), x, residual, b, d, rp, cv, av, n)
end

update_partial_extended_p_kernel!(backend::KernelAbstractions.Backend, args...) = ka_update_partial_extended_p_kernel!(backend, args...)
interpolation_product_kernel!(backend::KernelAbstractions.Backend, args...) = ka_interpolation_product_kernel!(backend, args...)
truncate_product_kernel!(backend::KernelAbstractions.Backend, args...) = ka_truncate_product_kernel!(backend, args...)
copy_kernel!(backend::KernelAbstractions.Backend, args...) = ka_copy_kernel!(backend, args...)
fill_kernel!(backend::KernelAbstractions.Backend, args...) = ka_fill_kernel!(backend, args...)
spmv_kernel!(backend::KernelAbstractions.Backend, args...) = ka_spmv_kernel!(backend, args...)
residual_kernel!(backend::KernelAbstractions.Backend, args...) = ka_residual_kernel!(backend, args...)
strength_kernel!(backend::KernelAbstractions.Backend, args...) = ka_strength_kernel!(backend, args...)
csr_to_dense_kernel!(backend::KernelAbstractions.Backend, args...) = ka_csr_to_dense_kernel!(backend, args...)
restrict_kernel!(backend::KernelAbstractions.Backend, args...) = ka_restrict_kernel!(backend, args...)
prolong_kernel!(backend::KernelAbstractions.Backend, args...) = ka_prolong_kernel!(backend, args...)
prolong_to_kernel!(backend::KernelAbstractions.Backend, args...) = ka_prolong_to_kernel!(backend, args...)
galerkin_kernel!(backend::KernelAbstractions.Backend, args...) = ka_galerkin_kernel!(backend, args...)
update_interpolation_p_kernel!(backend::KernelAbstractions.Backend, args...) = ka_update_interpolation_p_kernel!(backend, args...)
sparse_lu_factor_kernel!(backend::KernelAbstractions.Backend, args...) = ka_sparse_lu_factor_kernel!(backend, args...)
pivoted_sparse_lu_factor_kernel!(backend::KernelAbstractions.Backend, args...) = ka_pivoted_sparse_lu_factor_kernel!(backend, args...)
sparse_lu_solve_kernel!(backend::KernelAbstractions.Backend, args...) = ka_sparse_lu_solve_kernel!(backend, args...)
pivoted_sparse_lu_solve_kernel!(backend::KernelAbstractions.Backend, args...) = ka_pivoted_sparse_lu_solve_kernel!(backend, args...)
ilu0_factor_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu0_factor_level_kernel!(backend, args...)
dilu_factor_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_dilu_factor_level_kernel!(backend, args...)
ilu0_lower_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu0_lower_level_kernel!(backend, args...)
ilu0_upper_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu0_upper_level_kernel!(backend, args...)
dilu_lower_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_dilu_lower_level_kernel!(backend, args...)
dilu_upper_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_dilu_upper_level_kernel!(backend, args...)
ilu0_smooth_lower_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu0_smooth_lower_level_kernel!(backend, args...)
ilu0_smooth_upper_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu0_smooth_upper_level_kernel!(backend, args...)
dilu_smooth_lower_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_dilu_smooth_lower_level_kernel!(backend, args...)
dilu_smooth_upper_level_kernel!(backend::KernelAbstractions.Backend, args...) = ka_dilu_smooth_upper_level_kernel!(backend, args...)
axpy_kernel!(backend::KernelAbstractions.Backend, args...) = ka_axpy_kernel!(backend, args...)
ilu_gather_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu_gather_kernel!(backend, args...)
ilu_scatter_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ilu_scatter_kernel!(backend, args...)
spai0_setup_kernel!(backend::KernelAbstractions.Backend, args...) = ka_spai0_setup_kernel!(backend, args...)
spai0_step_kernel!(backend::KernelAbstractions.Backend, args...) = ka_spai0_step_kernel!(backend, args...)
spai0_apply_kernel!(backend::KernelAbstractions.Backend, args...) = ka_spai0_apply_kernel!(backend, args...)
spai0_residual_step_kernel!(backend::KernelAbstractions.Backend, args...) = ka_spai0_residual_step_kernel!(backend, args...)
spai0_zero_residual_step_kernel!(backend::KernelAbstractions.Backend, args...) = ka_spai0_zero_residual_step_kernel!(backend, args...)
zero_spai0_residual_kernel!(backend::KernelAbstractions.Backend, args...) = ka_zero_spai0_residual_kernel!(backend, args...)

end
