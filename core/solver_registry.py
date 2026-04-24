from ripple.physics.operators import TimeDerivative, Diffusion, Advection, Laplacian, Nonlinear, Source

def _extract_alpha(eq):
    alphas = [op.alpha for _, op in eq.terms if isinstance(op, Diffusion)]
    return alphas[0] if alphas else 0.0

def _extract_v(eq):
    vs = [op.v for _, op in eq.terms if isinstance(op, Advection)]
    return vs[0] if vs else 0.0

def _extract_beta(eq):
    betas = [coeff for coeff, op in eq.terms if isinstance(op, TimeDerivative) and op.order == 1]
    return betas[0] if betas else 0.0

def _extract_c(eq):
    c_sqs = [-coeff for coeff, op in eq.terms if isinstance(op, Laplacian)]
    return c_sqs[0]**0.5 if c_sqs else 1.0


def get_solver(equation):
    """Returns (solver_fn, extra_kwargs)."""
    from ripple.solvers import fd_solver
    
    key = select_solver(equation)
    
    if key == "wave":
        return fd_solver.solve_wave_fd_1d, lambda eq: {"c": _extract_c(eq)}
    if key == "advdiff":
        return fd_solver.solve_advdiff_fd_1d, lambda eq: {"alpha": _extract_alpha(eq), "v": _extract_v(eq)}
    if key == "diffusion":
        return fd_solver.solve_diffusion_fd_1d, lambda eq: {"alpha": _extract_alpha(eq)}
    if key == "advection":
        return fd_solver.solve_advection_fd_1d, lambda eq: {"v": _extract_v(eq)}
    if key == "reaction_diffusion":
        return fd_solver.solve_reaction_diffusion_fd_1d, lambda eq: {"alpha": _extract_alpha(eq), "equation": eq}
    if key == "first_order_nonlinear":
        return fd_solver.solve_reaction_diffusion_fd_1d, lambda eq: {"alpha": 0.0, "equation": eq}
    if key == "damped_wave":
        return fd_solver.solve_damped_wave_fd_1d, lambda eq: {"beta": _extract_beta(eq), "c": _extract_c(eq)}
    
    raise NotImplementedError(f"No FD solver for key: {key}")


def select_solver(equation) -> str:
    """Inspect equation.terms and return a solver key.

    Returns one of: 'wave', 'diffusion', 'advection', 'advdiff'
    Raises NotImplementedError for unrecognised configurations.
    """
    terms = equation.terms
    has_t2   = any(isinstance(op, TimeDerivative) and op.order == 2 for _, op in terms)
    has_t1   = any(isinstance(op, TimeDerivative) and op.order == 1 for _, op in terms)
    has_diff = any(isinstance(op, Diffusion)      for _, op in terms)
    has_adv  = any(isinstance(op, Advection)      for _, op in terms)
    has_non  = any(isinstance(op, (Nonlinear, Source)) for _, op in terms)

    if has_t2 and has_t1:
        return "damped_wave"
    if has_t2:
        return "wave"
    if has_t1 and has_diff and has_adv:
        return "advdiff"
    if has_t1 and has_diff and has_non:
        return "reaction_diffusion"
    if has_t1 and has_non:
        return "first_order_nonlinear"
    if has_t1 and has_diff:
        return "diffusion"
    if has_t1 and has_adv:
        return "advection"

    ops = [type(op).__name__ for _, op in terms]
    raise NotImplementedError(f"No solver for operators: {ops}")
