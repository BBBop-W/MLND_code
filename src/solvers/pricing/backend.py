"""One pricing call signature for the label and mathematical-program backends."""

import time


class PricingStopped(RuntimeError):
    """Pricing stopped before it could certify every requested group."""


def price_design(
    data,
    *,
    backend="label",
    time_limit=None,
    stop_requested=None,
    threads=1,
    seed=43,
    **kwargs,
):
    started = time.perf_counter()
    if backend not in {"label", "solver"}:
        raise ValueError("pricing must be 'label' or 'solver'")
    if stop_requested is not None and stop_requested():
        raise PricingStopped("user_stop")
    if time_limit is not None and time_limit <= 0:
        raise PricingStopped("time_limit")
    if backend == "label":
        from src.solvers.pricing.label import price_design as price

        result = price(data, **kwargs)
    else:
        import src.solvers.pricing.solver as solver

        try:
            result = solver.price_design(
                data,
                time_limit=time_limit,
                stop_requested=stop_requested,
                threads=threads,
                seed=seed,
                **kwargs,
            )
        except solver.PricingStopped as error:
            raise PricingStopped(str(error)) from error
    if stop_requested is not None and stop_requested():
        raise PricingStopped("user_stop")
    # Label kernels are checked at call boundaries; a late result cannot certify a cut.
    if time_limit is not None and time.perf_counter() - started >= time_limit:
        raise PricingStopped("time_limit")
    result["pricing_backend"] = backend
    return result
