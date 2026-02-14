"""Continuous edge function for entry conditions.

Uses a smooth, monotone function (doubt compression) that requires higher model
confidence when the market is more confident, with a hard probability floor.

    required_model_prob(p) = max(floor, 1 - (1 - p)^alpha)

Parameters:
    alpha: curvature exponent (1.0 = linear, >1 = more edge at high confidence)
    floor: minimum model probability below which no trade is taken
"""


def required_model_prob(
    market_prob: float, alpha: float = 1.5, floor: float = 0.65
) -> float:
    """Compute required model probability for a given market probability.

    Args:
        market_prob: Polymarket probability (0-1)
        alpha: Curvature exponent. 1.0 = linear (model doubt = market doubt),
               >1 = demands proportionally more edge at high market confidence.
        floor: Hard minimum on model probability.

    Returns:
        Required model probability (0-1).
    """
    if market_prob <= 0.0:
        return floor
    if market_prob >= 1.0:
        return 1.0
    raw = 1.0 - (1.0 - market_prob) ** alpha
    return max(floor, raw)


def has_edge(
    model_prob: float,
    market_prob: float,
    alpha: float = 1.5,
    floor: float = 0.65,
) -> bool:
    """Check if the model probability exceeds the required threshold.

    Args:
        model_prob: Model's estimated probability (0-1)
        market_prob: Polymarket probability (0-1)
        alpha: Curvature exponent for the required-probability curve.
        floor: Hard minimum on model probability.

    Returns:
        True if model_prob >= required_model_prob(market_prob, alpha, floor).
    """
    if market_prob < floor:
        return False
    return model_prob >= required_model_prob(market_prob, alpha, floor)
