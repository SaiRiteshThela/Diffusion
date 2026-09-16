import math
from typing import Any


def trial_exploded(result: dict[str, Any], abort_train_loss: float) -> bool:
    max_train = result.get("max_train")
    if result.get("aborted"):
        return True
    if max_train is None or not math.isfinite(float(max_train)):
        return True
    return float(max_train) > abort_train_loss


def select_winner(
    results: list[dict[str, Any]],
    abort_train_loss: float = 20.0,
) -> dict[str, Any] | None:
    """Pick the stable trial with the lowest final EMA validation loss."""
    viable = [result for result in results if not trial_exploded(result, abort_train_loss)]
    if not viable:
        return None
    return min(
        viable,
        key=lambda result: (
            float(result["final_val"]),
            float(result["max_train"]),
            float(result["learning_rate"]),
        ),
    )
