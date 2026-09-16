import pytest

from training.batch_size import align_batch_size, largest_fitting_batch, memory_within_fraction
from training.tune import select_winner, trial_exploded


def test_largest_fitting_batch_binary_search():
    calls = []

    def fits(batch_size: int) -> bool:
        calls.append(batch_size)
        return batch_size <= 176

    assert largest_fitting_batch(fits, 16, 384) == 176
    assert calls
    assert len(calls) <= 10


def test_largest_fitting_batch_rejects_empty_range():
    with pytest.raises(RuntimeError, match="no batch size"):
        largest_fitting_batch(lambda _: False, 8, 32)


def test_align_batch_size():
    assert align_batch_size(180) == 176
    assert align_batch_size(176) == 176
    with pytest.raises(RuntimeError, match="smaller than alignment"):
        align_batch_size(3)


def test_memory_within_fraction():
    total = 10 * 1024**3
    assert memory_within_fraction(int(0.88 * total), total, 0.88)
    assert not memory_within_fraction(int(0.89 * total), total, 0.88)


def test_select_winner_skips_exploded_trials():
    results = [
        {
            "learning_rate": 1e-4,
            "final_val": 0.20,
            "max_train": 80.0,
            "aborted": True,
        },
        {
            "learning_rate": 5e-5,
            "final_val": 0.31,
            "max_train": 0.4,
            "aborted": False,
        },
        {
            "learning_rate": 3e-5,
            "final_val": 0.33,
            "max_train": 0.5,
            "aborted": False,
        },
    ]
    assert trial_exploded(results[0], abort_train_loss=20.0)
    winner = select_winner(results, abort_train_loss=20.0)
    assert winner is not None
    assert winner["learning_rate"] == 5e-5


def test_select_winner_none_if_all_explode():
    results = [
        {"learning_rate": 2e-4, "final_val": 0.2, "max_train": 99.0, "aborted": True}
    ]
    assert select_winner(results, abort_train_loss=20.0) is None
