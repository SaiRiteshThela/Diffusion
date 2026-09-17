import numpy as np
import pytest
import torch

from sampling.fid_celeba import (
    activation_statistics,
    frechet_distance,
    generate_uint8,
    load_eval_weights,
    to_uint8,
)
from tests.helpers import TinyFlow


def test_to_uint8_maps_minus_one_one():
    images = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3)
    pixels = to_uint8(images)
    assert pixels.dtype == torch.uint8
    assert pixels.flatten().tolist() == [0, 128, 255]


def test_load_eval_weights_prefers_ema():
    raw = TinyFlow()
    ema = TinyFlow()
    with torch.no_grad():
        raw.conv.weight.fill_(0.1)
        ema.conv.weight.fill_(0.7)
    loaded = TinyFlow()
    load_eval_weights(
        loaded,
        {"model": raw.state_dict(), "ema_model": ema.state_dict()},
        use_ema=True,
    )
    assert torch.allclose(loaded.conv.weight, ema.conv.weight)


def test_load_eval_weights_falls_back_to_raw():
    raw = TinyFlow()
    with torch.no_grad():
        raw.conv.weight.fill_(0.3)
    loaded = TinyFlow()
    load_eval_weights(loaded, {"model": raw.state_dict(), "ema_model": None}, use_ema=True)
    assert torch.allclose(loaded.conv.weight, raw.conv.weight)


def test_generate_uint8_shape_and_dtype():
    model = TinyFlow(channels=3).eval()
    images = generate_uint8(
        model,
        num_samples=4,
        image_size=8,
        channels=3,
        ode_steps=2,
        batch_size=2,
        device=torch.device("cpu"),
        seed=0,
    )
    assert images.shape == (4, 3, 8, 8)
    assert images.dtype == torch.uint8


def test_activation_statistics_identity_covariance():
    torch.manual_seed(0)
    features = torch.eye(4).repeat(8, 1).numpy()
    mean, covariance = activation_statistics(features)
    assert mean.shape == (4,)
    assert covariance.shape == (4, 4)


def test_frechet_distance_identical_is_zero():
    mean = np.zeros(3)
    cov = np.eye(3)
    assert frechet_distance(mean, cov, mean, cov) == pytest.approx(0.0, abs=1e-6)


def test_generated_cache_identity_rejects_checkpoint_seed_and_ema_changes(tmp_path):
    from sampling.fid_celeba import generated_cache_identity

    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'first checkpoint')
    config = tmp_path / 'config.yaml'
    config.write_text('data: {image_size: 64}\n')
    kwargs = dict(num_samples=25, ode_steps=32, seed=42, use_ema=True,
                  image_size=64, channels=3, batch_size=25, precision='fp32')
    identity = generated_cache_identity(checkpoint, config, **kwargs)
    assert identity != generated_cache_identity(checkpoint, config, **(kwargs | {'seed': 43}))
    assert identity != generated_cache_identity(checkpoint, config, **(kwargs | {'use_ema': False}))
    assert identity != generated_cache_identity(checkpoint, config, **(kwargs | {'ode_steps': 100}))
    checkpoint.write_bytes(b'replaced checkpoint')
    assert identity != generated_cache_identity(checkpoint, config, **kwargs)


def test_cache_requires_matching_identity_and_unchanged_artifact(tmp_path):
    from sampling.fid_celeba import cache_matches, write_manifest

    artifact = tmp_path / 'generated.pt'
    artifact.write_bytes(b'original sample cache')
    identity = {'checkpoint': 'first', 'seed': 42}
    assert not cache_matches(artifact, identity)  # Legacy caches are never silently reused.
    write_manifest(artifact, identity)
    assert cache_matches(artifact, identity)
    assert not cache_matches(artifact, identity | {'checkpoint': 'second'})
    artifact.write_bytes(b'changed')
    assert not cache_matches(artifact, identity)


def test_deadline_interrupts_sampling_before_model_forward():
    import time
    from sampling.fid_celeba import generate_uint8

    class ShouldNotRun(torch.nn.Module):
        def forward(self, *_):
            pytest.fail('model ran after deadline')

    with pytest.raises(TimeoutError, match='deadline'):
        generate_uint8(ShouldNotRun(), 2, 4, 3, 10, 2, torch.device('cpu'), 42,
                       deadline_unix=time.time() - 1)


def test_deadline_is_checked_each_ode_step(monkeypatch):
    import sampling.fid_celeba as module

    checks = []
    def deadline(_):
        checks.append(True)
        if len(checks) == 3:
            raise TimeoutError('test step deadline')
    monkeypatch.setattr(module, 'check_deadline', deadline)
    with pytest.raises(TimeoutError, match='step deadline'):
        generate_uint8(TinyFlow(channels=3).eval(), 2, 4, 3, 10, 2,
                       torch.device('cpu'), 42, deadline_unix=1)
    assert len(checks) == 3


def test_legacy_real_stats_adoption_checks_provenance(tmp_path, monkeypatch):
    import json
    import os
    import sampling.fid_celeba as module

    cache = tmp_path / 'cache_train_4.pt'
    torch.save({'images': torch.zeros(3, 3, 4, 4, dtype=torch.uint8)}, cache)
    stats = tmp_path / 'celeba_train_4_stats.npz'
    stats.write_bytes(b'test stats')
    os.utime(stats, ns=(cache.stat().st_mtime_ns + 1000, cache.stat().st_mtime_ns + 1000))
    report = tmp_path / 'fid.json'
    report.write_text(json.dumps({'real_split': 'train', 'image_size': 4, 'num_real': 3,
                                  'inception': module.INCEPTION_PROTOCOL}))
    monkeypatch.setattr(module, 'validate_real_stats', lambda _: (np.zeros(1), np.eye(1), 3))
    module.adopt_legacy_real_stats(stats, cache, report, split='train', image_size=4)
    assert module.cache_matches(stats, module.real_cache_identity(cache, 'train', 4))
    metadata = json.loads(module.manifest_path(stats).read_text())
    assert 'not numerically recomputed' in metadata['provenance']
    with pytest.raises(ValueError, match='different real-data protocol'):
        module.adopt_legacy_real_stats(stats, cache, report, split='val', image_size=4)


def test_legacy_real_stats_rejects_changed_dataset(tmp_path, monkeypatch):
    import json
    import os
    import sampling.fid_celeba as module

    cache = tmp_path / 'cache_train_4.pt'
    torch.save({'images': torch.zeros(3, 3, 4, 4, dtype=torch.uint8)}, cache)
    stats = tmp_path / 'stats.npz'
    stats.write_bytes(b'test stats')
    os.utime(cache, ns=(stats.stat().st_mtime_ns + 1000, stats.stat().st_mtime_ns + 1000))
    report = tmp_path / 'fid.json'
    report.write_text(json.dumps({'real_split': 'train', 'image_size': 4, 'num_real': 3,
                                  'inception': module.INCEPTION_PROTOCOL}))
    with pytest.raises(ValueError, match='changed after'):
        module.adopt_legacy_real_stats(stats, cache, report, split='train', image_size=4)


def test_frechet_deadline_returns_same_small_result():
    import time
    from sampling.fid_celeba import frechet_with_deadline

    result = frechet_with_deadline(np.zeros(3), np.eye(3), np.ones(3), np.eye(3),
                                   deadline_unix=time.time() + 30)
    assert result == pytest.approx(3.0)


def test_immutable_generated_cache_retries_do_not_overwrite(tmp_path, monkeypatch):
    from sampling.fid_celeba import (
        cache_matches, publish_generated_cache, select_generated_path,
    )

    staging = tmp_path / 'local-stage'
    monkeypatch.setenv('DIFFUSION_CHECKPOINT_TMPDIR', str(staging))
    identity = {'num_samples': 2, 'ode_steps': 32, 'checkpoint': 'one'}
    original = select_generated_path(tmp_path, identity)
    original.write_bytes(b'interrupted copy without a manifest')
    retry = select_generated_path(tmp_path, identity)
    assert retry != original
    images = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
    publish_generated_cache(images, retry, identity)
    assert original.read_bytes() == b'interrupted copy without a manifest'
    assert cache_matches(retry, identity)
    assert torch.equal(torch.load(retry, weights_only=True), images)
    assert select_generated_path(tmp_path, identity) == retry
    regenerated = select_generated_path(tmp_path, identity, regenerate=True)
    assert regenerated not in {original, retry}
    other = select_generated_path(tmp_path, identity | {'checkpoint': 'two'})
    assert other not in {original, retry, regenerated}
    assert list(staging.iterdir()) == []


def test_generated_publication_refuses_existing_destination(tmp_path, monkeypatch):
    from sampling.fid_celeba import manifest_path, publish_generated_cache

    staging = tmp_path / 'local-stage'
    monkeypatch.setenv('DIFFUSION_CHECKPOINT_TMPDIR', str(staging))
    destination = tmp_path / 'existing.pt'
    destination.write_bytes(b'keep these existing bytes')
    with pytest.raises(FileExistsError):
        publish_generated_cache(torch.zeros(2), destination, {'num_samples': 2, 'ode_steps': 32})
    assert destination.read_bytes() == b'keep these existing bytes'
    assert not manifest_path(destination).exists()
    assert list(staging.iterdir()) == []
