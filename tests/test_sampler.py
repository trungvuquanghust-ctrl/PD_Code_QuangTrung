import torch

from tim_2026.data.episodic import FewshotDataset


def test_episode_sampling_is_deterministic() -> None:
    images = torch.arange(4 * 8 * 3 * 4 * 4, dtype=torch.float32).reshape(32, 3, 4, 4)
    labels = torch.arange(4).repeat_interleave(8)
    first = FewshotDataset(images, labels, 2, 4, 1, 1, seed=42)[0]
    second = FewshotDataset(images, labels, 2, 4, 1, 1, seed=42)[0]
    assert all(torch.equal(left, right) for left, right in zip(first, second))
