"""Episodic sampler for N-way K-shot few-shot learning."""

import torch
from torch.utils.data import Dataset


class FewshotDataset(Dataset):
    """N-way K-shot episode generator."""

    def __init__(
        self,
        data,
        labels,
        episode_num,
        way_num,
        shot_num,
        query_num,
        seed=None,
        hard_pool=None,
        hard_ratio=0.0,
        augment=False,
        augment_cfg=None,
        return_indices=False,
    ):
        self.data = data
        self.labels = labels
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.seed = seed if seed is not None else 0
        self.hard_ratio = float(hard_ratio)
        self.augment = bool(augment)
        self.return_indices = bool(return_indices)

        self.hard_pool = {}
        if hard_pool is not None:
            for class_id in range(self.way_num):
                indices = hard_pool.get(class_id, [])
                if isinstance(indices, torch.Tensor):
                    indices = indices.long().cpu()
                else:
                    indices = torch.tensor(list(indices), dtype=torch.long)
                self.hard_pool[class_id] = indices

        default_aug = {
            "time_shift_max": 4,
            "time_shift_prob": 0.5,
            "amp_scale_min": 0.9,
            "amp_scale_max": 1.1,
            "amp_scale_prob": 0.5,
            "time_mask_width": 4,
            "time_mask_prob": 0.25,
            "freq_mask_width": 4,
            "freq_mask_prob": 0.25,
        }
        self.augment_cfg = default_aug
        if augment_cfg:
            self.augment_cfg.update(augment_cfg)

        self.class_indices = {}
        for class_id in range(way_num):
            self.class_indices[class_id] = (labels == class_id).nonzero(as_tuple=True)[0]

        self._validate()

    def _validate(self):
        required = self.shot_num + self.query_num
        for class_id in range(self.way_num):
            available = len(self.class_indices[class_id])
            if available < required:
                print(f"Warning: Class {class_id} has {available} samples, need {required}")

    def __len__(self):
        return self.episode_num

    def _sample_hard_index(self, class_id, available, generator):
        if class_id not in self.hard_pool:
            return None
        hard_idx = self.hard_pool[class_id]
        if hard_idx.numel() == 0:
            return None

        hard_set = set(hard_idx.tolist())
        candidates = [idx.item() for idx in available if idx.item() in hard_set]
        if not candidates:
            return None

        pos = torch.randint(0, len(candidates), (1,), generator=generator).item()
        return int(candidates[pos])

    def _augment_one(self, image, generator):
        cfg = self.augment_cfg
        out = image.clone()
        _, height, width = out.shape

        if torch.rand(1, generator=generator).item() < cfg["time_shift_prob"]:
            shift = int(
                torch.randint(
                    -cfg["time_shift_max"],
                    cfg["time_shift_max"] + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=2)

        if torch.rand(1, generator=generator).item() < cfg["amp_scale_prob"]:
            scale = cfg["amp_scale_min"] + (
                cfg["amp_scale_max"] - cfg["amp_scale_min"]
            ) * torch.rand(1, generator=generator).item()
            out = out * scale

        if cfg["time_mask_width"] > 0 and torch.rand(1, generator=generator).item() < cfg["time_mask_prob"]:
            width_mask = min(int(cfg["time_mask_width"]), width)
            start = int(torch.randint(0, max(1, width - width_mask + 1), (1,), generator=generator).item())
            out[:, :, start : start + width_mask] = 0.0

        if cfg["freq_mask_width"] > 0 and torch.rand(1, generator=generator).item() < cfg["freq_mask_prob"]:
            height_mask = min(int(cfg["freq_mask_width"]), height)
            start = int(torch.randint(0, max(1, height - height_mask + 1), (1,), generator=generator).item())
            out[:, start : start + height_mask, :] = 0.0

        return out

    def _augment_batch(self, batch, generator):
        if not self.augment:
            return batch
        return torch.stack([self._augment_one(image, generator) for image in batch], dim=0)

    def __getitem__(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 10000 + index)

        support_images, support_targets = [], []
        query_images, query_targets = [], []
        support_indices, query_indices = [], []

        for class_id in range(self.way_num):
            indices = self.class_indices[class_id]
            perm = torch.randperm(len(indices), generator=generator)
            shuffled = indices[perm]

            hard_idx = None
            use_hard = (
                self.query_num > 0
                and self.hard_ratio > 0
                and torch.rand(1, generator=generator).item() < self.hard_ratio
            )
            if use_hard:
                hard_idx = self._sample_hard_index(class_id, shuffled, generator)

            if hard_idx is not None:
                hard_idx_t = torch.tensor([hard_idx], dtype=shuffled.dtype)
                remain = shuffled[shuffled != hard_idx]
                support_idx = remain[: self.shot_num]
                extra_query = remain[self.shot_num : self.shot_num + max(self.query_num - 1, 0)]
                query_idx = torch.cat([hard_idx_t, extra_query], dim=0)
            else:
                support_idx = shuffled[: self.shot_num]
                query_idx = shuffled[self.shot_num : self.shot_num + self.query_num]

            support_images.append(self.data[support_idx])
            query_images.append(self.data[query_idx])
            support_indices.append(support_idx)
            query_indices.append(query_idx)

            support_targets.append(torch.full((len(support_idx),), class_id, dtype=torch.long))
            query_targets.append(torch.full((len(query_idx),), class_id, dtype=torch.long))

        query_images = torch.cat(query_images)
        query_targets = torch.cat(query_targets)
        support_images = torch.cat(support_images)
        support_targets = torch.cat(support_targets)
        query_indices = torch.cat(query_indices)
        support_indices = torch.cat(support_indices)

        if self.augment:
            query_images = self._augment_batch(query_images, generator)
            support_images = self._augment_batch(support_images, generator)

        if self.return_indices:
            return (
                query_images,
                query_targets,
                support_images,
                support_targets,
                query_indices,
                support_indices,
            )

        return query_images, query_targets, support_images, support_targets


class RobustFewshotDataset(Dataset):
    """Episode generator for HROT robust final-test protocols.

    1-shot: support and query come from separate noisy pools.
    5-shot: support contains 4 clean shots plus 1 noisy outlier; query is noisy.
    """

    def __init__(
        self,
        episode_num,
        way_num,
        shot_num,
        query_num,
        seed=None,
        clean_data=None,
        clean_labels=None,
        support_data=None,
        support_labels=None,
        query_data=None,
        query_labels=None,
        outlier_data=None,
        outlier_labels=None,
        clean_source_ids=None,
        support_source_ids=None,
        query_source_ids=None,
        outlier_source_ids=None,
        return_indices=False,
    ):
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.seed = seed if seed is not None else 0
        self.return_indices = bool(return_indices)

        self.clean_data = clean_data
        self.clean_labels = clean_labels
        self.support_data = support_data
        self.support_labels = support_labels
        self.query_data = query_data
        self.query_labels = query_labels
        self.outlier_data = outlier_data
        self.outlier_labels = outlier_labels

        self.clean_source_ids = self._normalize_source_ids(clean_source_ids, clean_data)
        self.support_source_ids = self._normalize_source_ids(support_source_ids, support_data)
        self.query_source_ids = self._normalize_source_ids(query_source_ids, query_data)
        self.outlier_source_ids = self._normalize_source_ids(outlier_source_ids, outlier_data)

        if self.shot_num == 1:
            if support_data is None or support_labels is None:
                raise ValueError("1-shot robust protocol requires support_data/support_labels.")
            self.support_class_indices = self._build_class_indices(support_labels, "support")
        elif self.shot_num == 5:
            if clean_data is None or clean_labels is None:
                raise ValueError("5-shot robust protocol requires clean_data/clean_labels.")
            if outlier_data is None or outlier_labels is None:
                raise ValueError("5-shot robust protocol requires outlier_data/outlier_labels.")
            self.clean_class_indices = self._build_class_indices(clean_labels, "clean")
            self.outlier_class_indices = self._build_class_indices(outlier_labels, "outlier")
        else:
            raise ValueError(f"HROT robust protocol supports only 1-shot and 5-shot, got {shot_num}-shot.")

        if query_data is None or query_labels is None:
            raise ValueError("Robust protocol requires query_data/query_labels.")
        self.query_class_indices = self._build_class_indices(query_labels, "query")
        self._validate()

    @staticmethod
    def _normalize_source_ids(source_ids, data):
        if data is None:
            return None
        if source_ids is None:
            return [str(index) for index in range(len(data))]
        if len(source_ids) != len(data):
            raise ValueError(f"source_ids length {len(source_ids)} does not match data length {len(data)}.")
        return [str(source_id) for source_id in source_ids]

    def _build_class_indices(self, labels, pool_name):
        class_indices = {}
        for class_id in range(self.way_num):
            class_indices[class_id] = (labels == class_id).nonzero(as_tuple=True)[0]
            if len(class_indices[class_id]) == 0:
                print(f"Warning: Robust {pool_name} pool has no samples for class {class_id}")
        return class_indices

    def _validate(self):
        for class_id in range(self.way_num):
            query_available = len(self.query_class_indices[class_id])
            if query_available < self.query_num:
                print(f"Warning: Robust query class {class_id} has {query_available}, need {self.query_num}")
            if self.shot_num == 1:
                support_available = len(self.support_class_indices[class_id])
                if support_available < 1:
                    print(f"Warning: Robust support class {class_id} has {support_available}, need 1")
            else:
                clean_available = len(self.clean_class_indices[class_id])
                outlier_available = len(self.outlier_class_indices[class_id])
                if clean_available < 4:
                    print(f"Warning: Robust clean support class {class_id} has {clean_available}, need 4")
                if outlier_available < 1:
                    print(f"Warning: Robust outlier support class {class_id} has {outlier_available}, need 1")

    def __len__(self):
        return self.episode_num

    @staticmethod
    def _sample_indices(indices, count, generator):
        if len(indices) < count:
            raise ValueError(f"Need {count} samples, only have {len(indices)}.")
        perm = torch.randperm(len(indices), generator=generator)
        return indices[perm[:count]]

    def _exclude_sources(self, indices, source_ids, excluded_sources):
        if source_ids is None or not excluded_sources:
            return indices
        keep = [idx for idx in indices.tolist() if source_ids[int(idx)] not in excluded_sources]
        if not keep:
            return indices
        return torch.tensor(keep, dtype=indices.dtype)

    def __getitem__(self, index):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 10000 + index)

        support_images, support_targets = [], []
        query_images, query_targets = [], []
        support_indices, query_indices = [], []

        for class_id in range(self.way_num):
            excluded_sources = set()

            if self.shot_num == 1:
                support_idx = self._sample_indices(self.support_class_indices[class_id], 1, generator)
                if self.support_source_ids is not None:
                    excluded_sources.update(self.support_source_ids[int(idx)] for idx in support_idx.tolist())
                class_support = self.support_data[support_idx]
            else:
                clean_idx = self._sample_indices(self.clean_class_indices[class_id], 4, generator)
                if self.clean_source_ids is not None:
                    excluded_sources.update(self.clean_source_ids[int(idx)] for idx in clean_idx.tolist())
                outlier_candidates = self._exclude_sources(
                    self.outlier_class_indices[class_id],
                    self.outlier_source_ids,
                    excluded_sources,
                )
                outlier_idx = self._sample_indices(outlier_candidates, 1, generator)
                if self.outlier_source_ids is not None:
                    excluded_sources.update(self.outlier_source_ids[int(idx)] for idx in outlier_idx.tolist())
                class_support = torch.cat([self.clean_data[clean_idx], self.outlier_data[outlier_idx]], dim=0)
                support_idx = torch.cat([clean_idx, outlier_idx], dim=0)

            query_candidates = self._exclude_sources(
                self.query_class_indices[class_id],
                self.query_source_ids,
                excluded_sources,
            )
            query_idx = self._sample_indices(query_candidates, self.query_num, generator)

            support_images.append(class_support)
            query_images.append(self.query_data[query_idx])
            support_indices.append(support_idx)
            query_indices.append(query_idx)

            support_targets.append(torch.full((len(support_idx),), class_id, dtype=torch.long))
            query_targets.append(torch.full((len(query_idx),), class_id, dtype=torch.long))

        query_images = torch.cat(query_images)
        query_targets = torch.cat(query_targets)
        support_images = torch.cat(support_images)
        support_targets = torch.cat(support_targets)
        query_indices = torch.cat(query_indices)
        support_indices = torch.cat(support_indices)

        if self.return_indices:
            return (
                query_images,
                query_targets,
                support_images,
                support_targets,
                query_indices,
                support_indices,
            )

        return query_images, query_targets, support_images, support_targets
