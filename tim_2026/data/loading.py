"""PD Scalogram Dataset Loader.

- Input: Configurable size RGB images (64×64 default, 84×84 for standard benchmarks)
- Normalization: Auto-computed from dataset
- Supports both:
  1. Pre-split folders (train/val/test subfolders)
  2. HROT robust folders (train/val/test_clean plus robust test pools)
  3. Auto-split from a single folder
"""
import os
import random
import numpy as np
from PIL import Image
import torch

try:
    import torchvision.transforms as transforms
except ImportError:
    class _Compose:
        def __init__(self, transforms_list):
            self.transforms_list = transforms_list

        def __call__(self, image):
            out = image
            for transform in self.transforms_list:
                out = transform(out)
            return out

    class _Resize:
        def __init__(self, size):
            self.size = size

        def __call__(self, image):
            return image.resize(self.size, Image.BILINEAR)

    class _ToTensor:
        def __call__(self, image):
            array = np.asarray(image, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = array[:, :, None]
            array = np.transpose(array, (2, 0, 1))
            return torch.from_numpy(array)

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class _FallbackTransforms:
        Compose = _Compose
        Resize = _Resize
        ToTensor = _ToTensor
        Normalize = _Normalize

    transforms = _FallbackTransforms()


# Canonical 4-class setup for augmented split dataset
CLASS_ORDER = ['surface', 'internal', 'corona', 'notpd']
CLASS_MAP = {name: i for i, name in enumerate(CLASS_ORDER)}
CLASS_DISPLAY_NAMES = {
    'surface': 'Surface',
    'internal': 'Internal',
    'corona': 'Corona',
    'notpd': 'NotPD',
}

HROT_ROBUST_TEST_SPLITS = [
    'test_1shot_support_snr10',
    'test_1shot_query_snr10',
    'test_5shot_support_outlier_snr0',
    'test_5shot_query_snr10',
]


def canonicalize_class_name(name):
    """Normalize class folder names (supports legacy aliases)."""
    key = name.strip().lower()
    if key in ('notpd', 'nopd', 'not_pd'):
        return 'notpd'
    if key in ('surface', 'internal', 'corona'):
        return key
    return None


def list_image_files(folder):
    """List image files while excluding labeled helper images."""
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        and 'labeled' not in f.lower()
        and 'labeled:' not in f
    ])


def resolve_class_dirs(root_path):
    """
    Resolve canonical class -> actual folder path under root_path.
    If both alias folders exist (e.g., notpd + nopd), first encountered is used.
    """
    mapping = {}
    if not os.path.isdir(root_path):
        return mapping

    for folder in sorted(os.listdir(root_path)):
        full = os.path.join(root_path, folder)
        if not os.path.isdir(full):
            continue
        canonical = canonicalize_class_name(folder)
        if canonical is not None and canonical not in mapping:
            mapping[canonical] = full
    return mapping


def collect_split_files(split_path, classes=None):
    """Collect canonical class image files from one split folder."""
    class_names = list(classes or CLASS_ORDER)
    split_files = []
    class_dirs = resolve_class_dirs(split_path)
    for class_name in class_names:
        canonical = canonicalize_class_name(class_name)
        if canonical is None:
            print(f"Warning: Unknown class name for split loading: {class_name}")
            continue
        class_path = class_dirs.get(canonical)
        if class_path is None:
            print(f"Warning: Class folder not found: {split_path}/{class_name}")
            continue

        files = list_image_files(class_path)
        label = CLASS_MAP[canonical]
        split_files.extend([(os.path.join(class_path, f), label) for f in files])
    return split_files


def load_external_scalogram_split(split_path, transform, classes=None):
    """Load an external test split using an existing train-derived transform."""
    split_files = collect_split_files(split_path, classes=classes)
    images, labels = [], []
    for fpath, label in split_files:
        img = Image.open(fpath).convert('RGB')
        images.append(transform(img).numpy())
        labels.append(label)

    images = np.array(images) if images else np.array([])
    labels = np.array(labels) if labels else np.array([])
    return images, labels, split_files


class PDScalogramPreSplit:
    """Dataset loader for pre-split folders (train/val/test already separated).
    
    Expected folder structure (canonical):
        data_path/
            train/
                surface/
                internal/
                corona/
                notpd/  (or nopd legacy alias)
            val/
                surface/
                internal/
                corona/
                notpd/  (or nopd legacy alias)
            test/
                surface/
                internal/
                corona/
                notpd/  (or nopd legacy alias)
    """
    
    def __init__(self, data_path, image_size=64, test_split='test', extra_splits=None):
        """
        Args:
            data_path: Path to dataset directory containing train/val/test subfolders
            image_size: Input image size (default: 64, use 84 for standard benchmarks)
        """
        self.data_path = os.path.abspath(data_path)
        self.image_size = image_size
        self.test_split = test_split
        self.extra_split_names = list(extra_splits or [])
        self.classes = list(CLASS_ORDER)
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists placeholders
        self.train_files = []
        self.val_files = []
        self.test_files = []
        self.extra_split_files = {split_name: [] for split_name in self.extra_split_names}
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset (Pre-split): {self.data_path}')
        if self.test_split != 'test':
            print(f'Using {self.test_split}/ as clean test split')
        
        # 1. Scan pre-split folders
        self._scan_presplit_folders()
        
        # 2. Compute stats (ONLY on training data)
        self._compute_stats()
        
        # 3. Load images (apply normalization)
        self._load_images()
        
        self._shuffle_all()
    
    def _scan_presplit_folders(self):
        """Scan train/val/test subfolders and collect file paths."""
        splits = [
            ('train', 'train', self.train_files),
            ('val', 'val', self.val_files),
            ('test', self.test_split, self.test_files),
        ]
        splits.extend(
            (split_name, split_name, self.extra_split_files[split_name])
            for split_name in self.extra_split_names
        )
        
        for role_name, folder_name, file_list in splits:
            split_path = os.path.join(self.data_path, folder_name)
            if not os.path.exists(split_path):
                print(f"Warning: {folder_name} folder not found at {split_path}")
                continue

            class_dirs = resolve_class_dirs(split_path)
            for class_name in CLASS_ORDER:
                class_path = class_dirs.get(class_name)
                if class_path is None:
                    print(f"Warning: Class folder not found: {split_path}/{class_name}")
                    continue

                files = list_image_files(class_path)
                label = CLASS_MAP[class_name]
                file_list.extend([(os.path.join(class_path, f), label) for f in files])
        
        print(f'Found: Train={len(self.train_files)}, Val={len(self.val_files)}, Test={len(self.test_files)}')
        for split_name in self.extra_split_names:
            print(f'Found {split_name}: {len(self.extra_split_files[split_name])}')
    
    def _compute_stats(self):
        """Compute per-channel mean and std using ONLY training data."""
        print('Computing mean/std on training set...')
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')  # RGB
            pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data found for stats computation. Using default mean/std.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_file_list(self, file_list):
        images, labels = [], []
        for fpath, label in file_list:
            img = Image.open(fpath).convert('RGB')
            images.append(self.transform(img).numpy())
            labels.append(label)
        images = np.array(images) if images else np.array([])
        labels = np.array(labels) if labels else np.array([])
        return images, labels

    def _load_images(self):
        """Load images using the pre-computed splits and normalization."""
        self.X_train, self.y_train = self._load_file_list(self.train_files)
        self.X_val, self.y_val = self._load_file_list(self.val_files)
        self.X_test, self.y_test = self._load_file_list(self.test_files)

        for split_name in self.extra_split_names:
            images, labels = self._load_file_list(self.extra_split_files[split_name])
            setattr(self, f'X_{split_name}', images)
            setattr(self, f'y_{split_name}', labels)
            setattr(self, f'{split_name}_files', self.extra_split_files[split_name])
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
        for split_name in self.extra_split_names:
            print(f'Loaded {split_name}: {len(getattr(self, f"X_{split_name}"))}')
    
    def _shuffle_all(self):
        """Shuffle all splits with fixed seed."""
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
            if len(self.train_files) == len(idx):
                self.train_files = [self.train_files[i] for i in idx]
        
        if len(self.X_val) > 0:
            idx = np.arange(len(self.X_val))
            np.random.default_rng(1).shuffle(idx)
            self.X_val = self.X_val[idx]
            self.y_val = self.y_val[idx]
            if len(self.val_files) == len(idx):
                self.val_files = [self.val_files[i] for i in idx]
        
        if len(self.X_test) > 0:
            idx = np.arange(len(self.X_test))
            np.random.default_rng(2).shuffle(idx)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
            if len(self.test_files) == len(idx):
                self.test_files = [self.test_files[i] for i in idx]

        for split_offset, split_name in enumerate(self.extra_split_names, start=3):
            images = getattr(self, f'X_{split_name}', np.array([]))
            labels = getattr(self, f'y_{split_name}', np.array([]))
            files = getattr(self, f'{split_name}_files', [])
            if len(images) == 0:
                continue
            idx = np.arange(len(images))
            np.random.default_rng(split_offset).shuffle(idx)
            setattr(self, f'X_{split_name}', images[idx])
            setattr(self, f'y_{split_name}', labels[idx])
            if len(files) == len(idx):
                setattr(self, f'{split_name}_files', [files[i] for i in idx])


class PDScalogramHROTRobust(PDScalogramPreSplit):
    """Pre-split dataset with clean train/val and protocol-controlled robust test pools."""

    def __init__(self, data_path, image_size=64):
        available_extra_splits = [
            split_name
            for split_name in HROT_ROBUST_TEST_SPLITS
            if os.path.isdir(os.path.join(data_path, split_name))
        ]
        super().__init__(
            data_path,
            image_size=image_size,
            test_split='test_clean',
            extra_splits=available_extra_splits,
        )
        self.has_hrot_robust_protocol = all(
            os.path.isdir(os.path.join(self.data_path, split_name))
            for split_name in HROT_ROBUST_TEST_SPLITS
        )
        self.robust_test_splits = available_extra_splits


def load_dataset(data_path, image_size=64, val_per_class=60, test_per_class=60):
    """Auto-detect dataset structure and load appropriately.
    
    If data_path contains train/val/test subfolders, use PDScalogramPreSplit.
    If data_path contains train/val/test_clean and robust test pools, use
    PDScalogramHROTRobust with test_clean as the clean test split.
    Otherwise, use PDScalogram (auto-split).
    
    Args:
        data_path: Path to dataset
        image_size: Input image size
        val_per_class: (for auto-split only) Validation samples per class
        test_per_class: (for auto-split only) Test samples per class
    
    Returns:
        Dataset object (either PDScalogramPreSplit or PDScalogram)
    """
    # Check if pre-split structure exists
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    test_path = os.path.join(data_path, 'test')
    test_clean_path = os.path.join(data_path, 'test_clean')
    
    if os.path.isdir(train_path) and os.path.isdir(val_path) and os.path.isdir(test_path):
        print("Detected pre-split dataset structure (train/val/test folders)")
        return PDScalogramPreSplit(data_path, image_size=image_size)
    elif os.path.isdir(train_path) and os.path.isdir(val_path) and os.path.isdir(test_clean_path):
        print("Detected HROT robust dataset structure (train/val/test_clean + robust test pools)")
        return PDScalogramHROTRobust(data_path, image_size=image_size)
    else:
        print("Using auto-split dataset structure")
        return PDScalogram(data_path, val_per_class=val_per_class, 
                          test_per_class=test_per_class, image_size=image_size)


class PDScalogram:
    """Dataset loader with auto-computed normalization (from training set only)."""
    
    def __init__(self, data_path, val_per_class=60, test_per_class=60, image_size=64):
        """
        Args:
            data_path: Path to dataset directory
            val_per_class: Samples reserved for validation per class
            test_per_class: Samples reserved for test per class
            image_size: Input image size (default: 64, use 84 for standard benchmarks)
        """
        self.data_path = os.path.abspath(data_path)
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.image_size = image_size
        self.classes = list(CLASS_ORDER)
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists placeholders
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        
        # 1. Prepare splits (identify files for train/val/test)
        self._prepare_splits()
        
        # 2. Compute stats (ONLY on training data)
        self._compute_stats()
        
        # 3. Load images (apply normalization)
        self._load_images()
        
        self._shuffle_all()
    
    def _prepare_splits(self):
        """Scan directories and split files into train/val/test lists."""
        # Resolve class folders first (supports aliases like nopd -> notpd)
        class_dirs = resolve_class_dirs(self.data_path)

        missing = [c for c in CLASS_ORDER if c not in class_dirs]
        if missing:
            print(f"Warning: Missing class folders: {missing}")

        # Find min class size
        class_sizes = {}
        for class_name in CLASS_ORDER:
            path = class_dirs.get(class_name)
            class_sizes[class_name] = len(list_image_files(path)) if path else 0
        
        if not class_sizes:
            raise ValueError(f"No data found in {self.data_path}")

        min_size = min(class_sizes.values())
        if min_size == 0:
            print("Warning: Found empty class or no images.")
            return {}, {}, {}

        val_size = min(self.val_per_class, min_size)
        test_size = min(self.test_per_class, min_size - val_size)
        eval_total = val_size + test_size
        
        print(f'Split: {val_size}/class for val, {test_size}/class for test, rest for train')
        
        for class_name in CLASS_ORDER:
            path = class_dirs.get(class_name)
            if path is None:
                continue

            files = list_image_files(path)
            random.Random(42).shuffle(files)
            files = files[:min_size]  # Balance classes
            
            # Split: val_size for val, test_size for test, rest for train
            val_files_class = files[:val_size]
            test_files_class = files[val_size:eval_total]
            train_files_class = files[eval_total:]
            
            # Store as (full_path, label) tuples
            label = CLASS_MAP[class_name]
            self.val_files.extend([(os.path.join(path, f), label) for f in val_files_class])
            self.test_files.extend([(os.path.join(path, f), label) for f in test_files_class])
            self.train_files.extend([(os.path.join(path, f), label) for f in train_files_class])

    def _compute_stats(self):
        """Compute per-channel mean and std using ONLY training data."""
        print('Computing mean/std on training set...')
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')
            pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data found for stats computation. Using default mean/std.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_images(self):
        """Load images using the pre-computed splits and normalization."""
        # Load Train
        for fpath, label in self.train_files:
            img = Image.open(fpath).convert('RGB')
            self.X_train.append(self.transform(img).numpy())
            self.y_train.append(label)
            
        # Load Val
        for fpath, label in self.val_files:
            img = Image.open(fpath).convert('RGB')
            self.X_val.append(self.transform(img).numpy())
            self.y_val.append(label)
            
        # Load Test
        for fpath, label in self.test_files:
            img = Image.open(fpath).convert('RGB')
            self.X_test.append(self.transform(img).numpy())
            self.y_test.append(label)
        
        # Convert to arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle all splits with fixed seed."""
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
            if len(self.train_files) == len(idx):
                self.train_files = [self.train_files[i] for i in idx]
        
        if len(self.X_val) > 0:
            idx = np.arange(len(self.X_val))
            np.random.default_rng(1).shuffle(idx)
            self.X_val = self.X_val[idx]
            self.y_val = self.y_val[idx]
            if len(self.val_files) == len(idx):
                self.val_files = [self.val_files[i] for i in idx]
        
        if len(self.X_test) > 0:
            idx = np.arange(len(self.X_test))
            np.random.default_rng(2).shuffle(idx)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
            if len(self.test_files) == len(idx):
                self.test_files = [self.test_files[i] for i in idx]
