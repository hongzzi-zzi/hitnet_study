from .kitti_dataset import KITTIDataset
from .test_dataset import TESTDataset

__datasets__ = {
    "kitti": KITTIDataset,
    "test": TESTDataset
}
