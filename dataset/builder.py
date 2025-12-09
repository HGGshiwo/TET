from pathlib import Path
import yaml
from torch.utils.data import ConcatDataset
from utils import *
from .base import BaseDataset
from .egoschema import EgoSchemaDataset
from .moviechat import MovieChatDataset
from .nextqa import NextMCDataset, NextOEDataset
from .videomme import VideoMMEDataset
from .intentqa import IntentQADataset
from .mlvu import MLVUDataset
from .longvideo import LongVideoDataset

def build_dataset(dataset_config, name, is_training=False):
    if isinstance(dataset_config, (Path, str)):
        dataset_config = yaml.safe_load(Path(dataset_config).read_text())

    name_list = [name] if isinstance(name, str) else name
    if is_training:
        datasets = ConcatDataset(
            [BaseDataset.create(dataset_config, n) for n in name_list]
        )
    else:
        datasets = {n: BaseDataset.create(dataset_config, n) for n in name_list}
    return datasets if is_training or not isinstance(name, str) else datasets[name]
