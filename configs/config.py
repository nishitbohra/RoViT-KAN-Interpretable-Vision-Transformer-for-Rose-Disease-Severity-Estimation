from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    dataset_root: Path = Path("data")
    augmented_root: Path = Path("data/Augmented Image")
    original_root: Path = Path("data/Original Image")
    
    class_names: List[str] = field(default_factory=lambda: [
        "Healthy Leaf",
        "Leaf Holes",
        "Black Spot",
        "Dry Leaf"
    ])
    
    severity_map: Dict[str, int] = field(default_factory=lambda: {
        "Healthy Leaf": 0,
        "Leaf Holes": 1,
        "Black Spot": 2,
        "Dry Leaf": 3
    })
    
    num_classes: int = 4
    image_size: int = 224
    train_val_split: float = 0.8
    num_workers: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    use_curriculum: bool = True
    seeds: List[int] = field(default_factory=lambda: [42, 123, 999])
    stage_1_epochs: int = 10
    stage_2_epochs: int = 25
    stage_3_epochs: int = 40
    stage_4_epochs: int = 50


@dataclass
class LossConfig:
    lambda_ord: float = 1.0
    mu_unc: float = 0.5
    nu_kan: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: List[float] = field(default=None)   # populated at runtime from class weights


@dataclass
class ModelConfig:
    backbone: str = "deit_tiny_patch16_224"
    embed_dim: int = 192              # DeiT-Tiny actual embed dim (num_features=192)
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 4
    kan_layers: List[int] = field(default_factory=lambda: [192, 64, 16, 1])
    kan_num_knots: int = 5
    kan_degree: int = 3
    kan_hidden_dim: int = 64
    kan_num_splines: int = 5
    kan_spline_order: int = 3
    dropout: float = 0.3
    hidden_dim: int = 128


@dataclass
class PathConfig:
    checkpoints_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")
    figures_dir: Path = Path("results/figures")
    logs_dir: Path = Path("results/logs")
    
    def __post_init__(self):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FlagsConfig:
    use_mixup: bool = True
    use_cutmix: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixed_precision: bool = True
    curriculum: bool = True
    freeze_backbone_epochs: int = 5
    gradient_clip: float = 1.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    flags: FlagsConfig = field(default_factory=FlagsConfig)
    
    def get_stage_for_epoch(self, epoch: int) -> int:
        if not self.flags.curriculum:
            return 4
        if epoch <= self.train.stage_1_epochs:
            return 1
        elif epoch <= self.train.stage_2_epochs:
            return 2
        elif epoch <= self.train.stage_3_epochs:
            return 3
        else:
            return 4


def get_config() -> Config:
    return Config()
