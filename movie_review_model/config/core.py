"""Model and App configs"""
from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load
import movie_review_model

# Project Directories
PACKAGE_ROOT = Path(movie_review_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

class ModelConfig(BaseModel):
    """Model config object."""
    type: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    optimizer: str
    epsilon: float
    weight_decay: float
    seed: int
    tokenizer: str

class DataConfig(BaseModel):
    """Data config object."""
    train_data_path: str
    test_data_path: str

class OutputConfig(BaseModel):
    """Output config object."""
    output_model_path: str

class AppConfig(BaseModel):
    """App config object."""
    package_name: str
    pipeline_name: str
    pipeline_save_file: str

class Config(BaseModel):
    """Master config object."""
    app_config: AppConfig
    model: ModelConfig
    data: DataConfig
    output: OutputConfig

def fetch_config_from_yaml(cfg_path: Path = CONFIG_FILE_PATH) -> YAML:
    """Fetch config file."""
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at path: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as file:
        config_data = file.read()
    return load(config_data)

def create_and_validate_config() -> Config:
    """Create config object."""
    parsed_config = fetch_config_from_yaml()
    return Config(**parsed_config.data)

config = create_and_validate_config()
