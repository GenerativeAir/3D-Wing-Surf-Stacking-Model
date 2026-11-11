from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

HISTORICAL_DIR = DATA_DIR / "historical"
ROOT_DIR = HISTORICAL_DIR / "Airfoil_txt"
POLARS_DIR = HISTORICAL_DIR / "polars_txt"

MODELS_DIR = DATA_DIR / "models"

WEIGHTS_FILE = MODELS_DIR / "vae_weights.pth"
WEIGHTS_FILE_PARSEC = MODELS_DIR / "vae_weights_parsec.pth"
WEIGHTS_FILE_PRETRAINED = MODELS_DIR / "inverse_airfoil_pretrained.pth"
WEIGHTS_FILE_INVERSE = MODELS_DIR / "inverse_airfoil_weights.pth"