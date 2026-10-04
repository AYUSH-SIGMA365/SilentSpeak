import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Get base directory of the project
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Auto-AVSR settings
    AUTO_AVSR_PATH = os.getenv("AUTO_AVSR_PATH", r"C:\Users\Admin\Desktop\dl innov\auto_avsr")
    VSR_MODEL_PATH = os.getenv("VSR_MODEL_PATH", r"C:\Users\Admin\Desktop\dl innov\models\vsr_trlrs3_base.pth")

    # FastAPI backend settings
    HOST = "0.0.0.0"
    PORT = 8000
    
settings = Settings()
