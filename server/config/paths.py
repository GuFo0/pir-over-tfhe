from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "config" / "settings.json"
DB_PATH = BASE_DIR / "data" / "spam_db.npy"

ARTIFACTS_DIR = BASE_DIR / "build" / "artifacts"
SERVER_ZIP_PATH = ARTIFACTS_DIR / "server.zip"
# CLIENT_SPECS_PATH = ARTIFACTS_DIR / "client_specs.json"
