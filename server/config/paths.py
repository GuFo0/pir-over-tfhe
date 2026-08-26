from pathlib import Path

# __file__ punta a questo file (paths.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Percorsi dei dati e delle configurazioni
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
DB_PATH = BASE_DIR / "data" / "spam_db.npy"

# Percorsi degli artefatti generati dal compilatore FHE
ARTIFACTS_DIR = BASE_DIR / "build" / "artifacts"
SERVER_ZIP_PATH = ARTIFACTS_DIR / "server.zip"
CLIENT_SPECS_PATH = ARTIFACTS_DIR / "client_specs.json"
