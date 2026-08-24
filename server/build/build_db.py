# In server/build/build_db.py

import json
from pathlib import Path

import numpy as np

# Definizione dei percorsi
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
DB_PATH = BASE_DIR / "data" / "spam_db.npy"


def generate_db():
    """
    Inizializza un database vuoto (tutti zeri) basato sulle specifiche di configurazione
    e lo salva sul disco in formato .npy.
    """
    # 1. Carica le impostazioni
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    total_bits = config["TOTAL_BITS"]
    input_bits = config["INPUT_BITS"]
    output_bits = config["OUTPUT_BITS"]

    # 2. Calcolo delle dimensioni della matrice
    database_length = 2**input_bits
    remaining_bits = total_bits - input_bits
    num_sub_dbs = int(np.ceil((2**remaining_bits) / output_bits))

    # 3. Creazione della matrice vuota (tutti i numeri sono impostati a 0 = non spam)
    # Usiamo int64 per garantire la compatibilità con i tipi del compilatore FHE
    empty_db = np.zeros((num_sub_dbs, database_length), dtype=np.int64)

    # Assicurati che la cartella data/ esista
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4. Salvataggio su disco
    np.save(DB_PATH, empty_db)
    print(
        f"[BUILD] Database vuoto generato con successo in {DB_PATH}. Dimensioni: {empty_db.shape}"
    )
