import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
DB_PATH = BASE_DIR / "data" / "spam_db.npy"
SPAM_DB = None


def get_ith_element_of_database(
    one_hot_vector: np.ndarray, database: np.ndarray
) -> tuple:
    return tuple(np.dot(one_hot_vector, database[i]) for i in range(database.shape[0]))


def get_db() -> np.ndarray:
    """
    Restituisce l'istanza del database attualmente caricata in memoria RAM.
    Se non è presente, la carica automaticamente dal disco (lazy loading).
    """
    global SPAM_DB
    if SPAM_DB is None:
        if not DB_PATH.exists():
            # Inizializzazione automatica se il file non esiste
            from server.build.build_db import generate_db

            generate_db()
        print("[UTILS] Caricamento del database dal disco alla RAM...")
        SPAM_DB = np.load(DB_PATH)
    return SPAM_DB


def update_db_in_memory(updates: list[tuple[bool, int]]):
    global SPAM_DB

    database = get_db()

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    input_bits = config["INPUT_BITS"]
    output_bits = config["OUTPUT_BITS"]
    database_length = 2**input_bits

    for is_spam, phone_index in updates:
        idx_0 = phone_index & (database_length - 1)
        idx_1 = phone_index >> input_bits

        j = idx_1 // output_bits
        k = idx_1 % output_bits

        current_value = database[j, idx_0]

        if is_spam:
            new_value = current_value | (1 << k)
        else:
            new_value = current_value & ~(1 << k)

        database[j, idx_0] = new_value

    np.save(DB_PATH, database)
    print(
        f"[UTILS] Database modificato direttamente in RAM e salvato su disco. Record aggiornati: {len(updates)}"
    )
