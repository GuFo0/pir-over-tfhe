import csv
import json
import random
from pathlib import Path

import numpy as np

from server.build.build_db import generate_db
from server.config.paths import CONFIG_PATH, DB_PATH

SPAM_DB = None


def get_ith_element_of_database(
    one_hot_vector: np.ndarray, database: np.ndarray
) -> tuple:
    return tuple(np.dot(one_hot_vector, database[i]) for i in range(database.shape[0]))


def get_db() -> np.ndarray:
    global SPAM_DB
    if SPAM_DB is None:
        if not DB_PATH.exists():
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


def populate_db_randomly(num_spam_entries: int):
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    total_bits = config["TOTAL_BITS"]
    max_index = (2**total_bits) - 1

    if num_spam_entries > max_index:
        raise ValueError(
            "Il numero di spam richiesto supera la capacità totale del database."
        )

    print(f"[UTILS] Generazione di {num_spam_entries} numeri spam casuali...")
    random_indices = random.sample(range(max_index + 1), num_spam_entries)
    updates = [(True, idx) for idx in random_indices]
    update_db_in_memory(updates)


def populate_db_from_csv(filepath: str):
    file_path_obj = Path(filepath)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Il file {filepath} non esiste.")

    updates = []
    print(f"[UTILS] Lettura del file CSV: {filepath}...")

    with open(file_path_obj, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue  # Ignora le righe vuote

            raw_value = row[0].strip()

            # Ignora valori non numerici
            if raw_value.isdigit():
                phone_index = int(raw_value)
                updates.append((True, phone_index))

    if not updates:
        print("[UTILS] Nessun numero valido trovato nel CSV.")
        return
    update_db_in_memory(updates)


def clear_db_memory():
    global SPAM_DB
    SPAM_DB = None
    print("[UTILS] Memoria del database svuotata.")
