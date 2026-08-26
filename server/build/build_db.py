import json
from pathlib import Path

import numpy as np

from server.config.paths import CONFIG_PATH, DB_PATH


def generate_db():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    total_bits = config["TOTAL_BITS"]
    input_bits = config["INPUT_BITS"]
    output_bits = config["OUTPUT_BITS"]

    database_length = 2**input_bits
    remaining_bits = total_bits - input_bits
    num_sub_dbs = int(np.ceil((2**remaining_bits) / output_bits))

    empty_db = np.zeros((num_sub_dbs, database_length), dtype=np.int64)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(DB_PATH, empty_db)
    print(
        f"[BUILD] Database vuoto generato con successo in {DB_PATH}. Dimensioni: {empty_db.shape}"
    )
