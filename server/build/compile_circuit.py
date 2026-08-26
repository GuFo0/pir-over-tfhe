import json
from pathlib import Path

import numpy as np
from concrete import fhe

from server.app.utils import get_ith_element_of_database
from server.config.paths import CLIENT_SPECS_PATH, CONFIG_PATH, SERVER_ZIP_PATH


def compile_fhe_circuit():
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

    inputset = [
        (
            np.eye(database_length, dtype=np.int8)[np.random.randint(database_length)],
            np.random.randint(
                2**output_bits, size=(num_sub_dbs, database_length), dtype=np.int64
            ),
        )
        for _ in range(10)
    ]

    compiler = fhe.Compiler(
        get_ith_element_of_database,
        {"one_hot_vector": "encrypted", "database": "clear"},
    )

    print("Compilazione in corso...")
    circuit = compiler.compile(
        inputset,
        show_mlir=False,
        show_graph=False,
        use_gpu=False,
        show_progress=False,
        dataflow_parallelize=True,
    )
    SERVER_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    circuit.server.save(SERVER_ZIP_PATH)

    print(
        f"[BUILD] Compilazione completata! Artefatto salvati in:\n- {SERVER_ZIP_PATH}"
    )

    return circuit
