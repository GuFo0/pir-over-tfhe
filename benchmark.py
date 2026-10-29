#!/usr/bin/env python3
"""
- Test 1: Configurazione di Default (20, 14, 8) e scomposizione delle fasi
- Test 2: Trade-off della Geometria (B_in = 12 vs 14 vs 16)
- Test 3: Scalabilità del Database (B_total = 18 vs 19 vs 20)
- Test 4: Efficienza del Multithreading (dataflow_parallelize True vs False)
- Baseline in chiaro: Scansione lineare NumPy O(N) e Accesso Diretto O(1)
"""

import json
import time
from pathlib import Path

import numpy as np
from concrete import fhe

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "benchmark_results.json"

# Matrice degli esperimenti mirati (6 configurazioni chiave)
EXPERIMENTS = [
    # 1. Configurazione di Default (Punto di riferimento)
    {
        "id": "baseline_default",
        "desc": "Default (1M contatti, matrice 8x16384, parallelo)",
        "total_bits": 20,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    # 2-5. Asse Scalabilità B_total (da 262k a 4.2M contatti)
    {
        "id": "scale_btotal_18",
        "desc": "Scalabilità B_total=18 (262k contatti, 2 righe)",
        "total_bits": 18,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    {
        "id": "scale_btotal_19",
        "desc": "Scalabilità B_total=19 (524k contatti, 4 righe)",
        "total_bits": 19,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    {
        "id": "scale_btotal_21",
        "desc": "Scalabilità B_total=21 (2.1M contatti, 16 righe)",
        "total_bits": 21,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    {
        "id": "scale_btotal_22",
        "desc": "Scalabilità B_total=22 (4.2M contatti, 32 righe)",
        "total_bits": 22,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    {
        "id": "scale_btotal_23",
        "desc": "Scalabilità B_total=23 (8.4M contatti, 64 righe)",
        "total_bits": 23,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    {
        "id": "scale_btotal_24",
        "desc": "Scalabilità B_total=24 (16.8M contatti, 128 righe)",
        "total_bits": 24,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": True,
    },
    # 6-7. Asse Geometria B_in (Trade-off Upload vs Calcolo a 1M contatti)
    {
        "id": "geom_bin_12",
        "desc": "Geometria B_in=12 (Query corta 4096, 32 righe)",
        "total_bits": 20,
        "input_bits": 12,
        "output_bits": 8,
        "parallelize": True,
    },
    {
        "id": "geom_bin_16",
        "desc": "Geometria B_in=16 (Query lunga 65536, 2 righe)",
        "total_bits": 20,
        "input_bits": 16,
        "output_bits": 8,
        "parallelize": True,
    },
    # 8. Asse Multithreading (Impatto di dataflow_parallelize)
    {
        "id": "parallel_false",
        "desc": "Multithreading disattivato (Default sequenziale su 1 core)",
        "total_bits": 20,
        "input_bits": 14,
        "output_bits": 8,
        "parallelize": False,
    },
]

# EXPERIMENTS = [
#     {
#         "id": "scale_btotal_23",
#         "desc": "Scalabilità B_total=23 (8.4M contatti, 64 righe)",
#         "total_bits": 23,
#         "input_bits": 14,
#         "output_bits": 8,
#         "parallelize": True,
#     },
#     {
#         "id": "scale_btotal_24",
#         "desc": "Scalabilità B_total=24 (16.8M contatti, 128 righe)",
#         "total_bits": 24,
#         "input_bits": 14,
#         "output_bits": 8,
#         "parallelize": True,
#     },
# ]


def run_single_benchmark(exp, num_iterations=5):
    total_bits = exp["total_bits"]
    input_bits = exp["input_bits"]
    output_bits = exp["output_bits"]
    parallelize = exp["parallelize"]

    db_cols = 2**input_bits
    remaining_bits = total_bits - input_bits
    db_rows = int(np.ceil((2**remaining_bits) / output_bits))
    total_records = 2**total_bits
    db_size_mb = (db_rows * db_cols * 8) / (1024 * 1024)

    print(f"\n---> Esecuzione: [{exp['id']}] - {exp['desc']}")
    print(
        f"     Matrice: {db_rows} righe x {db_cols} colonne | Dimensione DB:"
        f" {db_size_mb:.2f} MB"
    )

    # 1. Compilazione
    t0 = time.perf_counter()
    inputset = [
        (
            np.eye(db_cols, dtype=np.int8)[np.random.randint(db_cols)],
            np.random.randint(2**output_bits, size=(db_rows, db_cols), dtype=np.int64),
        )
        for _ in range(10)
    ]

    def get_ith_element(one_hot_vector: np.ndarray, database: np.ndarray):
        return fhe.array(
            [np.dot(one_hot_vector, database[i]) for i in range(database.shape[0])]
        )

    compiler = fhe.Compiler(
        get_ith_element, {"one_hot_vector": "encrypted", "database": "clear"}
    )
    circuit = compiler.compile(inputset, dataflow_parallelize=parallelize)
    t_compilation = time.perf_counter() - t0

    # 2. Generazione Chiavi e Misure Dimensionali
    client = fhe.Client(circuit.client.specs)
    t0 = time.perf_counter()
    client.keygen()
    t_keygen = time.perf_counter() - t0

    eval_keys_serialized = client.evaluation_keys.serialize()
    eval_keys_mb = len(eval_keys_serialized) / (1024 * 1024)
    specs_kb = len(circuit.client.specs.serialize()) / 1024

    # Creazione database per il test
    spam_db = np.random.randint(2**output_bits, size=(db_rows, db_cols), dtype=np.int64)

    enc_times = []
    server_fhe_times = []
    dec_times = []
    numpy_clear_times = []
    direct_clear_times = []
    query_sizes_kb = []
    response_sizes_b = []

    # 3. Iterazioni di misura
    for it in range(num_iterations):
        phone_index = np.random.randint(0, total_records)
        idx_0 = phone_index & (db_cols - 1)
        idx_1 = phone_index >> input_bits
        j = idx_1 // output_bits
        k = idx_1 % output_bits

        # Cifratura
        one_hot = np.zeros(db_cols, dtype=np.int8)
        one_hot[idx_0] = 1

        t0 = time.perf_counter()
        encrypted_args, _ = client.encrypt(one_hot, None)
        enc_times.append(time.perf_counter() - t0)

        serialized_q = encrypted_args.serialize()
        query_sizes_kb.append(len(serialized_q) / 1024)

        # Server FHE
        deser_q = fhe.Value.deserialize(serialized_q)
        t0 = time.perf_counter()
        enc_res = circuit.server.run(
            deser_q, spam_db, evaluation_keys=client.evaluation_keys
        )
        server_fhe_times.append(time.perf_counter() - t0)

        serialized_ans = enc_res.serialize()
        response_sizes_b.append(len(serialized_ans))

        # Decifratura
        deser_ans = fhe.Value.deserialize(serialized_ans)
        t0 = time.perf_counter()
        dec_vec = client.decrypt(deser_ans)
        res_bit = (dec_vec[j] >> k) & 1
        dec_times.append(time.perf_counter() - t0)

        # Baseline 1: NumPy Lineare O(N)
        t0 = time.perf_counter()
        _ = np.array([np.dot(one_hot, spam_db[r]) for r in range(db_rows)])
        numpy_clear_times.append(time.perf_counter() - t0)

        # Baseline 2: Accesso Diretto O(1)
        t0 = time.perf_counter()
        direct_bit = (spam_db[j, idx_0] >> k) & 1
        direct_clear_times.append(time.perf_counter() - t0)

        assert res_bit == direct_bit, "Errore di correttezza!"

    # 4. Calcolo Metriche
    mean_server = np.mean(server_fhe_times) * 1000
    mean_enc = np.mean(enc_times) * 1000
    mean_dec = np.mean(dec_times) * 1000
    mean_numpy = np.mean(numpy_clear_times) * 1000
    mean_direct = np.mean(direct_clear_times) * 1000

    throughput_mb_s = db_size_mb / np.mean(server_fhe_times)
    slowdown_numpy = np.mean(server_fhe_times) / np.mean(numpy_clear_times)
    slowdown_direct = np.mean(server_fhe_times) / np.mean(direct_clear_times)

    print(
        f"     -> Calcolo Server: {mean_server:.1f} ms | Throughput:"
        f" {throughput_mb_s:.2f} MB/s"
    )
    print(f"     -> Cifratura: {mean_enc:.1f} ms | Decifratura: {mean_dec:.2f} ms")
    print(
        f"     -> Upload Query: {np.mean(query_sizes_kb):.1f} KB | Download:"
        f" {int(np.mean(response_sizes_b))} Byte"
    )
    print(f"     -> Slowdown vs NumPy: {slowdown_numpy:,.1f}x")

    return {
        "id": exp["id"],
        "desc": exp["desc"],
        "params": {
            "total_bits": total_bits,
            "input_bits": input_bits,
            "output_bits": output_bits,
            "total_records": total_records,
            "matrix_rows": db_rows,
            "matrix_cols": db_cols,
            "db_size_mb": round(db_size_mb, 3),
            "parallelize": parallelize,
        },
        "sizes": {
            "specs_kb": round(specs_kb, 2),
            "eval_keys_mb": round(eval_keys_mb, 2),
            "query_upload_kb": round(float(np.mean(query_sizes_kb)), 2),
            "response_download_bytes": int(np.mean(response_sizes_b)),
        },
        "timings_ms": {
            "compilation_s": round(t_compilation, 2),
            "keygen_s": round(t_keygen, 2),
            "encrypt_mean": round(float(mean_enc), 2),
            "encrypt_std": round(float(np.std(enc_times) * 1000), 2),
            "server_fhe_mean": round(float(mean_server), 2),
            "server_fhe_std": round(float(np.std(server_fhe_times) * 1000), 2),
            "decrypt_mean": round(float(mean_dec), 2),
            "decrypt_std": round(float(np.std(dec_times) * 1000), 2),
            "numpy_clear_mean": round(float(mean_numpy), 4),
            "direct_clear_mean": round(float(mean_direct), 5),
        },
        "metrics": {
            "throughput_mb_s": round(float(throughput_mb_s), 2),
            "slowdown_vs_numpy": round(float(slowdown_numpy), 1),
            "slowdown_vs_direct": round(float(slowdown_direct), 1),
        },
    }


def main():
    print("=" * 65)
    print("   SUITE DI BENCHMARK ACCADEMICA - PIR OVER TFHE")
    print("=" * 65)

    results = []
    t_total_start = time.perf_counter()

    for exp in EXPERIMENTS:
        res = run_single_benchmark(exp, num_iterations=5)
        results.append(res)

    total_elapsed = time.perf_counter() - t_total_start

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[COMPLETATO] 6 esperimenti eseguiti in {total_elapsed / 60:.1f} minuti.")
    print(f"[SALVATO] Risultati esportati in: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
