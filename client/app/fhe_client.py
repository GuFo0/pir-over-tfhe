import base64
import json
import os
from pathlib import Path

import numpy as np
import requests
from concrete import fhe


class PIRClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.client_id = None
        self.input_bits = None
        self.output_bits = None

        # Gestione dei percorsi locali del client
        self.base_dir = Path(__file__).resolve().parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.specs_path = self.artifacts_dir / "client_specs.json"
        self.client = None

    def initialize_session(self):

        print("[CLIENT] Connessione al server per scaricare le specifiche...")

        response = requests.get(f"{self.server_url}/api/specs")
        if response.status_code != 200:
            raise RuntimeError(
                f"Impossibile ottenere le specifiche dal server: {response.text}"
            )

        data = response.json()
        self.input_bits = int(data["input_bits"])
        self.output_bits = int(data["output_bits"])
        specs_data = base64.b64decode(data["specs"])

        print(
            f"[CLIENT] Configurazione di rete: INPUT_BITS={self.input_bits}, OUTPUT_BITS={self.output_bits}"
        )

        # 2. Scrittura locale temporanea delle specifiche del client
        # with open(self.specs_path, "w") as file:
        #     json.dump(specs_data, file)

        # 3. Istanza del client FHE basato sulle specifiche
        # client_specs = fhe.ClientSpecs.load(self.specs_path)
        client_specs = fhe.ClientSpecs.deserialize(specs_data)
        self.client = fhe.Client(client_specs)

        # 4. Generazione locale della chiave segreta e di valutazione
        print(
            "[CLIENT] Generazione delle chiavi FHE (l'operazione richiede qualche secondo)..."
        )
        self.client.keygen()

        # 5. Serializzazione e registrazione della chiave di valutazione sul server
        print("[CLIENT] Invio della chiave di valutazione al server...")
        serialized_eval_keys = self.client.evaluation_keys.serialize()

        headers = {
            "Client-ID": self.client_id,
            "Content-Type": "application/octet-stream",
        }

        reg_response = requests.post(
            f"{self.server_url}/api/register",
            data=serialized_eval_keys,
            headers=headers,
        )

        if reg_response.status_code != 200:
            raise RuntimeError(f"Registrazione fallita: {reg_response.text}")

        self.client_id = reg_response.json()["client_id"]
        print(f"[CLIENT] ID Sessione assegnato dal Server: {self.client_id}")
        print(
            "[CLIENT] Configurazione completata. Pronto per le interrogazioni private.\n"
        )

    def verify_phone_number(self, phone_index: int) -> bool:
        if self.client is None or self.client_id is None:
            raise RuntimeError(
                "La sessione non è stata inizializzata. Eseguire prima initialize_session()."
            )

        database_length = 2**self.input_bits

        # 1. Scomposizione dell'indice in idx_0 (cifrato) e idx_1 (in chiaro locale)
        idx_0 = phone_index & (database_length - 1)
        idx_1 = phone_index >> self.input_bits

        # 2. Generazione del vettore one-hot
        one_hot = np.zeros(database_length, dtype=np.int8)
        one_hot[idx_0] = 1

        # 3. Cifratura dell'argomento
        print(f"[CLIENT] Cifratura del vettore one-hot (indirizzo locale: {idx_0})...")
        encrypted_args, _ = self.client.encrypt(one_hot, None)
        serialized_query = encrypted_args.serialize()

        # 4. Invio query cifrata al server
        print("[CLIENT] Invio della query cifrata...")
        headers = {
            "Client-ID": self.client_id,
            "Content-Type": "application/octet-stream",
        }

        query_response = requests.post(
            f"{self.server_url}/api/query", data=serialized_query, headers=headers
        )

        if query_response.status_code != 200:
            raise RuntimeError(
                f"L'esecuzione della query ha fallito: {query_response.text}"
            )

        # 5. Ricezione e deserializzazione dei risultati pubblici
        print("[CLIENT] Risposta ricevuta dal server. Decifratura in corso...")
        serialized_result = query_response.content
        encrypted_result = fhe.Value.deserialize(serialized_result)

        # 6. Decifratura del vettore
        decrypted_vector = self.client.decrypt(encrypted_result)

        # 7. Ricostruzione logica del bit di spam usando idx_1
        j = idx_1 // self.output_bits
        k = idx_1 % self.output_bits

        target_integer = decrypted_vector[j]
        is_spam = (target_integer >> k) & 1

        return bool(is_spam)
