import json
import sys
from pathlib import Path

import uuid

import numpy as np
from concrete import fhe
from fastapi import FastAPI, Header, HTTPException, Request, Response

from server.build.build_db import generate_db
from server.build.compile_circuit import compile_fhe_circuit

from server.app.utils import *


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.append(str(BASE_DIR.parent))
DB_PATH = BASE_DIR / "data" / "spam_db.npy"
SERVER_ZIP_PATH = BASE_DIR / "build" / "artifacts" / "server.zip"
CLIENT_SPECS_PATH = BASE_DIR / "build" / "artifacts" / "client_specs.json"

app = FastAPI(title="PIR over TFHE Server")

FHE_SERVER = None
SPAM_DB = None
EVALUATION_KEYS_STORE = {}


def verify_and_build_resources()
    if not DB_PATH.exists():
        print("[SERVER] Database non trovato. Avvio generazione database...")
        try:
            generate_db()
        except Exception as e:
            raise RuntimeError(f"Errore durante la generazione del database: {e}")

        if not DB_PATH.exists():
            raise RuntimeError(
                "Generazione del database completata, ma il file npy non è stato trovato."
            )

    if not SERVER_ZIP_PATH.exists() or not CLIENT_SPECS_PATH.exists():
        print("[SERVER] Circuito o specifiche non trovate. Avvio compilazione FHE...")
        try:
            compile_fhe_circuit()
        except Exception as e:
            raise RuntimeError(f"Errore durante la compilazione del circuito FHE: {e}")

        if not SERVER_ZIP_PATH.exists() or not CLIENT_SPECS_PATH.exists():
            raise RuntimeError(
                "Compilazione completata, ma i file degli artefatti risultano mancanti."
            )


def load_resources_into_memory():
    global FHE_SERVER, SPAM_DB

    if FHE_SERVER is None:
        print("[SERVER] Caricamento del circuito FHE in memoria...")
        FHE_SERVER = fhe.Server.load(SERVER_ZIP_PATH)

    if SPAM_DB is None:
        print("[SERVER] Caricamento del database in memoria...")
        SPAM_DB = np.load(DB_PATH)


@app.get("/api/specs")
def get_client_specs():
    try:
        verify_and_build_resources()
        load_resources_into_memory()

        with open(CLIENT_SPECS_PATH, "r") as file:
            specs_data = json.load(file)

        return {
            "specs": specs_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore di inizializzazione: {str(e)}")

@app.post("/api/register")
async def register_keys(request: Request):
    global FHE_SERVER
    if FHE_SERVER is None:
        raise HTTPException(
            status_code=400, detail="Server FHE non ancora inizializzato."
        )

    try:
        raw_key_bytes = await request.body()
        evaluation_keys = fhe.EvaluationKeys.deserialize(raw_key_bytes)

        client_id = str(uuid.uuid4())


        EVALUATION_KEYS_STORE[client_id] = evaluation_keys
        return {
            "status": "success",
            "client_id": client_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Errore di registrazione chiavi: {str(e)}"
        )


@app.post("/api/query")
async def process_query(request: Request, client_id: str = Header(...)):
    global FHE_SERVER

    spam_db = get_db()

    if FHE_SERVER is None or spam_db is None:
        raise HTTPException(status_code=400, detail="Risorse del server non caricate.")

    if client_id not in EVALUATION_KEYS_STORE:
        raise HTTPException(
            status_code=401,
            detail="Chiavi di valutazione non registrate per questo client.",
        )

    try:
        raw_query_bytes = await request.body()

        encrypted_query = FHE_SERVER.client_specs.deserialize(
            raw_query_bytes
        )

        evaluation_keys = EVALUATION_KEYS_STORE[client_id]
        encrypted_result = FHE_SERVER.run(
            encrypted_query, spam_db, evaluation_keys=evaluation_keys
        )

        serialized_result = encrypted_result.serialize()
        return Response(
            content=serialized_result, media_type="application/octet-stream"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante l'elaborazione omomorfa: {str(e)}"
        )

@app.post("/api/admin/update-spam")
def admin_update_spam(updates: list[tuple[bool, int]]):
    try:
        update_db_in_memory(updates)
        return {"status": "success", "message": f"Aggiornati {len(updates)} record in RAM."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore di aggiornamento: {str(e)}")
