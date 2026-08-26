import base64
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from concrete import fhe
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from server.app.utils import *
from server.app.utils import (
    clear_db_memory,
    get_db,
    populate_db_from_csv,
    populate_db_randomly,
)
from server.build.build_db import generate_db
from server.build.compile_circuit import compile_fhe_circuit
from server.config.paths import (
    BASE_DIR,
    CLIENT_SPECS_PATH,
    CONFIG_PATH,
    DB_PATH,
    SERVER_ZIP_PATH,
)


class RandomPopulateRequest(BaseModel):
    num_entries: int


class CSVPopulateRequest(BaseModel):
    filepath: str


app = FastAPI(title="PIR over TFHE Server")

FHE_SERVER = None
SPAM_DB = None
EVALUATION_KEYS_STORE = {}


def verify_and_build_resources():
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


# 1
@app.get("/api/specs")
def get_client_specs():
    try:
        verify_and_build_resources()
        load_resources_into_memory()

        specs_data = base64.b64encode(FHE_SERVER.client.specs.serialize()).decode(
            "utf-8"
        )

        with open(BASE_DIR / "config" / "settings.json", "r") as config_file:
            config = json.load(config_file)

        return {
            "specs": specs_data,
            "input_bits": config["INPUT_BITS"],
            "output_bits": config["OUTPUT_BITS"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore di inizializzazione: {str(e)}"
        )


# 2
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


# 3
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

        encrypted_query = FHE_SERVER.client_specs.deserialize(raw_query_bytes)

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


# # ADMIN APIs


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    dashboard_path = BASE_DIR / "app" / "admin.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="File admin.html non trovato.")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/admin/stats")
def get_db_stats():
    spam_db = get_db()
    total_spam = int(np.unpackbits(spam_db.astype(np.uint8)).sum())

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    total_capacity = 2 ** config["TOTAL_BITS"]

    return {
        "total_capacity": total_capacity,
        "total_spam": total_spam,
        "matrix_shape": spam_db.shape,
    }


@app.post("/api/admin/update-spam")
def admin_update_spam(updates: list[tuple[bool, int]]):
    try:
        update_db_in_memory(updates)
        return {
            "status": "success",
            "message": f"Aggiornati {len(updates)} record in RAM.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore di aggiornamento: {str(e)}"
        )


@app.post("/api/admin/populate-random")
def api_populate_random(request: RandomPopulateRequest):
    try:
        populate_db_randomly(request.num_entries)
        return {
            "status": "success",
            "message": f"Aggiunti {request.num_entries} spam casuali.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/populate-csv")
def api_populate_csv(request: CSVPopulateRequest):
    try:
        populate_db_from_csv(request.filepath)
        return {
            "status": "success",
            "message": f"Database popolato a partire da {request.filepath}.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/admin/reset-db")
def api_reset_db():
    try:
        print("[ADMIN] Inizio azzeramento del database...")

        clear_db_memory()

        if DB_PATH.exists():
            DB_PATH.unlink()

        generate_db()  # Eventuali modifiche alla configurazione vengono considerate automaticamente
        get_db()

        return {"status": "success", "message": "Database azzerato con successo!"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante l'azzeramento del DB: {str(e)}"
        )


@app.post("/api/v1/admin/recompile-circuit")
def api_recompile_circuit():  # Ricompila il circuito ed ELIMINA I CLIENT REGISTRATI (ID: evaluation_key)
    global FHE_SERVER, EVALUATION_KEYS_STORE

    try:
        print("[ADMIN] Inizio ricompilazione del circuito...")

        FHE_SERVER = None
        EVALUATION_KEYS_STORE.clear()

        if SERVER_ZIP_PATH.exists():
            SERVER_ZIP_PATH.unlink()
        if CLIENT_SPECS_PATH.exists():
            CLIENT_SPECS_PATH.unlink()

        compile_fhe_circuit()

        load_resources_into_memory()

        return {
            "status": "success",
            "message": "Circuito ricompilato e chiavi client eliminate.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante la ricompilazione: {str(e)}"
        )
