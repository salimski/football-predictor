"""
Supabase-backed prediction tracker with local JSON fallback.

Requires Streamlit secrets:
    [supabase]
    url = "https://xxx.supabase.co"
    key = "eyJ..."

Supabase table 'predictions' schema — run this SQL in the Supabase SQL editor:

    CREATE TABLE predictions (
        id BIGSERIAL PRIMARY KEY,
        date TEXT,
        match TEXT,
        home_formation TEXT,
        away_formation TEXT,
        our_prob FLOAT,
        dc_prob FLOAT,
        blended_prob FLOAT,
        b365_implied FLOAT,
        poly_price FLOAT,
        edge FLOAT,
        signal TEXT,
        bet_placed BOOLEAN DEFAULT FALSE,
        result TEXT,
        final_score TEXT,
        dc_prob_u35 FLOAT,
        blended_u35 FLOAT,
        poly_u35 FLOAT,
        edge_u35 FLOAT,
        signal_u35 TEXT,
        bet_placed_u35 BOOLEAN DEFAULT FALSE,
        result_u35 TEXT,
        dc_prob_u45 FLOAT,
        blended_u45 FLOAT,
        poly_u45 FLOAT,
        edge_u45 FLOAT,
        signal_u45 TEXT,
        bet_placed_u45 BOOLEAN DEFAULT FALSE,
        result_u45 TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
"""

import math
import json
import os

import streamlit as st

TABLE = "predictions"

# All columns that can be edited by the user in the tracker UI
EDITABLE_COLS = [
    "bet_placed", "result", "final_score",
    "bet_placed_u35", "result_u35",
    "bet_placed_u45", "result_u45",
]


def _use_supabase():
    """Check if Supabase credentials are configured."""
    try:
        return bool(st.secrets.get("supabase", {}).get("url"))
    except Exception:
        return False


def _get_client():
    from supabase import create_client
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def _sanitize(record):
    """Replace NaN/inf with None for clean storage."""
    clean = {}
    for k, v in record.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean


# ── JSON fallback (local dev) ────────────────────────────────────────────

TRACKER_FILE = "data/prediction_tracker.json"


def _load_json():
    if os.path.exists(TRACKER_FILE):
        import re
        with open(TRACKER_FILE) as f:
            text = f.read()
        text = re.sub(r'\bNaN\b', 'null', text)
        records = json.loads(text)
        # Add synthetic id for consistency
        for i, rec in enumerate(records):
            if "id" not in rec:
                rec["id"] = i + 1
            if "final_score" not in rec:
                rec["final_score"] = None
        return records
    return []


def _save_json(records):
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    clean = [_sanitize(r) for r in records]
    with open(TRACKER_FILE, "w") as f:
        json.dump(clean, f, indent=2, default=str)


# ── Public API ────────────────────────────────────────────────────────────

def load_tracker():
    """Load all predictions, ordered by id."""
    if _use_supabase():
        client = _get_client()
        resp = client.table(TABLE).select("*").order("id").execute()
        rows = resp.data
        # Ensure final_score column exists for older rows
        for r in rows:
            if "final_score" not in r:
                r["final_score"] = None
        return rows
    return _load_json()


def insert_prediction(record):
    """Insert a new prediction. Returns the inserted row."""
    record = _sanitize(record)
    record.pop("id", None)  # let DB assign id
    record.pop("created_at", None)

    if _use_supabase():
        client = _get_client()
        resp = client.table(TABLE).insert(record).execute()
        return resp.data[0] if resp.data else record
    else:
        records = _load_json()
        record["id"] = max((r["id"] for r in records), default=0) + 1
        records.append(record)
        _save_json(records)
        return record


def update_prediction(pred_id, fields):
    """Update specific fields of a prediction by id."""
    fields = _sanitize(fields)

    if _use_supabase():
        client = _get_client()
        client.table(TABLE).update(fields).eq("id", pred_id).execute()
    else:
        records = _load_json()
        for rec in records:
            if rec["id"] == pred_id:
                rec.update(fields)
                break
        _save_json(records)


def bulk_update_editable(edited_records):
    """Update editable columns for a batch of records.

    edited_records: list of dicts, each must have 'id' plus editable fields.
    """
    if _use_supabase():
        client = _get_client()
        for rec in edited_records:
            pred_id = rec["id"]
            fields = {k: rec[k] for k in EDITABLE_COLS if k in rec}
            fields = _sanitize(fields)
            client.table(TABLE).update(fields).eq("id", pred_id).execute()
    else:
        records = _load_json()
        id_map = {r["id"]: r for r in records}
        for rec in edited_records:
            if rec["id"] in id_map:
                for k in EDITABLE_COLS:
                    if k in rec:
                        id_map[rec["id"]][k] = rec[k]
        _save_json(records)


def delete_predictions(pred_ids):
    """Delete predictions by id list."""
    if _use_supabase():
        client = _get_client()
        client.table(TABLE).delete().in_("id", list(pred_ids)).execute()
    else:
        records = _load_json()
        records = [r for r in records if r["id"] not in pred_ids]
        _save_json(records)
