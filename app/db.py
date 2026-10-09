"""SQLite storage for plants and analyses (swap for PostgreSQL later if needed)."""
import json
import sqlite3
import time
from contextlib import closing
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS plants(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, crop TEXT, location TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS analyses(
  id INTEGER PRIMARY KEY AUTOINCREMENT, plant_id INTEGER NOT NULL REFERENCES plants(id),
  observed_at REAL, created_at REAL, class_name TEXT, crop TEXT, disease TEXT, disease_key TEXT,
  confidence REAL, severity_pct REAL, severity_level TEXT,
  temperature REAL, humidity REAL, rainfall REAL, recent_rain INTEGER,
  risk_score REAL, risk_level TEXT, curve_json TEXT, report_json TEXT, model_version TEXT);
"""


def _conn(path=None):
    c = sqlite3.connect(path or DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init(path=None):
    with closing(_conn(path)) as c:
        c.executescript(SCHEMA)
        c.commit()


def create_plant(name, crop=None, location=None, path=None):
    with closing(_conn(path)) as c:
        cur = c.execute("INSERT INTO plants(name,crop,location,created_at) VALUES(?,?,?,?)",
                        (name.strip() or "My plant", crop, location, time.time()))
        c.commit()
        return cur.lastrowid


def list_plants(path=None):
    with closing(_conn(path)) as c:
        rows = c.execute("""SELECT p.*, (SELECT COUNT(*) FROM analyses a WHERE a.plant_id=p.id) AS scans
                            FROM plants p ORDER BY p.id""").fetchall()
        return [dict(r) for r in rows]


def get_plant(pid, path=None):
    with closing(_conn(path)) as c:
        r = c.execute("SELECT * FROM plants WHERE id=?", (pid,)).fetchone()
        return dict(r) if r else None


def save_analysis(plant_id, observed_at, d, path=None):
    with closing(_conn(path)) as c:
        cur = c.execute(
            """INSERT INTO analyses(plant_id,observed_at,created_at,class_name,crop,disease,disease_key,confidence,
               severity_pct,severity_level,temperature,humidity,rainfall,recent_rain,risk_score,risk_level,
               curve_json,report_json,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (plant_id, observed_at, time.time(), d["class_name"], d["crop"], d["disease"], d["disease_key"],
             d["confidence"], d["severity_pct"], d["severity_level"], d.get("temperature"), d.get("humidity"),
             d.get("rainfall"), int(bool(d.get("recent_rain"))), d.get("risk_score"), d.get("risk_level"),
             json.dumps(d.get("curve")) if d.get("curve") else None, "{}", d.get("model_version")))
        c.commit()
        return cur.lastrowid


def update_report(aid, report, path=None):
    with closing(_conn(path)) as c:
        c.execute("UPDATE analyses SET report_json=? WHERE id=?", (json.dumps(report), aid))
        c.commit()


def get_analysis(aid, path=None):
    with closing(_conn(path)) as c:
        r = c.execute("SELECT report_json FROM analyses WHERE id=?", (aid,)).fetchone()
        return json.loads(r["report_json"]) if r and r["report_json"] else None


def history(plant_id, path=None):
    with closing(_conn(path)) as c:
        rows = c.execute("SELECT * FROM analyses WHERE plant_id=? ORDER BY observed_at, id", (plant_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["curve"] = [p["median"] for p in json.loads(d.pop("curve_json"))] if d.get("curve_json") else None
        d.pop("report_json", None)
        out.append(d)
    return out
