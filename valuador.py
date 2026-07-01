#!/usr/bin/env python3
"""Front mínimo para valuar un departamento con el modelo del TP.

Uso:
    python valuador.py            # levanta el server en http://localhost:8000
    python valuador.py --selftest # prueba una predicción y sale

Sin dependencias extra: sólo stdlib + lo que ya usa el modelo (sklearn/joblib/pandas).
"""
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
KNN_K = 10
EARTH_KM = 6371.0
MAX_KM = 25.0            # coords más lejos que esto de un aviso del dataset -> fuera de scope

# --- Carga del modelo y de los agregados (reconstruidos SOLO con train, sin leakage) ---
best_model = joblib.load(PROC / "best_pipeline.pkl")
info = json.load(open(PROC / "best_model_info.json"))
train = pd.read_csv(PROC / "train.csv")

NUM_FEATURES = info["num_features"]
PETS_MODE = info["pets_mode"]
BAND = info["interval"]["band_pct"] / 100
CONF = info["interval"]["conf"]

_centroids = train.groupby("city_key")[["latitude", "longitude"]].mean()
_g = train.groupby("city_key")["price"]
_city_med, _city_mean = _g.median(), _g.mean()
_glob_med = float(train["price"].median())
_coords_tr = np.radians(train[["latitude", "longitude"]].to_numpy())
_tree = BallTree(_coords_tr, metric="haversine")
_price_tr = train["price"].to_numpy()
_city_tr = train["city_key"].to_numpy()


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_KM * 2 * np.arcsin(np.sqrt(a))


def coord_en_scope(lat, lon):
    """Devuelve (city_key, dist_al_vecino_km) del aviso de train más cercano, o None si está fuera de scope."""
    dist, idx = _tree.query(np.radians([[lat, lon]]), k=1)
    d_km = float(dist[0][0] * EARTH_KM)
    if d_km > MAX_KM:
        return None
    return _city_tr[idx[0][0]], d_km


def valuar(*, square_feet, bedrooms, bathrooms, latitude, longitude,
           n_amenities=0, has_photo=False, fee=False, pets=False):
    scope = coord_en_scope(latitude, longitude)
    if scope is None:
        raise ValueError("Las coordenadas están fuera del scope: no hay ninguna ciudad del "
                         "dataset lo bastante cerca.")
    city_key, _ = scope

    c = _centroids.loc[city_key]
    dist = float(_haversine_km(latitude, longitude, c["latitude"], c["longitude"]))
    _, nn = _tree.query(np.radians([[latitude, longitude]]), k=KNN_K)
    knn_price = float(np.median(_price_tr[nn[0]]))

    row = {
        "square_feet": square_feet, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "latitude": latitude, "longitude": longitude, "dist_city_center": dist,
        "knn_price": knn_price,
        "city_price_median": float(_city_med.get(city_key, _glob_med)),
        "city_price_mean": float(_city_mean.get(city_key, _glob_med)),
        "n_amenities": n_amenities, "has_amenities": int(n_amenities > 0),
        "has_photo_bin": int(bool(has_photo)), "fee_bin": int(bool(fee)),
        "pets_known": int(bool(pets)), "state": city_key.split(" | ")[0],
    }
    cols = list(NUM_FEATURES) + ["state"]
    if PETS_MODE == "as_no":
        row["pets_cat"] = "Cats,Dogs" if pets else "no"
        cols += ["pets_cat"]
    X = pd.DataFrame([row])[cols]

    est = float(best_model.predict(X)[0])
    return {
        "ciudad": city_key,
        "estimacion": round(est),
        "rango_min": round(est * (1 - BAND)),
        "rango_max": round(est * (1 + BAND)),
        "banda_pct": round(BAND * 100, 1),
        "confianza": int(CONF * 100),
    }


# ----------------------------- Front (HTML plano) -----------------------------
def _field(label, name, value="", **attrs):
    a = " ".join(f'{k}="{html.escape(str(v))}"' for k, v in attrs.items())
    return (f'<label>{html.escape(label)}'
            f'<input name="{name}" value="{html.escape(str(value))}" {a}></label>')


def render(form=None, result=None, error=None):
    form = form or {}
    def g(k): return form.get(k, "")
    out = ""
    if error:
        out = f'<div class="msg err">{html.escape(error)}</div>'
    elif result:
        out = (f'<div class="msg ok">'
               f'<div class="city">{html.escape(result["ciudad"])}</div>'
               f'<div class="price">USD {result["estimacion"]}<span>/mes</span></div>'
               f'<div class="range">Rango {result["confianza"]}%: '
               f'USD {result["rango_min"]} – {result["rango_max"]} (±{result["banda_pct"]}%)</div>'
               f'</div>')
    checks = ('<label class="chk"><input type="checkbox" name="has_photo" {p}> Con foto</label>'
              '<label class="chk"><input type="checkbox" name="fee" {f}> Con fee</label>'
              '<label class="chk"><input type="checkbox" name="pets" {m}> Acepta mascotas</label>').format(
        p="checked" if g("has_photo") else "", f="checked" if g("fee") else "",
        m="checked" if g("pets") else "")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Valuador de alquiler</title>
<style>
  * {{ box-sizing:border-box }}
  body {{ font-family:system-ui,sans-serif; background:#f4f4f5; color:#18181b;
         max-width:400px; margin:40px auto; padding:0 16px }}
  h1 {{ font-size:1.15rem; font-weight:600; margin:0 0 16px }}
  form {{ background:#fff; border:1px solid #e4e4e7; border-radius:8px; padding:20px }}
  label {{ display:block; font-size:.82rem; color:#52525b; margin-bottom:12px }}
  input[type=number] {{ width:100%; margin-top:4px; padding:7px 9px; font-size:.95rem;
         border:1px solid #d4d4d8; border-radius:6px }}
  input:focus {{ outline:none; border-color:#71717a }}
  .chk {{ font-size:.9rem; color:#18181b; margin:6px 0 }}
  .chk input {{ margin-right:6px }}
  button {{ width:100%; margin-top:14px; padding:10px; font-size:.95rem; font-weight:600;
         color:#fff; background:#18181b; border:none; border-radius:6px; cursor:pointer }}
  button:hover {{ background:#3f3f46 }}
  .msg {{ margin-top:16px; padding:16px; border-radius:8px }}
  .ok {{ background:#fff; border:1px solid #e4e4e7 }}
  .err {{ background:#fef2f2; border:1px solid #fecaca; color:#b91c1c; font-size:.9rem }}
  .city {{ font-size:.8rem; color:#71717a; margin-bottom:6px }}
  .price {{ font-size:1.8rem; font-weight:700 }}
  .price span {{ font-size:.9rem; font-weight:400; color:#71717a }}
  .range {{ font-size:.85rem; color:#52525b; margin-top:4px }}
</style></head>
<body>
<h1>Valuador de alquiler</h1>
<form method="post" action="/">
{_field("Superficie (sq ft)", "square_feet", g("square_feet"), type="number", step="1", required="")}
{_field("Dormitorios", "bedrooms", g("bedrooms"), type="number", step="1", required="")}
{_field("Baños", "bathrooms", g("bathrooms"), type="number", step="0.5", required="")}
{_field("Latitud", "latitude", g("latitude"), type="number", step="any", required="")}
{_field("Longitud", "longitude", g("longitude"), type="number", step="any", required="")}
{_field("Cantidad de amenities", "n_amenities", g("n_amenities") or "0", type="number", step="1")}
{checks}
<button type="submit">Valuar</button>
</form>
{out}
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if urlparse(self.path).path != "/":
            self.send_response(404); self.end_headers(); return
        self._send(render())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode("utf-8")).items()}
        try:
            res = valuar(
                square_feet=float(form["square_feet"]),
                bedrooms=float(form["bedrooms"]),
                bathrooms=float(form["bathrooms"]),
                latitude=float(form["latitude"]),
                longitude=float(form["longitude"]),
                n_amenities=int(float(form.get("n_amenities") or 0)),
                has_photo="has_photo" in form,
                fee="fee" in form,
                pets="pets" in form,
            )
            self._send(render(form=form, result=res))
        except ValueError as e:
            self._send(render(form=form, error=str(e)))
        except Exception as e:  # entrada inválida
            self._send(render(form=form, error=f"Entrada inválida: {e}"))

    def log_message(self, *a):
        pass


def main():
    if "--selftest" in sys.argv:
        print(valuar(square_feet=850, bedrooms=1, bathrooms=1,
                     latitude=34.05, longitude=-118.24, n_amenities=3, has_photo=True))
        print("fuera de scope ->", end=" ")
        try:
            valuar(square_feet=800, bedrooms=1, bathrooms=1, latitude=0.0, longitude=0.0)
        except ValueError as e:
            print(e)
        return
    port = 8000
    print(f"Valuador en http://localhost:{port}  (Ctrl+C para cortar)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
