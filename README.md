# Valuador de alquileres — TP Final de Aprendizaje Automático

Estimador del precio de referencia de un departamento en el momento de publicar el
aviso. Es un **apoyo de decisión** para propietarios independientes (no una tasación
legal): devuelve un precio puntual **y un rango** para orientar el listado inicial y
dar una base para negociar.

- **Dataset:** [UCI ML Repository #555 — *Apartment for Rent Classified* (100K)](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified).
- **Scope del modelo:** departamentos, precio mensual en USD, ciudades con ≥ 200 avisos
  (101 ciudades, ≈ 48.826 filas).
- **Modelo:** XGBoost. **MAE en test ≈ $163**, con intervalo **±22,3 %** calibrado a
  ~90 % de cobertura real.
- Las features geográficas (`knn_price`, `city_price_median/mean`, `dist_city_center`)
  se ajustan **sólo con train** para evitar *data leakage*.

## Estructura del repositorio

```
01_eda_limpieza.ipynb      Notebook 1: EDA, limpieza, scope, split y feature engineering
02_regresion.ipynb         Notebook 2: baselines, GridSearch por familia, selección y test
03_error_y_deploy.ipynb    Notebook 3: calibración del intervalo, error, sesgo-varianza, deploy
valuador.py                Front mínimo (HTTP) que sirve el modelo entrenado
requirements.txt           Dependencias de Python
resources/                 Consigna del TP
presentacion/entrega1/     Presentación de la Entrega 1 (reveal.js)
```

Los directorios de datos y artefactos (`dataset/`, `data/processed/`, `outputs/`) están
en `.gitignore` **a propósito**: no se versionan y se regeneran corriendo los notebooks.

## Requisitos

- Python 3.10 o superior.
- Las dependencias de `requirements.txt`. Los notebooks además **verifican e instalan
  solos** lo que falte en su primera celda, así que basta con tener el entorno creado.

## Puesta a punto

```bash
# desde la raíz del repo
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducción paso a paso

Correr los notebooks **en orden**, cada uno de principio a fin (`Kernel → Restart &
Run All`). Cada notebook consume lo que produjo el anterior.

### 1. `01_eda_limpieza.ipynb`
- **Descarga automáticamente** el dataset desde UCI a `dataset/` (no hay que bajar nada
  a mano; requiere conexión la primera vez).
- Limpia, aplica el filtro de scope, hace el **split train/test** (80/20, estratificado
  por decil de precio, `random_state=42`) y construye las features geográficas **sólo
  con train**.
- **Genera:**
  - `data/processed/train.csv`
  - `data/processed/test.csv`
  - `data/processed/feature_notes.txt`
  - Gráficos del EDA en `outputs/`

### 2. `02_regresion.ipynb`
- Carga `train.csv` / `test.csv`, corre la ablación de `pets_allowed`, los baselines y
  la búsqueda de hiperparámetros (`GridSearchCV`) por familia de modelos, selecciona el
  mejor (XGBoost), lo evalúa en **test** y calibra la banda del intervalo.
- **Genera:**
  - `data/processed/best_pipeline.pkl` — el pipeline entrenado
  - `data/processed/best_model_info.json` — modelo elegido, features, `pets_mode` e
    intervalo (banda y confianza)

### 3. `03_error_y_deploy.ipynb`
- Carga el modelo y los datos, refina la calibración del intervalo, calcula la
  importancia de variables (permutación), el análisis de error, el diagnóstico de
  sesgo-varianza (validation/learning curves) y documenta el deploy y el monitoreo.
- Genera gráficos en `outputs/`; no produce artefactos nuevos que consuma el front.

## Correr el valuador (front)

Requiere haber corrido al menos los notebooks 1 y 2 (usa `train.csv`,
`best_pipeline.pkl` y `best_model_info.json`).

```bash
python valuador.py --selftest        # prueba una predicción por consola y sale
python valuador.py                   # levanta el server en http://localhost:8000
```

En el navegador se cargan superficie, dormitorios, baños, latitud/longitud y amenities,
y devuelve la estimación puntual con su rango. Si las coordenadas caen fuera del scope
(ninguna ciudad del dataset lo bastante cerca) el modelo se abstiene.

## Notas de reproducibilidad

- **Semilla fija** (`random_state=42`) en el split y en los modelos: los resultados son
  estables entre corridas.
- **Sin *data leakage*:** todo agregado geográfico y estadístico se ajusta con train y
  se aplica a test. La misma regla debe repetirse en cada reentrenamiento.
- Los números de la presentación (MAE ≈ $163, banda ±22,3 %) salen de esta corrida; con
  otro entorno pueden variar en el último dígito por diferencias de versiones de
  `xgboost` / `scikit-learn`.
- El dataset es de **2019**: el nivel de precios está desactualizado. Para uso real hay
  que reentrenar contra precios actuales (ver Notebook 3, sección de monitoreo).
