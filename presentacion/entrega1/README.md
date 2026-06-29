# Presentación Entrega 1 — reveal.js

Esta carpeta contiene la presentación de la Entrega 1 del Trabajo Práctico Final.

## Cómo abrir la presentación

### Opción 1: servidor local (recomendada)

Desde la raíz del proyecto:

```bash
source venv/bin/activate
cd presentations/entrega1
python -m http.server 8000
```

Luego abrir en el navegador: http://localhost:8000

### Opción 2: abrir el archivo directamente

Abrir `presentations/entrega1/index.html` en cualquier navegador moderno.

## Navegación

- Flechas ← → para moverse entre diapositivas.
- `Esc` para ver la vista general.
- `S` para activar la vista de orador (speaker view).

## Exportar a PDF

1. Agregar `?print-pdf` a la URL, por ejemplo:
   `http://localhost:8000?print-pdf`
2. Usar la función de imprimir a PDF del navegador (Chrome recomendado).
3. En opciones de impresión: guardar como PDF, sin márgenes, fondos habilitados.

## Gráficos

Los gráficos de la presentación fueron generados en el notebook `notebooks/tp4_entrega1.ipynb` y exportados a `assets/img/`.
