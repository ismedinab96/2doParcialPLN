# Dashboard de defensa — Segundo parcial PLN II/2026

Presentación interactiva (14 diapositivas) que recorre todo el trabajo: datos, limpieza, BoW, TF-IDF, PCA,
Naive Bayes, SVM, comparación, análisis por comentario, clasificador en vivo, errores y conclusiones.

## Estructura esperada
```
parcial2-pln/
├── .streamlit/config.toml        # tema verde / olivo / blanco
├── dashboard/app.py
├── dashboard/preproc.py
├── artefactos/                   # lo genera el notebook (Fases 1 a 5)
│   ├── train.csv  test.csv  split.joblib
│   ├── modelos.joblib  svm_calibrado.joblib
│   └── recursos_preprocesamiento.joblib
├── notebook/Parcial2_PLN_Completo.ipynb
└── requirements.txt
```

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"   # una sola vez
streamlit run dashboard/app.py
```
Ejecútalo **desde la carpeta raíz** del proyecto. Si los artefactos están en otra ruta:
`ARTEFACTOS=/ruta/a/artefactos streamlit run dashboard/app.py` (en Windows PowerShell: `$env:ARTEFACTOS="C:\ruta"; streamlit run dashboard/app.py`).

## Uso en la defensa
- Flechas **← →** del teclado, o los botones «Anterior / Siguiente». El botón **Índice** salta a cualquier diapositiva.
- Las diapositivas «Limpieza», «Naive Bayes» y «Clasificador en vivo» aceptan texto escrito en el momento.
- Edita al inicio de `app.py`: `INTEGRANTES`, `DOCENTE`, `CURSO`.

## Importante
1. **Debe coincidir con tu notebook.** `dashboard/preproc.py` replica la función `limpiar_texto` del notebook.
   Si la cambiaste (por ejemplo, para conservar «¿?» en la clase *consulta*), aplica el mismo cambio en `Preprocesador.pasos()`;
   si no, los textos escritos en vivo se procesarán distinto a los de entrenamiento.
2. Las categorías se leen de `train.csv` / `test.csv`; sirve con 2 o 3 clases.
3. Los modelos se reconocen por su tipo (CountVectorizer/TfidfVectorizer + Naive Bayes/SVM), no por el nombre con que los guardaste.
4. Las comparativas (métricas de test, validación cruzada 5×3, matrices) se recalculan al iniciar con tus modelos y datos; tarda unos segundos la primera vez.
5. Si existe `cinco_nuevos.csv` se muestra en «Errores y análisis crítico».
