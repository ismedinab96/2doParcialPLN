"""
Dashboard de defensa — Segundo parcial PLN II/2026
Clasificación automática de comentarios universitarios (BoW, TF-IDF, PCA, Naive Bayes y SVM)

Ejecutar:  streamlit run dashboard/app.py
Lee los artefactos que exporta el notebook (carpeta `artefactos/`).
"""
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preproc import cargar_preprocesador  # noqa: E402

# ======================================================================================
# CONFIGURACIÓN EDITABLE
# ======================================================================================
TITULO = "Clasificación automática de comentarios universitarios"
SUBTITULO = "Bag of Words · TF-IDF · PCA · Naive Bayes · SVM"
CURSO = "Procesamiento del Lenguaje Natural — Segundo parcial II/2026"
INTEGRANTES = ["Medina Balboa Iver Samuel", "Espinoza Romero Andres Maximiliano"]      # <-- editar
DOCENTE = "Lic. Rosa Flores"                                  # <-- editar
SEMILLA = 42

ART = Path(os.environ.get("ARTEFACTOS", Path(__file__).resolve().parent.parent / "artefactos"))
if not ART.exists():
    ART = Path.cwd() / "artefactos"

# Paleta: verde, verde olivo y blanco
C = {"bosque": "#173F2B", "verde": "#2F8F4E", "hoja": "#5DB26F", "olivo": "#6B7A2A",
     "olivo_c": "#A9B84A", "niebla": "#F3F6EC", "linea": "#DCE4CC", "blanco": "#FFFFFF", "texto": "#173F2B"}
COLORES_BASE = {"positiv": "#2F8F4E", "consult": "#8A9A2B", "negativ": "#173F2B"}
EXTRA = ["#5DB26F", "#A9B84A", "#4F6B3A", "#C9D29A"]
SIMBOLOS = ["circle", "diamond", "square", "x", "cross"]

st.set_page_config(page_title="Defensa PLN — Clasificador de comentarios", page_icon="🌿",
                   layout="wide", initial_sidebar_state="collapsed")

# ======================================================================================
# ESTILOS
# ======================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');
:root{--bosque:#173F2B;--verde:#2F8F4E;--hoja:#5DB26F;--olivo:#6B7A2A;--olivoc:#A9B84A;--niebla:#F3F6EC;--linea:#DCE4CC;}
html, body, [class*="css"], .stApp{font-family:'DM Sans','Segoe UI',system-ui,sans-serif;color:var(--bosque);}
.stApp{background:#FFFFFF;}
header[data-testid="stHeader"]{background:transparent;height:0;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1480px;padding:1.1rem 2.2rem 5.2rem 2.2rem;}
h1,h2,h3{font-family:'Fraunces',Georgia,serif;color:var(--bosque);letter-spacing:-.01em;}
h3{font-size:1.25rem;margin-top:.4rem;}
.topbar{display:flex;align-items:center;gap:14px;margin-bottom:.35rem;color:var(--olivo);font-size:.92rem;}
.topbar b{color:var(--bosque);}
.prog{height:5px;border-radius:4px;background:var(--linea);overflow:hidden;margin-bottom:1.1rem;}
.prog>div{height:100%;background:linear-gradient(90deg,var(--verde),var(--olivo));transition:width .5s ease;}
.s-title{font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:2.45rem;line-height:1.1;margin:.2rem 0 .15rem 0;
  animation:entrada .55s ease both;}
.s-sub{color:var(--olivo);font-size:1.08rem;margin-bottom:1.1rem;max-width:980px;animation:entrada .55s .05s ease both;}
@keyframes entrada{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important;}}
.card{background:#fff;border:1px solid var(--linea);border-left:5px solid var(--olivo);border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem;}
.card.v{border-left-color:var(--verde);} .card.b{border-left-color:var(--bosque);}
.card h4{margin:.0rem 0 .35rem 0;font-family:'Fraunces',Georgia,serif;font-size:1.12rem;}
.card p, .card li{font-size:.98rem;line-height:1.5;margin:0 0 .25rem 0;}
.band{background:var(--niebla);border-radius:12px;padding:1.0rem 1.3rem;margin:.4rem 0 1rem 0;}
.kpi{background:var(--niebla);border-radius:12px;padding:.85rem 1rem;border-bottom:4px solid var(--verde);}
.kpi .v{font-family:'Fraunces',Georgia,serif;font-size:2.0rem;font-weight:700;line-height:1;}
.kpi .l{font-size:.88rem;color:var(--olivo);margin-top:.25rem;}
.kpi.o{border-bottom-color:var(--olivo);}
.hero{background:linear-gradient(135deg,var(--bosque) 0%,#245c3d 55%,var(--olivo) 100%);color:#fff;border-radius:18px;padding:3.2rem 3rem;}
.hero h1{color:#fff;font-size:3.3rem;line-height:1.05;margin:.2rem 0 .8rem 0;max-width:900px;}
.hero p{color:#E9F1DA;font-size:1.15rem;margin:.15rem 0;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:1.4rem;}
.chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:.35rem .9rem;font-size:.92rem;color:#fff;}
.pipe{display:flex;gap:6px;flex-wrap:wrap;align-items:stretch;margin:.4rem 0 1rem 0;}
.pipe div{flex:1 1 130px;background:var(--niebla);border-top:4px solid var(--verde);border-radius:8px;padding:.6rem .7rem;font-size:.92rem;}
.pipe div b{display:block;font-family:'Fraunces',Georgia,serif;font-size:1.02rem;}
.pipe div.on{background:var(--bosque);color:#fff;border-top-color:var(--olivoc);}
.paso{display:flex;gap:12px;align-items:flex-start;padding:.5rem 0;border-bottom:1px dashed var(--linea);}
.paso .n{min-width:28px;height:28px;border-radius:50%;background:var(--verde);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem;}
.paso .t{font-size:.86rem;color:var(--olivo);} .paso .r{font-family:ui-monospace,Consolas,monospace;font-size:.98rem;word-break:break-word;}
.res{border-radius:12px;padding:.9rem 1rem;background:#fff;border:1px solid var(--linea);}
.res .m{font-size:.85rem;color:var(--olivo);} .res .c{font-family:'Fraunces',Georgia,serif;font-size:1.55rem;font-weight:700;}
.res .p{font-size:.9rem;}
.veredicto{border-radius:14px;padding:1.1rem 1.4rem;color:#fff;font-size:1.15rem;margin:.6rem 0;}
div[data-testid="stButton"] button{border-radius:999px;border:1.5px solid var(--verde);color:var(--verde);background:#fff;font-weight:700;padding:.45rem 1.3rem;}
div[data-testid="stButton"] button:hover{background:var(--verde);color:#fff;border-color:var(--verde);}
div[data-testid="stButton"] button[kind="primary"]{background:var(--verde);color:#fff;}
button[data-baseweb="tab"]{font-weight:700;} button[data-baseweb="tab"][aria-selected="true"]{color:var(--verde);}
div[data-baseweb="tab-highlight"]{background-color:var(--olivo)!important;}
.stSlider [role="slider"]{background:var(--verde);}
.st-key-navbar{position:fixed;left:0;right:0;bottom:0;background:#ffffffef;backdrop-filter:blur(6px);border-top:1px solid var(--linea);padding:.5rem 2.2rem;z-index:999;}
.st-key-navbar div[data-testid="stHorizontalBlock"]{align-items:center;}
.dots{display:flex;gap:7px;justify-content:center;align-items:center;height:100%;}
.dots i{width:9px;height:9px;border-radius:50%;background:var(--linea);display:inline-block;}
.dots i.h{background:var(--olivoc);} .dots i.a{background:var(--verde);width:26px;border-radius:6px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================================================================================
# UTILIDADES
# ======================================================================================
def color_clase(nombre: str, idx: int = 0) -> str:
    n = nombre.lower()
    for k, v in COLORES_BASE.items():
        if n.startswith(k[:5]):
            return v
    return EXTRA[idx % len(EXTRA)]


def tinte(hex_color: str, f: float = .78) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def estilo(fig, h=420, leyenda=True):
    fig.update_layout(height=h, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                      font=dict(family="DM Sans, Segoe UI, sans-serif", color=C["bosque"], size=13),
                      margin=dict(l=10, r=10, t=45, b=10), showlegend=leyenda,
                      colorway=[C["verde"], C["olivo"], C["bosque"], C["hoja"], C["olivo_c"]],
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor="#EEF2E2", zerolinecolor="#DCE4CC")
    fig.update_yaxes(gridcolor="#EEF2E2", zerolinecolor="#DCE4CC")
    return fig


def plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def tabla(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def kpi(valor, etiqueta, clase=""):
    return f'<div class="kpi {clase}"><div class="v">{valor}</div><div class="l">{etiqueta}</div></div>'


def card(titulo, cuerpo, clase=""):
    return f'<div class="card {clase}"><h4>{titulo}</h4>{cuerpo}</div>'


def titulo_slide(titulo, sub=""):
    st.markdown(f'<div class="s-title">{titulo}</div>' + (f'<div class="s-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def pipeline_strip(activo: int):
    pasos = [("Datos", "Excel con comentarios"), ("Limpieza", "diccionario, stop words, stemming"),
             ("Vectorización", "BoW y TF-IDF"), ("PCA", "reducción y exploración"),
             ("Modelos", "Naive Bayes y SVM"), ("Evaluación", "métricas y errores")]
    html = "".join(f'<div class="{"on" if i == activo else ""}"><b>{a}</b>{b}</div>' for i, (a, b) in enumerate(pasos))
    st.markdown(f'<div class="pipe">{html}</div>', unsafe_allow_html=True)


# ======================================================================================
# CARGA Y CÁLCULOS (una sola vez)
# ======================================================================================
@st.cache_resource(show_spinner="Cargando artefactos y preparando las comparativas…")
def preparar():
    from scipy.special import softmax  # noqa: F401
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                                 silhouette_score)
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC, LinearSVC

    faltan = [f for f in ("train.csv", "test.csv", "modelos.joblib") if not (ART / f).exists()]
    if faltan:
        return {"error": f"No se encontraron {faltan} en {ART}. Ejecuta el notebook completo para generar los artefactos "
                         f"o define la variable de entorno ARTEFACTOS con la ruta correcta."}

    col_t, col_y = "comentario", "categoria"
    if (ART / "split.joblib").exists():
        sp = joblib.load(ART / "split.joblib")
        col_t, col_y = sp.get("col_texto", col_t), sp.get("col_etiqueta", col_y)
    train, test = pd.read_csv(ART / "train.csv"), pd.read_csv(ART / "test.csv")
    for d in (train, test):
        d[col_t] = d[col_t].astype(str)
        d["texto_proc"] = d["texto_proc"].fillna("").astype(str)
        if "texto_limpio" in d:
            d["texto_limpio"] = d["texto_limpio"].fillna("").astype(str)
        d["n_palabras"] = d[col_t].str.split().str.len()
    clases = sorted(set(train[col_y]) | set(test[col_y]))
    col = {c: color_clase(c, i) for i, c in enumerate(clases)}
    simb = {c: SIMBOLOS[i % len(SIMBOLOS)] for i, c in enumerate(clases)}

    modelos_raw = joblib.load(ART / "modelos.joblib")

    def nombre_modelo(p):
        vec, clf = p.named_steps["vec"], p.named_steps["clf"]
        rep = "TF-IDF" if isinstance(vec, TfidfVectorizer) else "BoW"
        fam = "Naive Bayes" if hasattr(clf, "feature_log_prob_") else "SVM"
        return f"{rep} + {fam}", rep, fam

    modelos, meta = {}, {}
    for _, p in modelos_raw.items():
        n, rep, fam = nombre_modelo(p)
        modelos[n] = p
        meta[n] = (rep, fam)
    orden = [n for n in ["BoW + Naive Bayes", "TF-IDF + Naive Bayes", "BoW + SVM", "TF-IDF + SVM"] if n in modelos]
    modelos = {n: modelos[n] for n in orden + [m for m in modelos if m not in orden]}

    Xtr, ytr, Xte, yte = train["texto_proc"], train[col_y], test["texto_proc"], test[col_y]
    pred = {n: m.predict(Xte) for n, m in modelos.items()}

    def mets(y, p):
        d = {"Accuracy": accuracy_score(y, p),
             "Precisión (macro)": precision_score(y, p, average="macro", zero_division=0),
             "Recall (macro)": recall_score(y, p, average="macro", zero_division=0),
             "F1 macro": f1_score(y, p, average="macro", zero_division=0)}
        for c, r in zip(clases, recall_score(y, p, labels=clases, average=None, zero_division=0)):
            d[f"Recall {c}"] = r
        return d
    res = pd.DataFrame({n: mets(yte, p) for n, p in pred.items()}).T

    # --- validación cruzada repetida sobre todo el corpus (mismos pliegues para los 4) ---
    X_all, y_all = pd.concat([Xtr, Xte]), pd.concat([ytr, yte])
    n_spl = int(max(2, min(5, y_all.value_counts().min())))
    cv = RepeatedStratifiedKFold(n_splits=n_spl, n_repeats=3, random_state=SEMILLA)
    cvr = {n: cross_validate(clone(m), X_all, y_all, cv=cv, scoring={"f1": "f1_macro", "acc": "accuracy"}, n_jobs=1)
           for n, m in modelos.items()}
    cv_f1 = {n: r["test_f1"] for n, r in cvr.items()}
    cv_tab = pd.DataFrame({n: {"F1 macro (media)": r["test_f1"].mean(), "F1 macro (±)": r["test_f1"].std(),
                               "Accuracy (media)": r["test_acc"].mean(), "Accuracy (±)": r["test_acc"].std()}
                           for n, r in cvr.items()}).T

    def mejor(familia):
        c = [n for n in modelos if meta[n][1] == familia]
        return max(c, key=lambda n: cv_tab.loc[n, "F1 macro (media)"]) if c else None
    best_nb, best_svm = mejor("Naive Bayes"), mejor("SVM")
    ganador = cv_tab.sort_values("F1 macro (media)", ascending=False).index[0]

    cal = None
    if (ART / "svm_calibrado.joblib").exists():
        try:
            cal = joblib.load(ART / "svm_calibrado.joblib")
            if cal is None or list(cal.classes_) != clases:
                cal = None
        except Exception:
            cal = None

    # --- Fase 2: vectorizadores base (los mismos de la Fase 2 del notebook) ---
    bow, tfidf = CountVectorizer().fit(Xtr), TfidfVectorizer().fit(Xtr)
    B_tr, T_tr = bow.transform(Xtr), tfidf.transform(Xtr)
    vocab = bow.get_feature_names_out()
    df_term = np.asarray((B_tr > 0).sum(axis=0)).ravel()

    # términos distintivos por clase (log-odds de la clase frente al resto)
    disc = {}
    for c in clases:
        m = (ytr == c).values
        a, b = np.asarray(B_tr[m].sum(axis=0)).ravel() + 1, np.asarray(B_tr[~m].sum(axis=0)).ravel() + 1
        lo = np.log(a / a.sum()) - np.log(b / b.sum())
        disc[c] = pd.DataFrame({"término": vocab, "log_odds": lo, "frec_clase": a - 1, "frec_resto": b - 1})

    # --- Fase 3: PCA (ajustado solo con entrenamiento) ---
    Ttr, Tte = T_tr.toarray(), tfidf.transform(Xte).toarray()
    pca = PCA(random_state=SEMILLA).fit(Ttr)
    var = pca.explained_variance_ratio_
    k3 = min(3, Ttr.shape[1])
    Z_tr, Z_te = pca.transform(Ttr)[:, :k3], pca.transform(Tte)[:, :k3]
    acum = np.cumsum(var)
    n80 = int(np.argmax(acum >= .80) + 1)
    sil = {"TF-IDF original": float(silhouette_score(Ttr, ytr)), "PCA 2 comp.": float(silhouette_score(Z_tr[:, :2], ytr))}
    sil[f"PCA {n80} comp. (≥80 %)"] = float(silhouette_score(PCA(n_components=max(n80, 2), random_state=SEMILLA).fit_transform(Ttr), ytr))

    # --- Fase 4: frontera SVM ilustrativa en 2D (solo conceptual) ---
    svm2d = SVC(kernel="linear", C=1.0).fit(Z_tr[:, :2], ytr)
    aux = Pipeline([("vec", TfidfVectorizer()), ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=SEMILLA))]).fit(Xtr, ytr)
    terms_aux = aux.named_steps["vec"].get_feature_names_out()
    coef_aux = aux.named_steps["clf"].coef_
    if coef_aux.shape[0] == 1:
        coef_aux = np.vstack([-coef_aux[0], coef_aux[0]])
    coef = {c: coef_aux[i] for i, c in enumerate(aux.named_steps["clf"].classes_)}

    pre = cargar_preprocesador(ART)
    nuevos = pd.read_csv(ART / "cinco_nuevos.csv") if (ART / "cinco_nuevos.csv").exists() else None

    return dict(col_t=col_t, col_y=col_y, train=train, test=test, clases=clases, col=col, simb=simb,
                modelos=modelos, meta=meta, pred=pred, res=res, cv_f1=cv_f1, cv_tab=cv_tab, best_nb=best_nb,
                best_svm=best_svm, ganador=ganador, cal=cal, bow=bow, tfidf=tfidf, B_tr=B_tr, T_tr=T_tr, vocab=vocab,
                df_term=df_term, disc=disc, pca=pca, var=var, acum=acum, Z_tr=Z_tr, Z_te=Z_te, n80=n80, sil=sil,
                svm2d=svm2d, coef=coef, terms_aux=terms_aux, pre=pre, nuevos=nuevos, n_spl=n_spl,
                n_total=len(train) + len(test), confusion_matrix=confusion_matrix)


def softmax_filas(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probas(D, nombre, textos_proc):
    """Probabilidades por clase (orden de D['clases']). NB: probabilidades; SVM: calibradas si hay, si no softmax."""
    m = D["modelos"][nombre]
    fam = D["meta"][nombre][1]
    if fam == "Naive Bayes":
        return m.predict_proba(textos_proc)
    if D["cal"] is not None and nombre == D["best_svm"]:
        return D["cal"].predict_proba(textos_proc)
    d = m.decision_function(textos_proc)
    if d.ndim == 1:
        d = np.c_[-d, d]
    return softmax_filas(d)


# ======================================================================================
# DIAPOSITIVAS
# ======================================================================================
def s_portada(D):
    integ = " · ".join(INTEGRANTES)
    st.markdown(f"""
    <div class="hero">
      <p>{CURSO}</p>
      <h1>{TITULO}</h1>
      <p>{SUBTITULO}</p>
      <div class="chips"><span class="chip">{D['n_total']} comentarios</span>
      <span class="chip">{len(D['clases'])} categorías: {', '.join(D['clases'])}</span>
      <span class="chip">4 modelos comparados</span><span class="chip">sin redes neuronales</span></div>
      <p style="margin-top:2rem;font-size:1rem">{integ}<br>{DOCENTE}</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    pipeline_strip(-1)
    st.caption("Usa las flechas del teclado (← →) o los botones inferiores para avanzar por la presentación.")


def s_problema(D):
    titulo_slide("Problema y objetivo",
                 "La universidad recibe muchos comentarios escritos y necesita clasificarlos automáticamente para atender primero los que importan.")
    a, b = st.columns([1.15, 1])
    with a:
        st.markdown(card("Pregunta de investigación",
                         "<p>¿Qué <b>representación</b> de los textos y qué <b>algoritmo de clasificación</b> permiten obtener "
                         "mejores resultados en la clasificación automática de comentarios?</p>", "v"), unsafe_allow_html=True)
        st.markdown(card("Objetivo general",
                         "<p>Desarrollar y evaluar un clasificador de textos mediante representación vectorial, reducción de "
                         "dimensionalidad y aprendizaje automático, comparando el desempeño de <b>Naive Bayes</b> y <b>SVM</b>.</p>"),
                    unsafe_allow_html=True)
        st.markdown(card("Categorías", "<p>" + " · ".join(f"<b>{c}</b>" for c in D["clases"]) +
                         "</p><p>Las quejas (negativas) son las que requieren atención prioritaria.</p>", "b"), unsafe_allow_html=True)
    with b:
        st.markdown("##### Los cinco temas aplicados")
        for i, (t, d) in enumerate([("Bag of Words", "frecuencias de palabras por comentario"),
                                    ("TF-IDF", "peso mayor a lo característico, menor a lo común"),
                                    ("PCA", "reducción de dimensiones y exploración visual"),
                                    ("Clasificación bayesiana", "probabilidades de palabras por categoría"),
                                    ("SVM", "frontera de máximo margen")], 1):
            st.markdown(f'<div class="paso"><div class="n">{i}</div><div><div class="r"><b>{t}</b></div><div class="t">{d}</div></div></div>',
                        unsafe_allow_html=True)
    st.markdown("##### Flujo de trabajo")
    pipeline_strip(-1)


def s_datos(D):
    titulo_slide("Los datos", "Corpus de comentarios universitarios ya clasificados, separado en 80 % entrenamiento y 20 % prueba.")
    tr, te, y = D["train"], D["test"], D["col_y"]
    c = st.columns(4)
    for col, (v, l, o) in zip(c, [(D["n_total"], "comentarios en total", ""), (len(tr), "entrenamiento (80 %)", "o"),
                                  (len(te), "prueba (20 %)", ""), (len(D["clases"]), "categorías", "o")]):
        col.markdown(kpi(v, l, o), unsafe_allow_html=True)
    st.markdown("")
    a, b = st.columns(2)
    todo = pd.concat([tr.assign(conjunto="Entrenamiento"), te.assign(conjunto="Prueba")])
    cnt = todo.groupby([y, "conjunto"]).size().reset_index(name="n")
    with a:
        fig = px.bar(cnt, x=y, y="n", color="conjunto", barmode="group", text="n", title="Distribución por categoría",
                     color_discrete_map={"Entrenamiento": C["verde"], "Prueba": C["olivo_c"]})
        fig.update_traces(textposition="outside")
        plot(estilo(fig, 390))
    with b:
        fig = px.violin(todo, x=y, y="n_palabras", color=y, box=True, points="all", color_discrete_map=D["col"],
                        title="Longitud de los comentarios (palabras)")
        plot(estilo(fig, 390, leyenda=False))
    tot = todo[y].value_counts()
    razon = tot.max() / tot.min()
    st.markdown(card("Balance de clases",
                     f"<p>Razón entre la clase más grande y la más pequeña: <b>{razon:.2f} : 1</b>. " +
                     ("Hay desbalance: por eso no basta la <i>accuracy</i>; se reportan F1 macro y recall por clase."
                      if razon > 1.5 else "Corpus razonablemente balanceado; aun así se reportan F1 macro y recall por clase.") + "</p>",
                     "v"), unsafe_allow_html=True)
    sel = st.selectbox("Explorar comentarios de la categoría", ["(todas)"] + D["clases"])
    ver = todo if sel == "(todas)" else todo[todo[y] == sel]
    tabla(ver[[D["col_t"], y, "conjunto"]].reset_index(drop=True), height=240)


def s_limpieza(D):
    titulo_slide("Preparación y limpieza del texto",
                 "Procesamiento clásico en español con diccionario propio: transparente, reproducible y sin redes neuronales.")
    pipeline_strip(1)
    t1, t2, t3 = st.tabs(["Paso a paso en vivo", "Antes → después", "Diccionario y stop words"])
    pre = D["pre"]
    with t1:
        ejemplo = D["test"][D["col_t"]].iloc[0]
        txt = st.text_input("Escribe un comentario para ver cómo se transforma", value=ejemplo, key="txt_limpieza")
        pasos, final = pre.pasos(txt)
        html = "".join(f'<div class="paso"><div class="n">{i}</div><div><div class="t">{n}</div><div class="r">{r if r else "—"}</div></div></div>'
                       for i, (n, r) in enumerate(pasos))
        st.markdown(html, unsafe_allow_html=True)
    with t2:
        tr = D["train"].sample(min(8, len(D["train"])), random_state=SEMILLA)
        cols = [D["col_t"], "texto_limpio", "texto_proc", D["col_y"]] if "texto_limpio" in tr else [D["col_t"], "texto_proc", D["col_y"]]
        tabla(tr[cols].rename(columns={D["col_t"]: "Original", "texto_limpio": "Limpio (sin stemming)",
                                       "texto_proc": "Procesado (stem + negación)", D["col_y"]: "Categoría"}).reset_index(drop=True))
        voc_o = {w for t in D["train"][D["col_t"]].str.lower() for w in re.findall(r"\w+", t)}
        voc_p = {w for t in D["train"]["texto_proc"] for w in t.split()}
        vv = pd.DataFrame({"etapa": ["Texto original", "Procesado"], "palabras distintas": [len(voc_o), len(voc_p)]})
        fig = px.bar(vv, x="palabras distintas", y="etapa", orientation="h", text="palabras distintas", title="Tamaño del vocabulario",
                     color="etapa", color_discrete_sequence=[C["olivo"], C["verde"]])
        plot(estilo(fig, 230, leyenda=False))
    with t3:
        a, b = st.columns(2)
        with a:
            st.markdown(card("Stop words con excepciones",
                             "<p>Se eliminan las stop words del español <b>excepto</b> las que cambian el sentido: "
                             "<b>no, nunca, jamás, tampoco, sin, ni, pero, aunque, sino, muy, más, menos, nada, poco, mucho, bien, mal, mejor, peor</b>.</p>"
                             "<p>«No me gustó» y «me gustó» son opuestos: la negación se conserva y se fusiona con la palabra siguiente (<code>no_gust</code>).</p>", "v"),
                        unsafe_allow_html=True)
            st.markdown(card("Otras decisiones", "<ul><li>Minúsculas, sin URLs, números ni puntuación.</li>"
                             "<li>Sin tildes (se conserva la ñ) y alargamientos colapsados («buenooo» → «bueno»).</li>"
                             "<li>Emojis y emoticones convertidos en tokens de polaridad.</li>"
                             "<li>Stemming Snowball (reglas, no neuronal).</li></ul>"), unsafe_allow_html=True)
        with b:
            d = pd.DataFrame(sorted(pre.diccionario.items()), columns=["escrito", "se normaliza como"])
            st.markdown(f"**Diccionario propio** — {len(d)} entradas (jerga, abreviaturas y errores ortográficos frecuentes)")
            tabla(d, height=330)


def s_bow(D):
    titulo_slide("Bag of Words", "Cada comentario se convierte en un vector con la frecuencia de cada palabra del vocabulario de entrenamiento.")
    pipeline_strip(2)
    n, v = D["B_tr"].shape
    disp = 1 - D["B_tr"].nnz / (n * v)
    c = st.columns(3)
    c[0].markdown(kpi(f"{n} × {v}", "documentos × términos (entrenamiento)"), unsafe_allow_html=True)
    c[1].markdown(kpi(f"{disp:.1%}", "de la matriz son ceros (dispersión)", "o"), unsafe_allow_html=True)
    c[2].markdown(kpi(len(D["clases"]), "categorías a distinguir"), unsafe_allow_html=True)
    st.markdown("")
    a, b = st.columns([1.15, 1])
    with a:
        k = st.slider("Documentos a mostrar", 3, 12, 6, key="bow_k")
        tot = np.asarray(D["B_tr"][:k].sum(axis=0)).ravel()
        idx = np.argsort(-tot)[:14]
        M = D["B_tr"][:k][:, idx].toarray()
        fig = px.imshow(M, x=D["vocab"][idx], y=[f"doc{i} · {D['train'][D['col_y']].iloc[i]}" for i in range(k)],
                        color_continuous_scale=[[0, "#FFFFFF"], [1, C["verde"]]], text_auto=True, aspect="auto",
                        title="Matriz BoW (términos más frecuentes en esos documentos)")
        plot(estilo(fig, 420, leyenda=False))
    with b:
        cl = st.selectbox("Palabras que distinguen a la categoría", D["clases"], key="bow_cl")
        d = D["disc"][cl]
        d = d[d.frec_clase >= 1].sort_values("log_odds", ascending=False).head(12).iloc[::-1]
        fig = px.bar(d, x="log_odds", y="término", orientation="h", title=f"Más características de «{cl}»",
                     color_discrete_sequence=[D["col"][cl]])
        plot(estilo(fig, 420, leyenda=False))
    st.markdown(card("Cómo leerlo", "<p>Cada fila es un comentario, cada columna una palabra y cada celda cuántas veces aparece. "
                     "El orden se pierde («bolsa de palabras»), por eso el marcado de negación es importante.</p>"), unsafe_allow_html=True)


def s_tfidf(D):
    titulo_slide("TF-IDF", "Pesa más lo característico de cada comentario y menos lo que aparece en casi todos.")
    pipeline_strip(2)
    st.latex(r"\text{tf-idf}(t,d)=\text{tf}(t,d)\cdot\text{idf}(t),\qquad \text{idf}(t)=\ln\frac{1+n}{1+\text{df}(t)}+1")
    a, b = st.columns([1.1, 1])
    with a:
        tr = D["train"]
        i = st.selectbox("Documento", list(range(len(tr))), format_func=lambda j: f"doc{j} [{tr[D['col_y']].iloc[j]}] — {tr[D['col_t']].iloc[j][:60]}", key="tf_doc")
        fila = D["T_tr"][i].toarray().ravel()
        cuenta = D["B_tr"][i].toarray().ravel()
        idx = np.argsort(-fila)[:8]
        t = pd.DataFrame({"término": D["vocab"][idx], "frecuencia BoW": cuenta[idx], "idf": D["tfidf"].idf_[idx].round(3),
                          "peso TF-IDF": fila[idx].round(3)})
        fig = go.Figure(go.Bar(x=t["peso TF-IDF"][::-1], y=t["término"][::-1], orientation="h", marker_color=C["verde"],
                               text=t["peso TF-IDF"][::-1], textposition="outside"))
        fig.update_layout(title="Términos con mayor peso TF-IDF en este documento")
        plot(estilo(fig, 360, leyenda=False))
        tabla(t, hide_index=True)
    with b:
        dfx = pd.DataFrame({"término": D["vocab"], "docs con el término": D["df_term"], "idf": D["tfidf"].idf_})
        fig = px.scatter(dfx, x="docs con el término", y="idf", hover_name="término", title="IDF según en cuántos documentos aparece",
                         color_discrete_sequence=[C["olivo"]])
        fig.update_traces(marker=dict(size=9, opacity=.7))
        plot(estilo(fig, 330, leyenda=False))
        comunes = dfx.sort_values("idf").head(6)["término"].tolist()
        raros = dfx.sort_values("idf", ascending=False).head(6)["término"].tolist()
        st.markdown(card("¿Una palabra omnipresente carece de importancia?",
                         "<p><b>No.</b> Su IDF es bajo porque <i>no distingue</i> un comentario de otro, no porque carezca de significado. "
                         f"Más comunes: <code>{'</code> <code>'.join(comunes)}</code>. Más raros: <code>{'</code> <code>'.join(raros)}</code>.</p>", "v"),
                    unsafe_allow_html=True)


def s_pca(D):
    titulo_slide("PCA: ¿se separan las categorías?", "Análisis exploratorio sobre TF-IDF, ajustado solo con entrenamiento.")
    pipeline_strip(3)
    t1, t2, t3 = st.tabs(["Dispersión de los comentarios", "Varianza explicada", "Lectura e interpretación"])
    y_tr, y_te = D["train"][D["col_y"]].values, D["test"][D["col_y"]].values
    with t1:
        modo = st.radio("Vista", ["2D", "3D"] if D["Z_tr"].shape[1] >= 3 else ["2D"], horizontal=True, key="pca_modo")
        ver_test = st.toggle("Mostrar comentarios de prueba", value=True, key="pca_test")
        v = D["var"]
        if modo == "2D":
            fig = go.Figure()
            for c in D["clases"]:
                m = y_tr == c
                fig.add_trace(go.Scatter(x=D["Z_tr"][m, 0], y=D["Z_tr"][m, 1], mode="markers", name=f"{c} (train)",
                                         marker=dict(color=D["col"][c], size=11, symbol=D["simb"][c], line=dict(color="white", width=1)),
                                         text=D["train"][D["col_t"]][m], hoverinfo="text"))
                if ver_test:
                    m2 = y_te == c
                    fig.add_trace(go.Scatter(x=D["Z_te"][m2, 0], y=D["Z_te"][m2, 1], mode="markers", name=f"{c} (test)",
                                             marker=dict(color="white", size=13, symbol=D["simb"][c], line=dict(color=D["col"][c], width=2.5)),
                                             text=D["test"][D["col_t"]][m2], hoverinfo="text"))
            fig.update_layout(xaxis_title=f"PC1 ({v[0]:.1%})", yaxis_title=f"PC2 ({v[1]:.1%})", title="Comentarios en 2 componentes principales")
            plot(estilo(fig, 520))
        else:
            fig = go.Figure()
            for c in D["clases"]:
                m = y_tr == c
                fig.add_trace(go.Scatter3d(x=D["Z_tr"][m, 0], y=D["Z_tr"][m, 1], z=D["Z_tr"][m, 2], mode="markers", name=f"{c} (train)",
                                           marker=dict(color=D["col"][c], size=5), text=D["train"][D["col_t"]][m], hoverinfo="text"))
                if ver_test:
                    m2 = y_te == c
                    fig.add_trace(go.Scatter3d(x=D["Z_te"][m2, 0], y=D["Z_te"][m2, 1], z=D["Z_te"][m2, 2], mode="markers", name=f"{c} (test)",
                                               marker=dict(color="white", size=6, line=dict(color=D["col"][c], width=3)),
                                               text=D["test"][D["col_t"]][m2], hoverinfo="text"))
            fig.update_layout(scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"), title="Primeras 3 componentes")
            plot(estilo(fig, 560))
    with t2:
        v, ac = D["var"], D["acum"]
        k = st.slider("Número de componentes", 1, min(len(v), 60), min(D["n80"], 60), key="pca_k")
        c = st.columns(3)
        c[0].markdown(kpi(f"{ac[k - 1]:.1%}", f"de la variabilidad con {k} componentes"), unsafe_allow_html=True)
        c[1].markdown(kpi(D["n80"], "componentes para llegar al 80 %", "o"), unsafe_allow_html=True)
        c[2].markdown(kpi(D["T_tr"].shape[1], "dimensiones originales (términos)"), unsafe_allow_html=True)
        m = min(25, len(v))
        fig = go.Figure()
        fig.add_bar(x=list(range(1, m + 1)), y=v[:m] * 100, name="Varianza por componente", marker_color=C["olivo_c"])
        fig.add_scatter(x=list(range(1, m + 1)), y=ac[:m] * 100, name="Varianza acumulada", mode="lines+markers", line=dict(color=C["verde"], width=3))
        fig.add_vline(x=k, line_dash="dash", line_color=C["bosque"])
        fig.update_layout(xaxis_title="Componente", yaxis_title="% de varianza", title="Varianza explicada")
        plot(estilo(fig, 400))
        tabla(pd.DataFrame({"componente": [f"PC{i + 1}" for i in range(min(10, len(v)))],
                            "varianza explicada %": (v[:10] * 100).round(2), "acumulada %": (ac[:10] * 100).round(2)}), hide_index=True)
    with t3:
        s = D["sil"]
        sil_df = pd.DataFrame({"representación": list(s), "silhouette": list(s.values())})
        a, b = st.columns([1, 1.1])
        with a:
            fig = px.bar(sil_df, x="silhouette", y="representación", orientation="h", text=sil_df["silhouette"].round(3),
                         title="Separación entre categorías (silhouette)", color_discrete_sequence=[C["olivo"]])
            plot(estilo(fig, 280, leyenda=False))
        with b:
            s2 = s["PCA 2 comp."]
            lectura = ("muestran cierta separación parcial" if s2 > 0.10 else "se mezclan en gran medida en el plano")
            st.markdown(card("Lectura del gráfico",
                             f"<p>Con 2 componentes solo se retiene <b>{D['acum'][1]:.1%}</b> de la variabilidad y las categorías {lectura} (silhouette {s2:.2f}).</p>", "v"),
                        unsafe_allow_html=True)
            st.markdown(card("¿Entonces ningún clasificador podrá separarlas?",
                             "<p><b>No.</b> PCA es no supervisado: maximiza varianza, no separación, y esta proyección descarta casi toda la información. "
                             "Los clasificadores trabajan en el espacio completo. Por eso PCA se usa aquí como <b>exploración</b>.</p>"), unsafe_allow_html=True)


def s_nb(D):
    titulo_slide("Clasificación bayesiana: Naive Bayes",
                 "Usa las probabilidades de las palabras en cada categoría y elige la de mayor probabilidad a posteriori.")
    pipeline_strip(4)
    nb = D["best_nb"]
    if nb is None:
        st.warning("No hay un modelo Naive Bayes en modelos.joblib.")
        return
    pipe = D["modelos"][nb]
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    st.latex(r"\log P(c\mid d)=\log P(c)+\sum_{w\in d} n_{w,d}\,\log P(w\mid c)\quad\Rightarrow\quad \hat{c}=\arg\max_c\ P(c\mid d)")
    a, b = st.columns([1, 1.1])
    terms = vec.get_feature_names_out()
    flp = clf.feature_log_prob_
    with a:
        st.markdown(f"**Modelo usado:** {nb}")
        cl = st.selectbox("Palabras que más empujan hacia", D["clases"], key="nb_cl")
        i = list(clf.classes_).index(cl)
        otros = [j for j in range(len(clf.classes_)) if j != i]
        sc = flp[i] - flp[otros].mean(axis=0)
        top = np.argsort(-sc)[:12][::-1]
        fig = go.Figure(go.Bar(x=sc[top], y=terms[top], orientation="h", marker_color=D["col"][cl]))
        fig.update_layout(title=f"Mayor log P(w|{cl}) frente al resto")
        plot(estilo(fig, 430, leyenda=False))
    with b:
        st.markdown("**Calcula la probabilidad de un comentario**")
        base = D["test"][D["col_t"]].iloc[0]
        txt = st.text_input("Comentario", value=base, key="nb_txt")
        proc = D["pre"].limpiar(txt)
        st.caption(f"Texto procesado: `{proc or '—'}`")
        x = vec.transform([proc])
        logpost = np.asarray(clf.class_log_prior_ + x @ flp.T).ravel()
        post = np.exp(logpost - logpost.max()); post /= post.sum()
        fig = go.Figure(go.Bar(x=list(clf.classes_), y=post, marker_color=[D["col"][c] for c in clf.classes_],
                               text=[f"{p:.1%}" for p in post], textposition="outside"))
        fig.update_layout(title="P(clase | comentario)", yaxis_range=[0, 1.12])
        plot(estilo(fig, 280, leyenda=False))
        presentes = np.nonzero(x.toarray().ravel())[0]
        if len(presentes):
            filas = {"término": terms[presentes], "valor": x.toarray().ravel()[presentes].round(3)}
            for j, c in enumerate(clf.classes_):
                filas[f"log P(w|{c})"] = flp[j][presentes].round(2)
            tabla(pd.DataFrame(filas), hide_index=True, height=170)
        else:
            st.info("Ninguna palabra del comentario está en el vocabulario de entrenamiento; solo cuenta la probabilidad a priori.")
        pri = " · ".join(f"P({c})={np.exp(p):.2f}" for c, p in zip(clf.classes_, clf.class_log_prior_))
        st.caption(f"Probabilidad a priori: {pri}")


def s_svm(D):
    titulo_slide("SVM: frontera de decisión", "Busca el hiperplano de máximo margen; solo los vectores de soporte lo determinan.")
    pipeline_strip(4)
    a, b = st.columns([1.15, 1])
    with a:
        svm = D["svm2d"]
        Z = D["Z_tr"][:, :2]
        xs = np.linspace(Z[:, 0].min() - .15, Z[:, 0].max() + .15, 220)
        ys = np.linspace(Z[:, 1].min() - .15, Z[:, 1].max() + .15, 220)
        xx, yy = np.meshgrid(xs, ys)
        cl = list(svm.classes_)
        zz = np.vectorize(cl.index)(svm.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        k = len(cl)
        cs = []
        for i, c in enumerate(cl):
            cs += [[i / k, tinte(D["col"][c])], [(i + 1) / k, tinte(D["col"][c])]]
        fig = go.Figure(go.Heatmap(x=xs, y=ys, z=zz, zmin=-0.5, zmax=k - 0.5, colorscale=cs, showscale=False, hoverinfo="skip"))
        yt = D["train"][D["col_y"]].values
        for c in cl:
            m = yt == c
            fig.add_trace(go.Scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", name=c,
                                     marker=dict(color=D["col"][c], size=10, symbol=D["simb"][c], line=dict(color="white", width=1))))
        sv = svm.support_vectors_
        fig.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode="markers", name="vectores de soporte",
                                 marker=dict(color="rgba(0,0,0,0)", size=17, line=dict(color=C["bosque"], width=1.8))))
        fig.update_layout(title="Regiones de decisión de un SVM lineal sobre PC1–PC2 (ilustración conceptual)",
                          xaxis_title="PC1", yaxis_title="PC2")
        plot(estilo(fig, 520))
        st.caption(f"{len(sv)} vectores de soporte de {len(Z)} puntos de entrenamiento. Los modelos reales se entrenan en el espacio completo, no en 2D.")
    with b:
        st.markdown(card("Idea central",
                         "<ul><li>El <b>hiperplano</b> w·x + b = 0 separa las categorías.</li>"
                         "<li>El <b>margen</b> es la distancia a los puntos más cercanos: se busca maximizarlo.</li>"
                         "<li><b>C</b> equilibra margen ancho y errores de entrenamiento.</li>"
                         "<li>Con varias categorías se combinan fronteras «una contra el resto».</li></ul>", "v"), unsafe_allow_html=True)
        if D["best_svm"]:
            clf = D["modelos"][D["best_svm"]].named_steps["clf"]
            st.markdown(f"**Mejor SVM:** {D['best_svm']} — kernel `{getattr(clf, 'kernel', 'linear')}`, C = `{getattr(clf, 'C', '?')}`")
        cs_ = st.selectbox("Términos con más peso hacia", D["clases"], key="svm_cl")
        w = D["coef"][cs_]
        top = np.argsort(-w)[:10][::-1]
        fig = go.Figure(go.Bar(x=w[top], y=D["terms_aux"][top], orientation="h", marker_color=D["col"][cs_]))
        fig.update_layout(title=f"Coeficientes (SVM lineal sobre TF-IDF) → «{cs_}»")
        plot(estilo(fig, 330, leyenda=False))


def s_comparativa(D):
    titulo_slide("Comparación de los cuatro modelos", "Mismo conjunto de prueba, mismos pliegues de validación cruzada.")
    pipeline_strip(5)
    res, cvt = D["res"], D["cv_tab"]
    t1, t2, t3, t4 = st.tabs(["Conjunto de prueba", "Validación cruzada 5×3", "Matrices de confusión", "Recall por categoría"])
    with t1:
        met = st.radio("Métrica", ["F1 macro", "Accuracy", "Precisión (macro)", "Recall (macro)"], horizontal=True, key="cmp_met")
        d = res.reset_index().rename(columns={"index": "modelo"})
        d["Representación"] = [D["meta"][m][0] for m in d["modelo"]]
        d["Clasificador"] = [D["meta"][m][1] for m in d["modelo"]]
        fig = px.bar(d, x="Clasificador", y=met, color="Representación", barmode="group", text=d[met].round(3),
                     color_discrete_map={"BoW": C["olivo"], "TF-IDF": C["verde"]}, title=f"{met} en el conjunto de prueba")
        fig.update_traces(textposition="outside"); fig.update_yaxes(range=[0, 1.1])
        plot(estilo(fig, 380))
        cols = ["Accuracy", "Precisión (macro)", "Recall (macro)", "F1 macro"]
        sty = res[cols].round(3)
        try:
            tabla(sty.style.background_gradient(cmap="Greens", axis=None).format("{:.3f}"))
        except Exception:
            tabla(sty)
        st.caption(f"Conjunto de prueba: {len(D['test'])} comentarios. Con una muestra pequeña, una o dos respuestas cambian mucho las cifras: por eso se confirma con validación cruzada.")
    with t2:
        fig = go.Figure()
        for n, v in D["cv_f1"].items():
            fig.add_trace(go.Box(y=v, name=n, boxmean=True, marker_color=C["verde"] if "TF-IDF" in n else C["olivo"]))
        fig.update_layout(yaxis_title="F1 macro por pliegue", title=f"{D['n_spl']} pliegues × 3 repeticiones (sobre todo el corpus)")
        plot(estilo(fig, 400, leyenda=False))
        tabla(cvt.round(3))
        st.markdown(card("Por qué es una comparación justa",
                         "<p>Los cuatro modelos se evalúan sobre <b>exactamente los mismos pliegues</b> y el vectorizador se ajusta solo con los datos de entrenamiento de cada pliegue. "
                         "Así la diferencia entre BoW y TF-IDF no se debe a una partición distinta.</p>", "v"), unsafe_allow_html=True)
    with t3:
        m = st.selectbox("Modelo", list(D["modelos"]), index=list(D["modelos"]).index(D["ganador"]), key="cm_m")
        norm = st.toggle("Mostrar porcentajes por fila", value=False, key="cm_norm")
        cm = D["confusion_matrix"](D["test"][D["col_y"]], D["pred"][m], labels=D["clases"])
        z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if norm else cm
        fig = px.imshow(z, x=D["clases"], y=D["clases"], text_auto=".0%" if norm else True, aspect="auto",
                        color_continuous_scale=[[0, "#FFFFFF"], [1, C["verde"]]], labels=dict(x="Predicho", y="Real"), title=f"Matriz de confusión — {m}")
        plot(estilo(fig, 420, leyenda=False))
    with t4:
        rc = [c for c in res.columns if c.startswith("Recall ") and "macro" not in c]
        d = res[rc].copy(); d.columns = [c.replace("Recall ", "") for c in rc]
        fig = px.imshow(d.round(2), text_auto=True, aspect="auto", color_continuous_scale=[[0, "#FFFFFF"], [1, C["verde"]]],
                        title="Recall por categoría (cuántos comentarios de cada clase se detectan)")
        plot(estilo(fig, 340, leyenda=False))
        st.caption("Que una queja se clasifique como positiva es el error más costoso: se vigila especialmente el recall de la categoría negativa.")


def s_por_comentario(D):
    titulo_slide("Comentario por comentario: ¿Naive Bayes o SVM?",
                 f"Se comparan la mejor versión de cada familia: {D['best_nb']} frente a {D['best_svm']}.")
    pipeline_strip(5)
    te, y = D["test"], D["col_y"]
    nb, sv = D["best_nb"], D["best_svm"]
    Xte = te["texto_proc"]
    pn, ps = probas(D, nb, Xte), probas(D, sv, Xte)
    cl = D["clases"]
    cmp = pd.DataFrame({"comentario": te[D["col_t"]], "real": te[y], "NB": D["pred"][nb], "SVM": D["pred"][sv],
                        "conf. NB": pn.max(axis=1), "conf. SVM": ps.max(axis=1)})
    cmp["acierta NB"], cmp["acierta SVM"] = cmp["NB"] == cmp["real"], cmp["SVM"] == cmp["real"]

    def quien(r):
        if r["acierta NB"] and r["acierta SVM"]: return "ambos aciertan"
        if r["acierta NB"]: return "Naive Bayes"
        if r["acierta SVM"]: return "SVM"
        return "ninguno"
    cmp["aplica mejor"] = cmp.apply(quien, axis=1)
    cuenta = cmp["aplica mejor"].value_counts()
    a, b = st.columns([1, 1.4])
    with a:
        fig = px.pie(values=cuenta.values, names=cuenta.index, hole=.55, title="¿Quién acierta en el test?",
                     color=cuenta.index, color_discrete_map={"ambos aciertan": C["verde"], "Naive Bayes": C["olivo"], "SVM": C["hoja"], "ninguno": C["bosque"]})
        plot(estilo(fig, 340))
    with b:
        filtro = st.radio("Mostrar", ["Todos", "Solo desacuerdos o errores"], horizontal=True, key="cmp_f")
        v = cmp if filtro == "Todos" else cmp[cmp["aplica mejor"] != "ambos aciertan"]
        if v.empty:
            st.success("En el conjunto de prueba ambos modelos aciertan en todos los comentarios.")
        else:
            tabla(v.drop(columns=["acierta NB", "acierta SVM"]).round(3).reset_index(drop=True), height=300)
    st.markdown("##### Detalle de un comentario")
    i = st.selectbox("Comentario de prueba", list(range(len(te))), format_func=lambda j: f"[{te[y].iloc[j]}] {te[D['col_t']].iloc[j][:80]}", key="cmp_i")
    d = pd.DataFrame({"categoría": cl * 2, "modelo": ["Naive Bayes"] * len(cl) + ["SVM"] * len(cl),
                      "probabilidad": list(pn[i]) + list(ps[i])})
    fig = px.bar(d, x="categoría", y="probabilidad", color="modelo", barmode="group", color_discrete_map={"Naive Bayes": C["olivo"], "SVM": C["verde"]},
                 title=f"Real: {te[y].iloc[i]} · NB → {cmp['NB'].iloc[i]} · SVM → {cmp['SVM'].iloc[i]}")
    fig.update_yaxes(range=[0, 1.05])
    plot(estilo(fig, 320))
    st.caption("En el test se conoce la etiqueta real, así que «aplica mejor» es un hecho observado. En comentarios nuevos solo se puede comparar la confianza, que es una estimación.")


EJEMPLOS = ["Excelente atención en la biblioteca, el personal fue muy amable y rápido.",
            "Pésimo servicio, llevo semanas esperando una respuesta y nadie me atiende.",
            "¿Cuándo comienza el periodo de inscripciones para el próximo semestre?",
            "No me gustó nada cómo explican los profesores, no se entiende.",
            "Qué maravilla, otra vez sin internet en el laboratorio, justo lo que necesitaba."]


def s_vivo(D):
    titulo_slide("Clasificador en vivo", "Escribe un comentario y mira cómo lo procesa y lo clasifica cada modelo.")
    st.session_state.setdefault("vivo_txt", EJEMPLOS[0])
    cols = st.columns(len(EJEMPLOS))
    for i, (c, e) in enumerate(zip(cols, EJEMPLOS)):
        if c.button(f"Ejemplo {i + 1}", key=f"ej{i}"):
            st.session_state["vivo_txt"] = e
    txt = st.text_area("Comentario", key="vivo_txt", height=100)
    if not txt.strip():
        st.info("Escribe un comentario o elige un ejemplo para clasificarlo.")
        return
    pasos, proc = D["pre"].pasos(txt)
    if not proc.strip():
        st.warning("Tras la limpieza no queda ninguna palabra con contenido. Prueba con un comentario más largo.")
        return
    nombres = list(D["modelos"])
    P = {n: probas(D, n, [proc])[0] for n in nombres}
    cl = D["clases"]
    preds = {n: cl[int(np.argmax(P[n]))] for n in nombres}
    cols = st.columns(len(nombres))
    for c, n in zip(cols, nombres):
        pr = preds[n]
        c.markdown(f'<div class="res" style="border-top:5px solid {D["col"][pr]}"><div class="m">{n}</div>'
                   f'<div class="c" style="color:{D["col"][pr]}">{pr}</div><div class="p">confianza {P[n].max():.0%}</div></div>', unsafe_allow_html=True)
    votos = pd.Series(list(preds.values())).value_counts()
    ganadora = votos.index[0]
    if len(votos) == 1:
        msg, fondo = f"Consenso: los 4 modelos clasifican el comentario como <b>{ganadora}</b>.", D["col"][ganadora]
    else:
        msg = (f"Hay desacuerdo entre modelos ({', '.join(f'{k}: {v}' for k, v in votos.items())}). "
               "Conviene revisar este comentario a mano.")
        fondo = C["olivo"]
    nb, sv = D["best_nb"], D["best_svm"]
    mas_conf = "Naive Bayes" if P[nb].max() >= P[sv].max() else "SVM"
    st.markdown(f'<div class="veredicto" style="background:{fondo}">{msg}<br><span style="font-size:.95rem">Entre las mejores versiones de cada '
                f'familia, el más seguro de su respuesta es <b>{mas_conf}</b> (NB {P[nb].max():.0%} · SVM {P[sv].max():.0%}).</span></div>', unsafe_allow_html=True)
    a, b = st.columns([1.2, 1])
    with a:
        d = pd.DataFrame([(n, c, P[n][i]) for n in nombres for i, c in enumerate(cl)], columns=["modelo", "categoría", "probabilidad"])
        fig = px.bar(d, x="modelo", y="probabilidad", color="categoría", barmode="group", color_discrete_map=D["col"], title="Probabilidad por categoría")
        fig.update_yaxes(range=[0, 1.05])
        plot(estilo(fig, 360))
    with b:
        pipe = D["modelos"][nb]
        vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
        x = vec.transform([proc]).toarray().ravel()
        pres = np.nonzero(x)[0]
        if len(pres):
            pc = int(np.argmax(P[nb]))
            orden = np.argsort(P[nb])[::-1]
            seg = orden[1] if len(orden) > 1 else orden[0]
            contrib = x[pres] * (clf.feature_log_prob_[pc][pres] - clf.feature_log_prob_[seg][pres])
            t = pd.DataFrame({"término": vec.get_feature_names_out()[pres], "aporte": contrib}).sort_values("aporte")
            fig = go.Figure(go.Bar(x=t["aporte"], y=t["término"], orientation="h",
                                   marker_color=[C["verde"] if v > 0 else C["olivo"] for v in t["aporte"]]))
            fig.update_layout(title=f"Palabras que empujan a «{cl[pc]}» frente a «{cl[seg]}» (NB)")
            plot(estilo(fig, 360, leyenda=False))
        else:
            st.info("Ninguna palabra del comentario está en el vocabulario de entrenamiento.")
    with st.expander("Ver el procesamiento paso a paso"):
        st.markdown("".join(f'<div class="paso"><div class="n">{i}</div><div><div class="t">{n}</div><div class="r">{r if r else "—"}</div></div></div>'
                            for i, (n, r) in enumerate(pasos)), unsafe_allow_html=True)


def hipotesis(orig, proc, vocab):
    h, toks = [], proc.split()
    if re.search(r"\b(no|nunca|jam[aá]s|tampoco|sin|ni)\b", orig, re.I): h.append("negación")
    if re.search(r"\b(pero|aunque|sino)\b", orig, re.I): h.append("contraste")
    if toks and sum(t not in vocab for t in toks) / len(toks) > .30: h.append("vocabulario nuevo")
    if len(toks) <= 3: h.append("texto muy corto")
    if "?" in orig or "¿" in orig: h.append("formulado como pregunta")
    return ", ".join(h) if h else "sin patrón evidente (¿ironía o etiqueta ambigua?)"


def s_errores(D):
    titulo_slide("Análisis crítico: errores y generalización", "Qué se equivoca, por qué, y qué consecuencias tiene cada error.")
    te, y = D["test"], D["col_y"]
    vocab = set(D["vocab"])
    nombres = list(D["modelos"])
    e = te[[D["col_t"], y, "texto_proc"]].copy()
    for n in nombres:
        e[n] = D["pred"][n]
    e["modelos que fallan"] = sum((e[n] != e[y]).astype(int) for n in nombres)
    err = e[e["modelos que fallan"] > 0].sort_values("modelos que fallan", ascending=False).copy()
    err["hipótesis de causa"] = [hipotesis(a, b, vocab) for a, b in zip(err[D["col_t"]], err["texto_proc"])]
    t1, t2, t3 = st.tabs(["Errores en el test", "Cinco comentarios nuevos", "Consecuencias y mejoras"])
    with t1:
        if err.empty:
            st.success("Ningún modelo falló en el conjunto de prueba. La validación cruzada (diapositiva de comparación) ofrece una evaluación más exigente.")
        else:
            st.markdown(f"**{len(err)} de {len(te)}** comentarios de prueba fueron mal clasificados por al menos un modelo.")
            tabla(err.drop(columns=["texto_proc"]).reset_index(drop=True))
            ml = err.melt(id_vars=[D["col_t"], y], value_vars=nombres, var_name="modelo", value_name="pred")
            ml = ml[ml["pred"] != ml[y]]
            fig = px.histogram(ml, x="modelo", color=y, color_discrete_map=D["col"], title="Errores por modelo y categoría real")
            plot(estilo(fig, 300))
    with t2:
        if D["nuevos"] is not None:
            tabla(D["nuevos"].round(3))
        else:
            st.info("No se encontró cinco_nuevos.csv; usa la diapositiva «Clasificador en vivo» para probar comentarios nuevos.")
        st.markdown(card("Generalización", "<p>Acertar en el test no garantiza el mismo rendimiento con <b>ironía</b>, "
                         "<b>faltas de ortografía</b> o <b>vocabulario nuevo</b>: los modelos de bolsa de palabras solo conocen lo que vieron al entrenar.</p>", "v"),
                    unsafe_allow_html=True)
    with t3:
        a, b = st.columns(2)
        a.markdown(card("Costo de cada error",
                        "<ul><li><b>Queja clasificada como positiva o consulta</b>: es el error más grave, porque una queja prioritaria no se atiende.</li>"
                        "<li><b>Consulta clasificada como negativa</b>: genera prioridad innecesaria.</li>"
                        "<li>Por eso se vigila el recall de la categoría negativa.</li></ul>", "b"), unsafe_allow_html=True)
        b.markdown(card("Mejoras propuestas",
                        "<ul><li>Más datos y un diccionario ampliado con el vocabulario real de la universidad.</li>"
                        "<li>N-gramas (1,2) para capturar negaciones y expresiones.</li>"
                        "<li>Léxico de polaridad en español como característica adicional.</li>"
                        "<li>Umbral de confianza: derivar a revisión humana cuando los modelos discrepan.</li></ul>"), unsafe_allow_html=True)


def s_conclusion(D):
    titulo_slide("Conclusiones", "Qué representación y qué algoritmo conviene para este problema.")
    cvt = D["cv_tab"].sort_values("F1 macro (media)", ascending=False)
    g = D["ganador"]
    r1 = cvt.iloc[0]
    dif = r1["F1 macro (media)"] - (cvt.iloc[1]["F1 macro (media)"] if len(cvt) > 1 else 0)
    sd = max(cvt.iloc[0]["F1 macro (±)"], cvt.iloc[1]["F1 macro (±)"]) if len(cvt) > 1 else 0
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown(f'<div class="veredicto" style="background:{C["bosque"]}"><span style="font-size:.95rem">Modelo recomendado</span><br>'
                    f'<span style="font-family:Fraunces,Georgia,serif;font-size:2rem;font-weight:700">{g}</span><br>'
                    f'F1 macro en validación cruzada: {r1["F1 macro (media)"]:.3f} (± {r1["F1 macro (±)"]:.3f})</div>', unsafe_allow_html=True)
        st.markdown(card("Cómo se decidió",
                         "<p>Por <b>F1 macro</b> en validación cruzada repetida, no por la accuracy de una sola partición, "
                         "revisando además el recall de cada categoría y los errores. " +
                         ("La diferencia con el segundo es menor que la variabilidad entre pliegues: ambos son comparables y conviene preferir el más simple e interpretable."
                          if dif < sd else "La ventaja sobre el segundo supera la variabilidad entre pliegues.") + "</p>", "v"), unsafe_allow_html=True)
    with b:
        fig = px.bar(cvt.reset_index().rename(columns={"index": "modelo"}), x="F1 macro (media)", y="modelo", orientation="h",
                     error_x="F1 macro (±)", text=cvt["F1 macro (media)"].round(3).values, title="Ranking por F1 macro (validación cruzada)",
                     color_discrete_sequence=[C["verde"]])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        plot(estilo(fig, 300, leyenda=False))
    st.markdown("##### Preguntas de pensamiento crítico")
    qa = [("1. BoW vs TF-IDF: ¿una palabra omnipresente carece de importancia?",
           "No. Su IDF es bajo porque no distingue un comentario de otro, no porque carezca de significado."),
          ("2. NB 85 % vs SVM 82 %: ¿se elige automáticamente NB?",
           "No. Se revisan F1 macro, recall por clase, matrices de confusión, variabilidad en validación cruzada, tamaño de la muestra y costo de cada error."),
          ("3. Clases mezcladas en el gráfico de 2 componentes: ¿ningún clasificador podrá separarlas?",
           "No. Dos componentes retienen solo una parte de la varianza y PCA no busca separar categorías."),
          ("4. ¿Qué consecuencias tienen los errores?",
           "Una queja clasificada como positiva o consulta no se atiende; una consulta clasificada como negativa genera prioridad innecesaria."),
          ("5. Generalización con ironía, faltas de ortografía o vocabulario nuevo",
           "No se puede afirmar: el test solo acredita desempeño sobre comentarios parecidos a los de entrenamiento."),
          ("6. Si 90 de 120 comentarios fueran de una sola categoría, ¿basta la accuracy?",
           "No. Predecir siempre la clase mayoritaria daría 75 % sin aprender nada; se usan F1 macro, recall por clase, matrices y class_weight."),
          ("7. ¿Cómo comprobar que la mejora se debe a TF-IDF y no a la partición?",
           "Comparando con los mismos pliegues de validación cruzada repetida y ajustando el vectorizador solo con el entrenamiento de cada pliegue.")]
    for q, r in qa:
        with st.expander(q):
            st.write(r)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hero" style="padding:1.6rem 2rem"><p style="font-size:1.35rem;font-family:Fraunces,Georgia,serif">'
                'Gracias. ¿Preguntas?</p></div>', unsafe_allow_html=True)


SLIDES = [("Portada", s_portada), ("Problema y objetivo", s_problema), ("Los datos", s_datos), ("Limpieza del texto", s_limpieza),
          ("Bag of Words", s_bow), ("TF-IDF", s_tfidf), ("PCA", s_pca), ("Naive Bayes", s_nb), ("SVM", s_svm),
          ("Comparación de modelos", s_comparativa), ("Comentario por comentario", s_por_comentario),
          ("Clasificador en vivo", s_vivo), ("Errores y análisis crítico", s_errores), ("Conclusiones", s_conclusion)]

# ======================================================================================
# NAVEGACIÓN
# ======================================================================================
st.session_state.setdefault("slide", 0)
N = len(SLIDES)


def ir(i):
    st.session_state["slide"] = int(min(max(i, 0), N - 1))


D = preparar()
if "error" in D:
    st.error(D["error"])
    st.stop()

i = st.session_state["slide"]
nombre, fn = SLIDES[i]
st.markdown(f'<div class="topbar"><b>{i + 1} / {N}</b><span>{nombre}</span></div>'
            f'<div class="prog"><div style="width:{(i + 1) / N * 100:.1f}%"></div></div>', unsafe_allow_html=True)

fn(D)

# barra inferior
nb_ = st.container(key="navbar")
with nb_:
    c1, c2, c3, c4 = st.columns([1, 1, 4, 1.2])
    c1.button("Anterior", on_click=ir, args=(i - 1,), disabled=i == 0, key="prev")
    c2.button("Siguiente", on_click=ir, args=(i + 1,), disabled=i == N - 1, type="primary", key="next")
    dots = "".join(f'<i class="{"a" if j == i else ("h" if j < i else "")}"></i>' for j in range(N))
    c3.markdown(f'<div class="dots">{dots}</div>', unsafe_allow_html=True)
    with c4:
        pop = st.popover("Índice") if hasattr(st, "popover") else st.expander("Índice")
        with pop:
            for j, (n, _) in enumerate(SLIDES):
                st.button(f"{j + 1}. {n}", key=f"idx{j}", on_click=ir, args=(j,), disabled=j == i)

# atajos de teclado ← →
SCRIPT_TECLAS = """<script>
const d = window.parent.document;
if (!d.__navPLN) { d.__navPLN = true;
  d.addEventListener('keydown', e => {
    const t = d.activeElement ? d.activeElement.tagName : '';
    if (['INPUT','TEXTAREA','SELECT'].includes(t)) return;
    const b = [...d.querySelectorAll('button')];
    const f = txt => b.find(x => x.innerText.trim() === txt && !x.disabled);
    if (e.key === 'ArrowRight') { const x = f('Siguiente'); if (x) x.click(); }
    if (e.key === 'ArrowLeft')  { const x = f('Anterior');  if (x) x.click(); }
  });
}</script>"""
try:
    if hasattr(st, "iframe"):
        st.iframe(SCRIPT_TECLAS, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(SCRIPT_TECLAS, height=0)
except Exception:
    pass
