"""
Preprocesamiento del dashboard.

Replica EXACTAMENTE la función `limpiar_texto` del notebook (Fase 1), pero devuelve
también cada paso intermedio para poder mostrarlo en la defensa.

Si en el notebook modificaste `limpiar_texto` (por ejemplo para conservar los signos
de interrogación de la clase «consulta»), aplica el mismo cambio aquí en `pasos()`;
de lo contrario el texto escrito en vivo se procesará distinto al de entrenamiento.
"""
import re
import unicodedata
from pathlib import Path

from nltk.stem import SnowballStemmer

# ------------------------------- recursos por defecto -------------------------------
DICCIONARIO_PROPIO = {
    "q": "que", "xq": "porque", "pq": "porque", "porq": "porque", "x": "por", "xa": "para",
    "pa": "para", "tb": "tambien", "tmb": "tambien", "tbn": "tambien", "d": "de", "k": "que",
    "bn": "bien", "ntp": "no_hay_problema", "grax": "gracias", "dnd": "donde", "cmo": "como",
    "kien": "quien", "sii": "si", "sip": "si", "nop": "no", "toy": "estoy", "ta": "esta", "pk": "porque",
    "profe": "profesor", "profes": "profesores", "prof": "profesor", "docentes": "docente",
    "uni": "universidad", "univ": "universidad", "umsa": "universidad",
    "mate": "matematica", "mates": "matematica", "info": "informacion", "inscrip": "inscripcion",
    "inscripciones": "inscripcion", "bibliot": "biblioteca", "lab": "laboratorio",
    "labs": "laboratorio", "wifi": "internet", "wi-fi": "internet",
    "exelente": "excelente", "ecxelente": "excelente", "exelentes": "excelente",
    "pecimo": "pesimo", "orible": "horrible", "orrible": "horrible",
    "aburrio": "aburrido", "aburida": "aburrida", "vien": "bien", "muybien": "muy bien",
    "agradeser": "agradecer", "ermoso": "hermoso", "aser": "hacer", "haser": "hacer",
    "atension": "atencion", "atencon": "atencion", "deficiensia": "deficiencia",
    "tardansa": "tardanza", "nesesito": "necesito", "nesecito": "necesito", "buenizimo": "buenisimo",
}
NEGADORES = {"no", "nunca", "jamas", "tampoco", "sin", "ni"}
CONSERVAR = NEGADORES | {"pero", "aunque", "sino", "muy", "mas", "menos", "nada", "poco", "mucho",
                         "bien", "mal", "mejor", "peor", "todo", "algo"}

EMOJIS = {
    "positivo": ["😀", "😃", "😄", "😁", "😊", "🙂", "😍", "👍", "👏", "❤", "💯", "🎉", "😎", "🥰"],
    "negativo": ["😡", "😠", "😞", "😢", "😭", "👎", "💔", "😤", "🤬", "😒", "😩", "😫"],
}
EMOTICONES = {r":\)": "emopos", r":-\)": "emopos", r":d\b": "emopos", r":\(": "emoneg", r":-\(": "emoneg"}


def sin_tildes(s: str) -> str:
    s = s.replace("ñ", "\x00").replace("Ñ", "\x01")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.replace("\x00", "ñ").replace("\x01", "Ñ")


class Preprocesador:
    def __init__(self, recursos: dict | None = None):
        recursos = recursos or {}
        self.diccionario = dict(recursos.get("diccionario", DICCIONARIO_PROPIO))
        self.negadores = set(recursos.get("negadores", NEGADORES))
        sw = recursos.get("stopwords")
        if sw is None:
            try:
                from nltk.corpus import stopwords
                base = set(stopwords.words("spanish"))
            except LookupError:
                import nltk
                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords
                base = set(stopwords.words("spanish"))
            sw = {sin_tildes(w) for w in base} - {sin_tildes(w) for w in CONSERVAR}
        self.stopwords = set(sw)
        self.stemmer = SnowballStemmer("spanish")

    # -------------------------------------------------------------------------------
    def _emojis(self, t: str) -> str:
        for e in EMOJIS["positivo"]:
            t = t.replace(e, " emopos ")
        for e in EMOJIS["negativo"]:
            t = t.replace(e, " emoneg ")
        for patron, tok in EMOTICONES.items():
            t = re.sub(patron, f" {tok} ", t)
        return t

    def pasos(self, texto: str):
        """Devuelve [(nombre_del_paso, resultado)] y el texto final (stem + negación)."""
        out = [("Texto original", str(texto))]
        t = self._emojis(str(texto).lower())
        out.append(("Minúsculas y emojis convertidos en tokens", t.strip()))
        t = re.sub(r"(https?://\S+|www\.\S+|\S+@\S+|[@#]\w+)", " ", t)
        t = re.sub(r"\d+", " ", t)
        out.append(("Sin URLs, menciones ni números", re.sub(r"\s+", " ", t).strip()))
        t = sin_tildes(t)
        t = re.sub(r"(.)\1{2,}", r"\1", t)
        t = re.sub(r"[^a-zñ_\s]", " ", t)
        tokens = t.split()
        out.append(("Sin tildes, alargamientos ni puntuación", " ".join(tokens)))
        nuevos = []
        for tok in tokens:
            nuevos.extend(self.diccionario.get(tok, tok).split())
        out.append(("Diccionario propio (jerga y ortografía)", " ".join(nuevos)))
        tokens = [w for w in nuevos if w not in self.stopwords and len(w) > 1]
        out.append(("Sin stop words (se conservan las negaciones)", " ".join(tokens)))
        tokens = [w if w in self.negadores or w in ("emopos", "emoneg") else self.stemmer.stem(w) for w in tokens]
        out.append(("Stemming (raíz de cada palabra)", " ".join(tokens)))
        res, i = [], 0
        while i < len(tokens):
            if tokens[i] in self.negadores and i + 1 < len(tokens) and tokens[i + 1] not in self.negadores:
                res.append(f"{tokens[i]}_{tokens[i + 1]}")
                i += 2
            else:
                res.append(tokens[i])
                i += 1
        final = " ".join(res)
        out.append(("Marcado de negación → texto que ven los modelos", final))
        return out, final

    def limpiar(self, texto: str) -> str:
        return self.pasos(texto)[1]

    def limpiar_basico(self, texto: str) -> str:
        """Equivale a la columna `texto_limpio` (sin stemming ni negación)."""
        pasos, _ = self.pasos(texto)
        return pasos[5][1]


def cargar_preprocesador(dir_artefactos: Path) -> Preprocesador:
    import joblib
    ruta = Path(dir_artefactos) / "recursos_preprocesamiento.joblib"
    recursos = joblib.load(ruta) if ruta.exists() else None
    return Preprocesador(recursos)
