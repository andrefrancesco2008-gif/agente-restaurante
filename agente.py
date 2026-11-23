import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# 1. Cargar variables de entorno desde .env
load_dotenv(Path(__file__).parent / ".env")

PROVEEDOR = os.getenv("PROVEEDOR", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPEN_API_KEY = os.getenv("OPEN_API_KEY", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

# Google ADK busca GOOGLE_API_KEY por defecto cuando se usa Gemini
if GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# 2. Seleccionar el modelo segun PROVEEDOR en .env
if PROVEEDOR == "openai":
    modelo = LiteLlm(
        model="openai/gpt-4o-mini",
        api_key=OPEN_API_KEY,
    )
elif PROVEEDOR == "ollama":
    modelo = LiteLlm(
        model="openai/gemma3:12b",
        api_base="https://ollama.com/v1",
        api_key=OLLAMA_API_KEY,
    )
else:
    modelo = "gemini-3.8-flash"

# 3. Ruta a la carpeta de datasets CSV
carpeta = Path(__file__).parent / "datos"


# =========================================================
# GRUPO A. CONSULTAS (5 herramientas)
# =========================================================

def resenas_pendientes() -> dict:
    """Lista todas las resenas en estado Pendiente, ordenadas por urgencia (Urgente, Media, Baja) y menor calificacion."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    pendientes = resenas[resenas["estado"] == "Pendiente"].copy()

    orden = {"Urgente": 1, "Media": 2, "Baja": 3}
    pendientes["prioridad_num"] = pendientes["urgencia"].map(orden).fillna(4)
    pendientes = pendientes.sort_values(by=["prioridad_num", "calificacion"])
    pendientes = pendientes.drop(columns=["prioridad_num"])

    return {
        "cantidad": int(len(pendientes)),
        "urgentes": int((pendientes["urgencia"] == "Urgente").sum()),
        "resenas": pendientes.to_dict(orient="records"),
    }


def buscar_cliente(nombre: str) -> dict:
    """Busca clientes por su nombre o parte del nombre en clientes.csv y devuelve sus datos e id_cliente."""
    clientes = pd.read_csv(carpeta / "clientes.csv")
    encontrados = clientes[clientes["nombre"].str.contains(nombre.strip(), case=False, na=False)]
    return {
        "cantidad": int(len(encontrados)),
        "clientes": encontrados.to_dict(orient="records"),
    }


def resenas_de_cliente(id_cliente: str) -> dict:
    """Lista todas las resenas que ha dejado un cliente usando su id_cliente (por ejemplo: C002)."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    codigo = id_cliente.strip().upper()
    del_cliente = resenas[resenas["id_cliente"].str.upper() == codigo]
    return {
        "id_cliente": codigo,
        "cantidad": int(len(del_cliente)),
        "resenas": del_cliente.to_dict(orient="records"),
    }


def detalle_resena(id_resena: str) -> dict:
    """Muestra el detalle completo de una resena por su codigo (por ejemplo: R002), incluyendo su respuesta oficial si ya fue atendida."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    respuestas = pd.read_csv(carpeta / "respuestas.csv")
    codigo = id_resena.strip().upper()

    fila = resenas[resenas["id_resena"].str.upper() == codigo]
    if fila.empty:
        return {"error": f"No se encontro la resena con codigo {codigo}."}

    resp_asociadas = respuestas[respuestas["id_resena"].str.upper() == codigo]
    return {
        "resena": fila.iloc[0].to_dict(),
        "respuestas": resp_asociadas.to_dict(orient="records"),
    }


def resumen_reputacion() -> dict:
    """Calcula el resumen general de reputacion: total de resenas, promedio de estrellas, respondidas, pendientes y urgentes pendientes."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    total = int(len(resenas))
    promedio = round(float(resenas["calificacion"].mean()), 2) if total > 0 else 0.0
    respondidas = int((resenas["estado"] == "Respondida").sum())
    pendientes = int((resenas["estado"] == "Pendiente").sum())
    urgentes_pendientes = int(
        ((resenas["estado"] == "Pendiente") & (resenas["urgencia"] == "Urgente")).sum()
    )

    return {
        "total_resenas": total,
        "promedio_calificacion": promedio,
        "respondidas": respondidas,
        "pendientes": pendientes,
        "urgentes_pendientes": urgentes_pendientes,
    }


# =========================================================
# GRUPO B. GRAFICOS (3 herramientas)
# =========================================================

def resenas_por_plataforma() -> dict:
    """Cuenta la cantidad de resenas recibidas por cada plataforma (Google Maps, TripAdvisor, PedidosYa, Rappi, Instagram)."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    conteo = resenas.groupby("plataforma")["id_resena"].count()
    return {
        "etiquetas": conteo.index.tolist(),
        "valores": [int(v) for v in conteo.tolist()],
    }


def resenas_por_categoria() -> dict:
    """Cuenta la cantidad de resenas recibidas por cada categoria operativa (Comida, Servicio, Tiempos, Delivery, Ambiente)."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    conteo = resenas.groupby("categoria")["id_resena"].count()
    return {
        "etiquetas": conteo.index.tolist(),
        "valores": [int(v) for v in conteo.tolist()],
    }


def crear_grafico(titulo: str, tipo: str, etiquetas: list[str], valores: list[float]) -> dict:
    """Envia los datos estructurados para dibujar un grafico en la pagina web. tipo debe ser: bar, line o pie."""
    return {
        "titulo": titulo,
        "tipo": tipo,
        "etiquetas": etiquetas,
        "valores": [float(v) for v in valores],
    }


# =========================================================
# GRUPO C. REGISTROS (2 herramientas)
# =========================================================

def registrar_resena(
    id_cliente: str,
    plataforma: str,
    calificacion: int,
    categoria: str,
    texto: str,
) -> dict:
    """Registra una nueva resena en resenas.csv. Recibe id_cliente, plataforma, calificacion (1 a 5), categoria y texto del comentario."""
    if calificacion < 1 or calificacion > 5:
        return {"error": "La calificacion debe ser un numero entero entre 1 y 5."}

    clientes = pd.read_csv(carpeta / "clientes.csv")
    resenas = pd.read_csv(carpeta / "resenas.csv")

    cod_cliente = id_cliente.strip().upper()
    cli = clientes[clientes["id_cliente"].str.upper() == cod_cliente]
    if cli.empty:
        return {"error": f"No existe un cliente con codigo {cod_cliente}."}

    nombre_cliente = str(cli.iloc[0]["nombre"])
    nuevo_id = f"R{len(resenas) + 1:03d}"

    if calificacion <= 2:
        urgencia = "Urgente"
    elif calificacion == 3:
        urgencia = "Media"
    else:
        urgencia = "Baja"

    nueva_fila = {
        "id_resena": nuevo_id,
        "id_cliente": cod_cliente,
        "cliente": nombre_cliente,
        "plataforma": plataforma.strip(),
        "calificacion": int(calificacion),
        "categoria": categoria.strip(),
        "urgencia": urgencia,
        "estado": "Pendiente",
        "fecha": str(datetime.now())[:16],
        "texto": texto.strip(),
    }

    resenas = pd.concat([resenas, pd.DataFrame([nueva_fila])], ignore_index=True)
    resenas.to_csv(carpeta / "resenas.csv", index=False)

    return {
        "mensaje": "Resena registrada correctamente.",
        "id_resena": nuevo_id,
        "cliente": nombre_cliente,
        "urgencia": urgencia,
    }


def registrar_respuesta(
    id_resena: str,
    texto_respuesta: str,
    cortesia_ofrecida: str = "Ninguna",
    responsable: str = "Administrador",
) -> dict:
    """Registra una respuesta oficial en respuestas.csv y actualiza el estado de la resena a Respondida en resenas.csv."""
    resenas = pd.read_csv(carpeta / "resenas.csv")
    respuestas = pd.read_csv(carpeta / "respuestas.csv")

    cod_resena = id_resena.strip().upper()
    mascara = resenas["id_resena"].str.upper() == cod_resena

    if not mascara.any():
        return {"error": f"No existe una resena con codigo {cod_resena}."}

    nuevo_id_resp = f"RP{len(respuestas) + 1:03d}"
    nueva_respuesta = {
        "id_respuesta": nuevo_id_resp,
        "id_resena": cod_resena,
        "responsable": responsable.strip(),
        "fecha_respuesta": str(datetime.now())[:16],
        "cortesia_ofrecida": cortesia_ofrecida.strip(),
        "texto_respuesta": texto_respuesta.strip(),
    }

    respuestas = pd.concat([respuestas, pd.DataFrame([nueva_respuesta])], ignore_index=True)
    respuestas.to_csv(carpeta / "respuestas.csv", index=False)

    resenas.loc[mascara, "estado"] = "Respondida"
    resenas.to_csv(carpeta / "resenas.csv", index=False)

    return {
        "mensaje": "Respuesta registrada y resena marcada como Respondida.",
        "id_respuesta": nuevo_id_resp,
        "id_resena": cod_resena,
        "cortesia_ofrecida": cortesia_ofrecida.strip(),
    }


# =========================================================
# INSTRUCCIONES Y AGENTE CENTRAL (ADK)
# =========================================================

INSTRUCCION = """
ROL
Eres el asistente inteligente de gestion de reputacion y resenas de un restaurante peruano.
Respondes siempre en espanol, de forma clara, profesional y concisa para que tus respuestas tambien se escuchen bien por voz.

REGLAS GENERALES
1. Usa siempre las herramientas disponibles para consultar o guardar datos reales.
2. Nunca inventes clientes, codigos de resenas ni estadisticas.
3. Cuando te pidan sugerencias de respuesta para quejas urgentes, redacta disculpas empaticas y personalizadas segun el problema reportado.

CLIENTES
Si el usuario menciona a un cliente por nombre, usa primero buscar_cliente para obtener su id_cliente antes de consultar sus resenas o registrar una nueva.

RESENAS Y REPUTACION
Usa resenas_pendientes para listar y priorizar quejas sin atender.
Usa detalle_resena para consultar una resena especifica por su codigo, por ejemplo R002.
Usa resumen_reputacion para reportar el promedio de estrellas y el conteo general.

GRAFICOS
Cuando el usuario pida un grafico:
1. Primero llama a resenas_por_plataforma o resenas_por_categoria para obtener las etiquetas y valores reales.
2. Luego llama obligatoriamente a crear_grafico pasando titulo, tipo (bar, line o pie), etiquetas y valores.
3. Acompana el grafico con un breve resumen en texto de los datos.

CONFIRMACION ANTES DE REGISTRAR
Antes de llamar a registrar_resena o registrar_respuesta, repite de forma resumida los datos clave que vas a guardar y pregunta siempre: ¿Confirmas?
Solo ejecuta la herramienta de registro cuando el usuario responda afirmativamente.
"""

root_agent = Agent(
    model=modelo,
    name="agente_restaurante",
    instruction=INSTRUCCION,
    tools=[
        resenas_pendientes,
        buscar_cliente,
        resenas_de_cliente,
        detalle_resena,
        resumen_reputacion,
        resenas_por_plataforma,
        resenas_por_categoria,
        crear_grafico,
        registrar_resena,
        registrar_respuesta,
    ],
)