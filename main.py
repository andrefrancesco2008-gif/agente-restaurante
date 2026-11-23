import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from google.adk.runners import InMemoryRunner
from pydantic import BaseModel

from agente import root_agent

app = FastAPI(title="Agente de Reputacion de Restaurante")
ejecutor = InMemoryRunner(agent=root_agent)

RUTA_HTML = Path(__file__).parent / "static" / "index.html"


class Pregunta(BaseModel):
    texto: str


@app.get("/")
def inicio():
    return FileResponse(RUTA_HTML, headers={"Cache-Control": "no-store"})


@app.post("/chat")
async def chat(pregunta: Pregunta):
    eventos = None
    for intento in range(3):
        try:
            eventos = await ejecutor.run_debug(pregunta.texto, quiet=True)
            break
        except Exception as error:
            if "503" in str(error) and intento < 2:
                await asyncio.sleep(3)
                continue
            return {
                "respuesta": f"Ocurrio un error al consultar el modelo: {error}",
                "grafico": None,
            }

    texto_respuesta = ""
    datos_grafico = None

    for evento in eventos:
        for resultado in evento.get_function_responses():
            if resultado.name == "crear_grafico" and isinstance(resultado.response, dict):
                resp = resultado.response
                if "etiquetas" in resp and "valores" in resp:
                    datos_grafico = resp
                elif isinstance(resp.get("result"), dict):
                    datos_grafico = resp["result"]

        if evento.is_final_response() and evento.content and evento.content.parts:
            partes_texto = [
                parte.text for parte in evento.content.parts if getattr(parte, "text", None)
            ]
            if partes_texto:
                texto_respuesta = "\n".join(partes_texto)

    if not texto_respuesta:
        texto_respuesta = "No se recibio respuesta del agente. Intenta nuevamente."

    return {
        "respuesta": texto_respuesta,
        "grafico": datos_grafico,
    }