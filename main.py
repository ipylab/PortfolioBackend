from fastapi import FastAPI
from pydantic import BaseModel
from optimizador import optimizar_portafolio
from typing import Optional

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Portfolio backend online ✅"}

@app.post("/optimizar")
def optimizar(data: dict):
    return optimizar_portafolio(
        activos=data["activos"],
        capital=data["capital"],
        horizonte=data["horizonte"],
        metodo=data["metodo"],
        peso_max=data.get("peso_max", 1.0),
        pesos_manual=data.get("pesos_manual")
    )
    return resultado
