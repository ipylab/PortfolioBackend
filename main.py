from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from optimizador import optimizar_portafolio
from typing import Optional

app = FastAPI()

# CORS para permitir peticiones desde cualquier origen (ideal para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes cambiar esto por tu dominio específico en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


