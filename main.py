from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from optimizador import optimizar_portafolio

app = FastAPI()

# Middleware para permitir solicitudes CORS desde cualquier frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a ["https://tudominio.com"] si lo deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint raíz para comprobar si el backend está activo
@app.get("/")
def root():
    return {"message": "Portfolio backend online ✅"}

# Modelo de entrada para la optimización
class OptimInput(BaseModel):
    activos: list[str]
    capital: float
    horizonte: int
    metodo: str
    peso_max: Optional[float] = 1.0
    pesos_manual: Optional[list[float]] = None

# Endpoint POST principal
@app.post("/optimizar")
def optimizar(data: OptimInput):
    return optimizar_portafolio(
        activos=data.activos,
        capital=data.capital,
        horizonte=data.horizonte,
        metodo=data.metodo,
        peso_max=data.peso_max,
        pesos_manual=data.pesos_manual
    )


