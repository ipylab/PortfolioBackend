from fastapi import FastAPI
from pydantic import BaseModel
from optimizador import optimizar_portafolio

app = FastAPI()

class OptimInput(BaseModel):
    activos: list[str]
    capital: float
    horizonte: int
    metodo: str
    peso_max: float

@app.post("/optimizar")
def optimizar(data: OptimInput):
    resultado = optimizar_portafolio(
        activos=data.activos,
        capital=data.capital,
        horizonte=data.horizonte,
        metodo=data.metodo,
        peso_max=data.peso_max
    )
    return resultado