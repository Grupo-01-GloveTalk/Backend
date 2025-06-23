from fastapi import FastAPI
from app.api.routes import router

# Inicialización de la aplicación FastAPI para GloveTalk
app = FastAPI(title="GloveTalk API")

# Inclusión de las rutas principales bajo el prefijo /api y el tag Prediction
app.include_router(router, prefix="/api", tags=["Prediction"])
