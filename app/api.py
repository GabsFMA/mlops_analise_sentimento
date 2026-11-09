from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from loguru import logger
import json

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="API de Sentimento", version='0.1.0')

class Entrada(BaseModel):
    texto: str = Field(min_length=3, max_length=1000, examples=["Adorei o produto!"])
    
class Saida(BaseModel):
    sentimento: str
    confianca: float
    versao_modelo: str
    
@app.get("/health", summary="Verifica a saúde da API")
def health():
    return {"status": "ok", "versao_modelo": "0.1.0"}

@app.post("/predict", response_model=Saida, summary="Realiza a predição do sentimento")
def predict(entrada: Entrada):
    probas = model.predict_proba([entrada.texto])[0]
    i = int(probas.argmax())
    resultado = Saida(
        sentimento=str(model.classes_[i]),
        confianca=round(float(probas[i]), 4),
        versao_modelo="0.1.0" 
    )
    
    logger.info(json.dumps({
        "evento": "predicao",
        "sentimento": resultado.sentimento,
        "confianca": resultado.confianca,
        "tamanho_texto": len(entrada.texto),
        "versao_modelo": resultado.versao_modelo,
    }))
    
    return resultado