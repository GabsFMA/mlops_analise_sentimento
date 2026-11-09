import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"

def main():
    textos = ["Este é um ótimo produto!",
              "Não gostei do serviço."]
    
    modelo = joblib.load(MODEL_PATH)
    
    previsoes = modelo.predict(textos)
    
    for texto, previsao in zip(textos, previsoes):
        print(f"Texto: {texto} => Previsão: {previsao}")
        print("--"*40)
        
if __name__ == "__main__":
    main()