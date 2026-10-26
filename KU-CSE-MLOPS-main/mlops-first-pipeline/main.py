from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field



MODEL_PATH = Path("models/iris_random_forest_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Random Forest API")


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_iris(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    species = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    return {
        "prediction": prediction,
        "species": species[prediction]
    }

