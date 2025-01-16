import fastapi
import pandas as pd
from fastapi import HTTPException
from typing import Dict, List

from challenge.model import DelayModel

app = fastapi.FastAPI()

VALID_MES_VALUES = set(range(1, 13))
VALID_TIPOVUELO_VALUES = {"I", "N"}

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(json: Dict) -> dict:
    if "flights" not in json:
        raise HTTPException(status_code=400, detail="Missing 'flights' key in the request body.")

    data = pd.DataFrame(json["flights"])

    if data.empty:
        raise HTTPException(status_code=400, detail="No flight data provided.")

    required_columns = ["OPERA", "TIPOVUELO", "MES"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing_columns)}."
        )

    for index, row in data.iterrows():
        if not isinstance(row["MES"], int):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid MES value (not an integer) at index {index}."
            )
        if row["MES"] not in VALID_MES_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid MES value {row['MES']} at index {index}. Must be between 1 and 12."
            )
        if not isinstance(row["TIPOVUELO"], str):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid TIPOVUELO value (not a string) at index {index}."
            )
        if row["TIPOVUELO"] not in VALID_TIPOVUELO_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid TIPOVUELO value '{row['TIPOVUELO']}' at index {index}. Must be 'I' or 'N'."
            )
        if not isinstance(row["OPERA"], str) or not row["OPERA"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid OPERA value (must be a non-empty string) at index {index}."
            )

    delay_model = DelayModel()

    try:
        features = delay_model.preprocess(data=data)
        prediction = delay_model.predict(features=features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    return {"predict": prediction}
