# app/main.py
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from schemas import BulkForecastRequest, BulkForecastResponse
from inference import run_forecast, available_game_ids, reload_data

app = FastAPI()

# # Load models at startup
# @app.on_event("startup")
# def startup_event():
#     load_models()

@app.get("/")
def read_root():
    return {"message": "Welcome to the forecasting API. Go to /docs for API documentation."}


@app.post("/predict", response_model=BulkForecastResponse)
def predict_bulk(request_body: BulkForecastRequest):
    # Rate limit: if requests length > 50, raise error
    if len(request_body.requests) > 50:
        raise HTTPException(status_code=400, detail="Too many requests, max allowed is 50")

    start_time = time.time()
    results = []

    for single_req in request_body.requests:
        # Validate game ID
        if single_req.game_id not in available_game_ids:
            raise HTTPException(status_code=400, detail=f"Game ID {single_req.game_id} not supported")
        
        # Basic validation
        if single_req.use_market_channels and single_req.channel_spends is None and single_req.total_cac is None:
            raise HTTPException(status_code=400, detail="Either channel_spends or total_cac is required when using market segmentation")
        
        # If not using market channels, total_cac is required
        if not single_req.use_market_channels and single_req.total_cac is None:
            raise HTTPException(status_code=400, detail="total_cac is required when not using market segmentation")

        try:
            res = run_forecast(single_req)
            results.append(res)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    end_time = time.time()
    runtime_seconds = end_time - start_time

    return BulkForecastResponse(
        status_code=200,
        runtime_seconds=runtime_seconds,
        results=results
    )

@app.post("/refresh_models")
def refresh_models():
    """Reload the prediction store (final_M2.csv) and helper files from disk."""
    try:
        stats = reload_data()
        return JSONResponse(content={"message": "Models refreshed successfully", **stats}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8080)
