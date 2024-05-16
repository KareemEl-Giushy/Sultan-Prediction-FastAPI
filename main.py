from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pickle

# Create an instance of the FastAPI class
app = FastAPI()

# Define a route using a decorator and async function
@app.get("/")  # Decorator for GET requests to the root URL
async def read_root():
    return {"message": "Hello, World!"}

class Item(BaseModel):
    hr: float
    respr: float
    time: float

# Define another route
@app.post("/predict/", response_model=Item)  
async def predict(item: Item) -> JSONResponse:
    response = prediction(item.hr, item.respr, item.time)
    return JSONResponse({
        "prediction": response,
    })

def prediction(hr, respr, time):
    f = open("standard_scaler", "rb")
    scaler = pickle.load(f)

    x = scaler.transform([[hr, respr, time]])
    
    f = open("model_pickle", "rb")
    model = pickle.load(f)
    d = model.predict(x)
    
    return str(d[0])



