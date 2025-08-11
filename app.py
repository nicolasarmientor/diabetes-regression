from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from train import train_and_score
from report import generate_pdf

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
RESULT_DIR = STATIC_DIR / "results"
CSS_DIR = STATIC_DIR / "css" / "style.css"
PDF_DIR = RESULT_DIR / "diabetes_regression_results.pdf"
JSON_DIR = RESULT_DIR / "linear_model_results.json"

results = train_and_score()
results.to_json(JSON_DIR, orient="records", indent=4)
generate_pdf(results, PDF_DIR)

app = FastAPI(title="Diabetes Regression Results")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    results = train_and_score()
    results.to_json(JSON_DIR, orient="records", indent=4)
    generate_pdf(results, PDF_DIR)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/results", response_class=JSONResponse)
async def get_results():
    with JSON_DIR.open() as json_file:
        return json.load(json_file)

@app.get("/api/report", response_class=FileResponse)
async def get_report():
    return FileResponse(path=str(PDF_DIR), media_type="application/pdf", filename=PDF_DIR.name)