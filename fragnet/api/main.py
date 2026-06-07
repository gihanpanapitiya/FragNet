import os
from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from the project root (two levels above this file)
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fragnet.api.routers import analyze, optimize, llm

app = FastAPI(title="FragNet API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api")
app.include_router(optimize.router, prefix="/api")
app.include_router(llm.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the React production build — must be registered last
_dist = os.path.join(os.path.dirname(__file__), "../../frontend/dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
