import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core.getlimits")
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import interview, session

app = FastAPI(title="AI Interview Analyzer API")

# Setup CORS to allow Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(interview.router, prefix="/api")
app.include_router(session.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "AI Interview Analyzer API Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

