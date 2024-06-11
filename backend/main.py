from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

from Services.genai import YoutubeProcessor, GeminiProcessor

class VideoAnalysisRequest(BaseModel):
    youtube_link : HttpUrl


app = FastAPI()

# configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ["*"]
)


@app.post("/analyze_video")
def analyze_video(request : VideoAnalysisRequest):
    processor = YoutubeProcessor()
    result = processor.retrieve_youtube_documents(str(request.youtube_link), verbose = True)
    genai_processor = GeminiProcessor(model_name = 'gemini-pro', 
                                      project = 'gemini-dynamo-423808')
    summary = genai_processor.generate_document_summary(documents = result, 
                                                        verbose = True)
    return {'summary' : summary}