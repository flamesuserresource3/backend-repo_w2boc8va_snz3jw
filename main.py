import os
import math
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from database import db, create_document, get_documents
from schemas import AnalyzeRequest, AnalyzeResponse, Clip, RateRequest, Analysis

# Optional advanced AI imports
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI

app = FastAPI(title="Agung Clip Viral API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_youtube_id(url: str) -> str:
    try:
        from urllib.parse import urlparse, parse_qs
        u = urlparse(url)
        host = (u.hostname or "").lower()
        if "youtu.be" in host:
            return u.path.lstrip("/")
        if "youtube.com" in host:
            qs = parse_qs(u.query)
            return qs.get("v", [""])[0]
    except Exception:
        pass
    return ""


def fetch_transcript_text(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer English or Indonesian if available
        preferred = None
        for lang in ["en", "id", "en-US", "en-GB"]:
            if transcript_list.find_transcript([lang]):
                preferred = transcript_list.find_transcript([lang])
                break
        if preferred is None:
            preferred = transcript_list.find_transcript(transcript_list._manually_created_transcripts)
        transcript = preferred.fetch()
        text = " ".join([t.get("text", "") for t in transcript])
        return text[:35000]  # limit tokens
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return ""


def openai_segment(video_id: str, transcript_text: str) -> List[Clip]:
    """Use OpenAI to propose 10 high-signal segments (30-60s)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are an expert short-form video editor. Given a transcript from a YouTube video, "
        "identify 10 highly engaging, self-contained segments optimized for viral short clips. "
        "Each segment must be 30-60 seconds and include: start_time_sec, end_time_sec, and a punchy title (<80 chars). "
        "Favor moments with hooks, strong emotions, surprises, or actionable insights. "
        "Respond strictly as JSON with an array under 'clips'."
    )
    content = transcript_text if transcript_text else "Transcript not available. Propose likely segments based on common pacing (assume 15 min length)."
    try:
        completion = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        raw = completion.output_text
        import json
        data = json.loads(raw)
        out: List[Clip] = []
        seen = set()
        for item in data.get("clips", [])[:10]:
            try:
                s = int(max(0, round(float(item.get("start_time_sec", 0)))))
                e = int(max(s + 30, round(float(item.get("end_time_sec", s + 30)))))
                # Enforce max 60s
                if e - s > 60:
                    e = s + 60
                key = (s, e)
                if key in seen:
                    continue
                seen.add(key)
                title = str(item.get("title", "Viral Highlight"))[:80]
                out.append(Clip(video_id=video_id, start=s, end=e, title=title, rating=0))
            except ValidationError:
                continue
        return out
    except Exception:
        return []


def fallback_segments(video_id: str, total_seconds: int = 15 * 60) -> List[Clip]:
    # evenly space 10 segments of 36s each
    step = max(1, total_seconds // 10)
    clips: List[Clip] = []
    cur = 0
    for i in range(10):
        s = cur + 5
        e = s + 36
        clips.append(Clip(video_id=video_id, start=s, end=e, title=f"Viral Highlight #{i+1}", rating=0))
        cur += step
    return clips


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    video_id = extract_youtube_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript_text = fetch_transcript_text(video_id)
    clips = openai_segment(video_id, transcript_text)
    if not clips:
        # If AI not configured or failed, provide reasonable defaults
        clips = fallback_segments(video_id)

    # persist job and initial clips as separate documents
    try:
        create_document("analysis", Analysis(url=req.url, video_id=video_id, with_subtitle=req.with_subtitle, language=req.language))
        for c in clips:
            create_document("clip", c)
    except Exception:
        # database optional - continue even if missing
        pass

    return AnalyzeResponse(video_id=video_id, clips=clips)


@app.post("/rate")
async def rate_clip(req: RateRequest):
    if db is None:
        return {"status": "ok"}
    # Update first matching clip by index ordering within this video
    clips = db["clip"].find({"video_id": req.video_id}).sort("created_at", 1)
    clips = list(clips)
    if req.index < 0 or req.index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    target = clips[req.index]
    db["clip"].update_one({"_id": target["_id"]}, {"$set": {"rating": req.rating, "updated_at": __import__("datetime").datetime.utcnow()}})
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "Agung Clip Viral Backend Running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
