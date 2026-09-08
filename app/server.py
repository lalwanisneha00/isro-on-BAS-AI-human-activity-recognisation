"""FastAPI application serving the live video stream and the dashboard shell."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .camera import CameraSource
from .pipeline import ProcessingPipeline

WEB_DIR = Path(__file__).parent / "web"

camera = CameraSource()
pipeline = ProcessingPipeline(camera)


@asynccontextmanager
async def lifespan(_: FastAPI):
    camera.start()
    pipeline.start()
    yield
    pipeline.stop()
    camera.stop()


app = FastAPI(title="BAS Crew Activity Recognition", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


def _mjpeg_frames():
    """Yield JPEG frames as a multipart stream the browser renders as video."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
    frame_interval = 1.0 / config.TARGET_FPS

    while True:
        started = time.time()
        frame = pipeline.read()
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")
        remaining = frame_interval - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/video")
def video():
    return StreamingResponse(
        _mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/mode")
def set_mode(payload: dict = Body(default={})):
    """Switch between the live camera and Demo Mode."""
    return pipeline.set_mode(payload.get("mode", "live"))


@app.get("/api/log")
def log(limit: int = 25):
    """Logged activity segments and cumulative time per activity."""
    return pipeline.log_view(limit)


@app.get("/api/activity")
def activity():
    """Current activity label with its confidence and supporting evidence."""
    return pipeline.activity()


@app.get("/api/telemetry")
def telemetry():
    """Normalisation numbers, polled fast enough to watch them move."""
    return pipeline.telemetry()


@app.get("/api/status")
def status():
    return {
        "camera_status": camera.status,
        "detail": camera.detail,
        "fps": camera.fps,
        "crew_tracked": len(pipeline.crew.tracks),
        "pose_status": pipeline.pose_status,
        "pose_detail": pipeline.pose_detail,
        "process_fps": pipeline.process_fps,
        "latency_ms": pipeline.latency_ms,
        "mission": config.MISSION_NAME,
        "module": config.MODULE_NAME,
        **pipeline.mode_state(),
    }
