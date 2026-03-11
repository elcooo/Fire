import io
import os

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from utils.fast_pipeline import load_fast_pipeline

MODEL_ID = os.environ.get("MODEL_ID", "FireRedTeam/FireRed-Image-Edit-1.1")
API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(title="FireRed Image Edit API")

pipe = None

@app.on_event("startup")
def startup_event():
    global pipe
    print(f"Loading pipeline: {MODEL_ID}")
    pipe = load_fast_pipeline(MODEL_ID)
    print("Pipeline ready.")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID}

@app.post("/edit")
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    seed: int = Form(49),
    steps: int = Form(30),
    x_api_key: str | None = Header(default=None),
):
    global pipe

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            output = pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=steps,
                generator=generator,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )

        out_img = output.images[0]
        out_img.save("/workspace/FireRed-Image-Edit/api_test_output.png")

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("ERROR IN /edit:", repr(e))
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
