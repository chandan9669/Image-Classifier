"""
app.py — Gradio Web UI for the CV Detection / Tracking / Classification Pipeline.

Layout
──────
  ┌─────────────────────────────────────────────────┐
  │  🔍  CV Detection & Tracking  (header)           │
  ├────────────────┬────────────────────────────────┤
  │  ⚙ Settings   │  📷 Image Tab │ 🎬 Video Tab    │
  │  (sidebar)    │                                 │
  └────────────────┴────────────────────────────────┘

Run:
    python app.py                    # → http://127.0.0.1:7860
    python app.py --port 8080 This is new brach
    python app.py --share            # public Gradio tunnel
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

# ── Import the pipeline (same directory) ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline, EntityCategory

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cv_app")

# ── Persistent output directory ───────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Lazy pipeline singleton ───────────────────────────────────────────────────
_pipeline_instance: Optional[Pipeline] = None
_pipeline_settings: Dict = {}


def _get_pipeline(
    model_name: str,
    confidence: float,
    iou: float,
    track_buffer: int,
    device: str,
    skip_frames: int = 0,
) -> Pipeline:
    """Return a cached Pipeline; rebuild only when any setting changes."""
    global _pipeline_instance, _pipeline_settings
    key = dict(
        model_name=model_name,
        confidence=confidence,
        iou=iou,
        track_buffer=track_buffer,
        device=device,
        skip_frames=skip_frames,
    )
    if _pipeline_instance is None or key != _pipeline_settings:
        logger.info("Initialising pipeline: %s", key)
        _pipeline_instance = Pipeline(
            model_path=model_name,
            confidence_threshold=confidence,
            iou_threshold=iou,
            device=None if device == "auto" else device,
            output_dir=str(OUTPUT_DIR),
            show_hud=True,
            lost_track_buffer=track_buffer,
            skip_frames=skip_frames,
            log_level="INFO",
        )
        _pipeline_settings = key
    return _pipeline_instance

# ═══════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def process_image(
    image: Optional[np.ndarray],
    model_name: str,
    confidence: float,
    iou: float,
    track_buffer: int,
    device: str,
) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Detect objects in a single uploaded image.
    Returns (annotated_rgb_array, json_text, summary_markdown).
    """
    if image is None:
        return None, _jerr("No image uploaded."), "*Upload an image to get started.*"

    t0 = time.perf_counter()
    try:
        pipeline = _get_pipeline(model_name, confidence, iou, track_buffer, device)

        # Write the numpy frame to a temp JPEG so InputHandler can open it
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=str(OUTPUT_DIR))
        tmp_path = tmp.name
        tmp.close()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        result = pipeline.run(tmp_path)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if result is None:
            return None, _jerr("Pipeline returned no result."), ""

        elapsed = round(time.perf_counter() - t0, 3)
        result["ui_elapsed_s"] = elapsed

        # Load annotated output back as RGB
        ann_bgr = cv2.imread(result["output"])
        ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        json_text = json.dumps(result, indent=2)
        summary   = _make_summary(result)
        logger.info("Image done in %.2fs — %d detections.", elapsed, len(result["analysis"]["detections"]))
        return ann_rgb, json_text, summary

    except Exception as exc:
        logger.exception("Image processing error: %s", exc)
        return None, _jerr(str(exc)), f"**Error:** {exc}"


def process_video(
    video_path: Optional[str],
    model_name: str,
    confidence: float,
    iou: float,
    track_buffer: int,
    skip_frames: int,
    device: str,
) -> Tuple[Optional[str], str]:
    """
    Run detection + tracking on an uploaded video.
    Returns (output_video_path, status_text).
    """
    if video_path is None:
        return None, "⚠️  No video uploaded."

    t0 = time.perf_counter()
    try:
        pipeline = _get_pipeline(model_name, confidence, iou, track_buffer, device, skip_frames)
        pipeline.run(video_path)

        stem     = Path(video_path).stem
        out_path = OUTPUT_DIR / f"{stem}_tracked.mp4"
        elapsed  = round(time.perf_counter() - t0, 1)

        status = f"✅  Done in {elapsed}s\n📁  Saved → {out_path}"
        logger.info("Video done in %.1fs → %s", elapsed, out_path)
        return str(out_path), status

    except Exception as exc:
        logger.exception("Video processing error: %s", exc)
        return None, f"❌  Error: {exc}"


def _jerr(msg: str) -> str:
    return json.dumps({"error": msg}, indent=2)


def _make_summary(result: Dict) -> str:
    counts  = result["analysis"]["counts"]
    n       = len(result["analysis"]["detections"])
    infer   = result.get("inference_ms", "?")
    elapsed = result.get("ui_elapsed_s", "?")
    return (
        f"**{n} detection{'s' if n != 1 else ''}**\n\n"
        f"👤 Persons: **{counts.get('Person', 0)}**  "
        f"🐾 Animals: **{counts.get('Animal', 0)}**  "
        f"📦 Objects: **{counts.get('Object', 0)}**\n\n"
        f"⚡ Inference: {infer} ms &nbsp;|&nbsp; Total: {elapsed} s"
    )

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
/* Header */
#app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0e7490 100%);
    border-radius: 12px;
    padding: 26px 32px 20px;
    margin-bottom: 4px;
}
#app-header h1 { color: #f0f9ff; font-size: 1.9rem; font-weight: 700; margin: 0 0 6px; }
#app-header p  { color: #94d2e8; margin: 0; font-size: 0.92rem; }

/* Settings column */
#settings-col { background: #0f172a; border-radius: 10px; padding: 14px; border: 1px solid #1e3a5f; }

/* Run buttons */
#img-run, #vid-run {
    background: linear-gradient(90deg,#0e7490,#0284c7) !important;
    color: #fff !important; font-weight: 700 !important;
    border-radius: 8px !important; border: none !important;
    min-height: 46px !important;
}

/* Summary card */
#summary { background: #0f172a; border: 1px solid #164e63; border-radius: 10px;
           padding: 14px 18px; color: #e0f2fe; min-height: 90px; }

/* JSON box */
#json-out textarea {
    font-family: 'JetBrains Mono','Fira Code',monospace !important;
    font-size: 0.76rem !important; background: #020617 !important;
    color: #86efac !important; border-radius: 8px !important;
}

/* Video status */
#vid-status textarea { background: #0f172a !important; color: #94d2e8 !important;
                        font-size: 0.85rem !important; border-radius: 8px !important; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# UI BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CV Detection & Tracking") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
          <h1>🔍 CV Detection &amp; Tracking</h1>
          <p>Detect &amp; classify <strong>Persons · Animals · Objects</strong> in images and
             videos using <strong>YOLOv8 + ByteTrack</strong>.</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Settings sidebar ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=230, elem_id="settings-col"):
                gr.Markdown("### ⚙️ Settings")
                model_dd = gr.Dropdown(
                    label="Model weights",
                    choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"],
                    value="yolov8x.pt",
                    info="n=fastest · x=most accurate",
                )
                device_dd = gr.Dropdown(
                    label="Device",
                    choices=["auto","cuda","mps","cpu"],
                    value="auto",
                    info="auto → CUDA › MPS › CPU",
                )
                conf_sl = gr.Slider(0.10, 0.95, value=0.35, step=0.05, label="Confidence threshold")
                iou_sl  = gr.Slider(0.10, 0.95, value=0.45, step=0.05, label="NMS IoU threshold")
                buf_sl  = gr.Slider(5, 120, value=30, step=5,  label="Track buffer (frames)",
                                    info="Higher = tolerates longer occlusions")
                skip_sl = gr.Slider(0, 5,   value=0, step=1,  label="Skip frames (video only)",
                                    info="0 = every frame")
                gr.Markdown("<small style='color:#475569'>Outputs saved to <code>output/</code></small>")

            # ── Main panel ────────────────────────────────────────────────
            with gr.Column(scale=3):
                with gr.Tabs():

                    # ── IMAGE TAB ─────────────────────────────────────────
                    with gr.Tab("📷  Image"):
                        with gr.Row():
                            img_in = gr.Image(
                                label="Upload / Paste Image",
                                type="numpy",
                                height=360,
                            )
                            img_out = gr.Image(
                                label="Annotated Output",
                                type="numpy",
                                interactive=False,
                                height=360,
                            )

                        img_btn = gr.Button("▶  Run Detection", elem_id="img-run", variant="primary")

                        with gr.Row():
                            img_summary = gr.Markdown(
                                value="*Upload an image and press **Run Detection**.*",
                                elem_id="summary",
                            )
                        img_json = gr.Textbox(
                            label="Full JSON Analysis",
                            lines=12,
                            max_lines=28,
                            interactive=False,
                            elem_id="json-out",
                        )

                        # ── Events ────────────────────────────────────────
                        _img_inputs  = [img_in, model_dd, conf_sl, iou_sl, buf_sl, device_dd]
                        _img_outputs = [img_out, img_json, img_summary]

                        img_btn.click(process_image, _img_inputs, _img_outputs)
                        img_in.upload(process_image, _img_inputs, _img_outputs)

                    # ── VIDEO TAB ─────────────────────────────────────────
                    with gr.Tab("🎬  Video"):
                        vid_in = gr.Video(
                            label="Upload Video",
                            height=300,
                        )
                        vid_btn = gr.Button(
                            "▶  Run Detection & Tracking",
                            elem_id="vid-run",
                            variant="primary",
                        )
                        vid_out = gr.Video(
                            label="Annotated & Tracked Output",
                            interactive=False,
                            height=380,
                            autoplay=True,
                        )
                        vid_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            interactive=False,
                            elem_id="vid-status",
                        )

                        _vid_inputs  = [vid_in, model_dd, conf_sl, iou_sl, buf_sl, skip_sl, device_dd]
                        _vid_outputs = [vid_out, vid_status]

                        vid_btn.click(process_video, _vid_inputs, _vid_outputs)

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:14px 0 2px;color:#475569;font-size:0.76rem;">
            YOLOv8 &nbsp;·&nbsp; ByteTrack &nbsp;·&nbsp; OpenCV &nbsp;·&nbsp; Supervision
            &nbsp;|&nbsp; Output: <code>output/</code>
        </div>
        """)

    return demo

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV Pipeline Gradio Web UI")
    p.add_argument("--port",  type=int, default=7860)
    p.add_argument("--host",  type=str, default="127.0.0.1")
    p.add_argument("--share", action="store_true", help="Create a public Gradio tunnel.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info("Starting CV Pipeline UI on http://%s:%d", args.host, args.port)
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        css=_CSS,
    )
