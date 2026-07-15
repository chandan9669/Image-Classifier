"""
pipeline.py — Production-Ready Human / Animal / Object Detection,
              Tracking, and Classification Pipeline.

Stack  : Python 3.10+ | PyTorch | Ultralytics YOLOv8 | Supervision
         (ByteTrack) | OpenCV | Rich logging

Author : Senior CV & ML Engineer
Version: 1.0.0
"""

# ── Standard library ────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

# ── Third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from ultralytics import YOLO
import supervision as sv
from PIL import Image


# ── Logging setup ────────────────────────────────────────────────────────────
_console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=_console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("cv_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

class EntityCategory(str, Enum):
    """High-level semantic category assigned to each detected entity."""
    PERSON  = "Person"
    ANIMAL  = "Animal"
    OBJECT  = "Object"


# COCO class indices that map to each category.
# Reference: https://docs.ultralytics.com/datasets/detect/coco/
_PERSON_CLASSES: frozenset[int] = frozenset({0})            # person

_ANIMAL_CLASSES: frozenset[int] = frozenset({               # all COCO animals
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,                # bird→bear
})

# Everything else is treated as a generic Object.

# BGR colour palette — one colour per class index (mod len)
_PALETTE: List[Tuple[int, int, int]] = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31),
    (255, 178,  29), (207, 210,   49), ( 72, 249,  10),
    ( 146, 204,  23), ( 61, 219, 134), ( 26, 147,  52),
    ( 0, 212, 187), (44, 153, 168), (0, 194, 255),
    ( 52,  69, 147), (100,  45, 255), (142, 46, 210),
    (204,  37, 41), (167, 103,  38), (109, 198,  25),
]

def _class_colour(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


# ═══════════════════════════════════════════════════════════════════════════
# 2. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Normalised detection result for a single entity."""
    track_id:   Optional[int]
    class_id:   int
    class_name: str
    category:   EntityCategory
    confidence: float
    bbox:       Tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max)

    def to_dict(self) -> Dict:
        return {
            "track_id":   self.track_id,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "category":   self.category.value,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x_min": self.bbox[0],
                "y_min": self.bbox[1],
                "x_max": self.bbox[2],
                "y_max": self.bbox[3],
            },
        }


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame or image."""
    frame_index:  int
    timestamp_ms: float
    detections:   List[Detection] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {cat.value: 0 for cat in EntityCategory}
        for det in self.detections:
            tally[det.category.value] += 1
        return tally

    def to_dict(self) -> Dict:
        return {
            "frame_index":  self.frame_index,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "counts":       self.counts,
            "detections":   [d.to_dict() for d in self.detections],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════════════

class InputHandler:
    """
    Validates and categorises an input source as either a static image,
    a local video file, or a live stream URL (RTSP / HTTP).
    """

    _IMAGE_SUFFIXES: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    )
    _VIDEO_SUFFIXES: frozenset[str] = frozenset(
        {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}
    )

    def __init__(self, source: Union[str, Path]) -> None:
        self.source: str = str(source)
        self._is_stream: bool = self._detect_stream()
        self._is_image: bool  = self._detect_image()
        self._is_video: bool  = self._detect_video()
        self._validate()

    # ── Private helpers ──────────────────────────────────────────────────

    def _detect_stream(self) -> bool:
        lowered = self.source.lower()
        return lowered.startswith("rtsp://") or lowered.startswith("http://") \
               or lowered.startswith("https://")

    def _detect_image(self) -> bool:
        if self._is_stream:
            return False
        return Path(self.source).suffix.lower() in self._IMAGE_SUFFIXES

    def _detect_video(self) -> bool:
        if self._is_stream:
            return True
        return Path(self.source).suffix.lower() in self._VIDEO_SUFFIXES

    def _validate(self) -> None:
        if self._is_stream:
            logger.info("Source recognised as a [bold]live stream[/bold]: %s", self.source)
            return
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Input source not found: {self.source}")
        if not self._is_image and not self._is_video:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported images: {self._IMAGE_SUFFIXES}, "
                f"Supported videos: {self._VIDEO_SUFFIXES}"
            )
        logger.info(
            "Source validated as [bold]%s[/bold]: %s",
            "image" if self._is_image else "video",
            self.source,
        )

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def is_image(self) -> bool:
        return self._is_image

    @property
    def is_video(self) -> bool:
        return self._is_video or self._is_stream

    def open_capture(self) -> cv2.VideoCapture:
        """Return an opened cv2.VideoCapture for video / stream sources."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        return cap

    def read_image(self) -> np.ndarray:
        """Load a static image as a BGR numpy array."""
        img = cv2.imread(self.source)
        if img is None:
            raise RuntimeError(f"cv2.imread failed for: {self.source}")
        return img


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

class ModelInference:
    """
    Wraps an Ultralytics YOLO model and provides a clean interface for
    running inference on single frames with configurable thresholds.
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.device               = device or self._auto_device()

        logger.info(
            "Loading model [bold]%s[/bold] on device [bold]%s[/bold] …",
            model_path, self.device,
        )
        try:
            self.model: YOLO = YOLO(model_path)
            self.model.to(self.device)
        except Exception as exc:
            logger.exception("Model load failed: %s", exc)
            raise

        self.class_names: Dict[int, str] = self.model.names  # type: ignore[assignment]
        logger.info("Model loaded — %d classes available.", len(self.class_names))

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Auto-selected device: [bold]%s[/bold]", device)
        return device

    @staticmethod
    def _resolve_category(class_id: int) -> EntityCategory:
        if class_id in _PERSON_CLASSES:
            return EntityCategory.PERSON
        if class_id in _ANIMAL_CLASSES:
            return EntityCategory.ANIMAL
        return EntityCategory.OBJECT

    # ── Public API ───────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single BGR frame and return a
        supervision.Detections object.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return sv.Detections.from_ultralytics(results[0])

    def parse_detections(
        self,
        sv_detections: sv.Detections,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
    ) -> FrameAnalysis:
        """Convert a supervision.Detections object into a FrameAnalysis."""
        analysis = FrameAnalysis(frame_index=frame_index, timestamp_ms=timestamp_ms)

        if len(sv_detections) == 0:
            return analysis

        for i in range(len(sv_detections)):
            xyxy       = sv_detections.xyxy[i]
            class_id   = int(sv_detections.class_id[i])
            confidence = float(sv_detections.confidence[i])
            track_id   = (
                int(sv_detections.tracker_id[i])
                if sv_detections.tracker_id is not None
                else None
            )
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            category   = self._resolve_category(class_id)

            bbox = (
                int(xyxy[0]), int(xyxy[1]),
                int(xyxy[2]), int(xyxy[3]),
            )
            analysis.detections.append(
                Detection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    category=category,
                    confidence=confidence,
                    bbox=bbox,
                )
            )
        return analysis


# ═══════════════════════════════════════════════════════════════════════════
# 5. OBJECT TRACKER  (ByteTrack via Supervision)
# ═══════════════════════════════════════════════════════════════════════════

class ObjectTracker:
    """
    Wraps supervision's ByteTrack implementation to assign persistent
    IDs across video frames while handling brief occlusions gracefully.
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        frame_rate                  : Nominal FPS used to scale track buffers.
        lost_track_buffer           : Frames to keep a lost track alive
                                      (handles occlusions).
        minimum_matching_threshold  : IoU threshold for re-association.
        minimum_consecutive_frames  : Detections needed before a new track
                                      is confirmed.
        """
        self.tracker = sv.ByteTrack(
            frame_rate=int(frame_rate),
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        logger.info(
            "ByteTrack initialised — buffer=%d frames, IoU threshold=%.2f",
            lost_track_buffer,
            minimum_matching_threshold,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Feed raw detections into the tracker and receive back detections
        enriched with `tracker_id` arrays.
        """
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state (useful between unrelated video clips)."""
        self.tracker.reset()
        logger.debug("Tracker state reset.")


# ═══════════════════════════════════════════════════════════════════════════
# 6. VISUALISER
# ═══════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Renders anti-aliased bounding boxes, category-coloured labels, and
    optional frame-level HUD (frame index, FPS, entity counts) onto BGR
    numpy arrays.
    """

    # Font parameters
    _FONT            = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE      = 0.55
    _FONT_THICKNESS  = 1
    _BOX_THICKNESS   = 2
    _LABEL_PADDING   = 5

    # Per-category colour overrides (BGR)
    _CATEGORY_COLOURS: Dict[EntityCategory, Tuple[int, int, int]] = {
        EntityCategory.PERSON: (0, 200, 255),    # amber
        EntityCategory.ANIMAL: (50, 205,  50),   # lime green
        EntityCategory.OBJECT: (70,  70, 255),   # coral-red
    }

    def __init__(self, show_hud: bool = True) -> None:
        self.show_hud = show_hud

    # ── Private helpers ──────────────────────────────────────────────────

    def _box_colour(self, detection: Detection) -> Tuple[int, int, int]:
        """Return the BGR colour for a detection's category."""
        return self._CATEGORY_COLOURS.get(detection.category, _class_colour(detection.class_id))

    def _build_label(self, det: Detection) -> str:
        id_part   = f" #{det.track_id}" if det.track_id is not None else ""
        conf_part = f"{det.confidence * 100:.1f}%"
        return f"{det.class_name}{id_part} | {conf_part}"

    @staticmethod
    def _draw_aa_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        colour: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw an anti-aliased rectangle using cv2.LINE_AA."""
        cv2.rectangle(img, pt1, pt2, colour, thickness, lineType=cv2.LINE_AA)

    def _draw_label(
        self,
        img: np.ndarray,
        label: str,
        anchor: Tuple[int, int],
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw a filled label background and white text above the box."""
        (tw, th), baseline = cv2.getTextSize(
            label, self._FONT, self._FONT_SCALE, self._FONT_THICKNESS
        )
        p   = self._LABEL_PADDING
        x, y = anchor

        # Clip so the label never goes off-screen at the top
        y_text = max(y - baseline - p, th + p)
        x2 = min(x + tw + 2 * p, img.shape[1])

        # Filled background
        cv2.rectangle(
            img,
            (x, y_text - th - 2 * p),
            (x2, y_text + baseline),
            colour,
            cv2.FILLED,
        )
        # White text
        cv2.putText(
            img,
            label,
            (x + p, y_text - p),
            self._FONT,
            self._FONT_SCALE,
            (255, 255, 255),
            self._FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def annotate(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw all detections onto *a copy* of `frame` and return it.
        The original frame is never mutated.
        """
        canvas = frame.copy()

        for det in analysis.detections:
            colour = self._box_colour(det)
            x1, y1, x2, y2 = det.bbox

            self._draw_aa_rect(canvas, (x1, y1), (x2, y2), colour, self._BOX_THICKNESS)
            self._draw_label(canvas, self._build_label(det), (x1, y1), colour)

        if self.show_hud:
            self._draw_hud(canvas, analysis, fps)

        return canvas

    def _draw_hud(
        self,
        canvas: np.ndarray,
        analysis: FrameAnalysis,
        fps: Optional[float],
    ) -> None:
        """Overlay a small semi-transparent HUD in the top-right corner."""
        h, w = canvas.shape[:2]
        counts = analysis.counts
        lines = [
            f"Frame: {analysis.frame_index}",
            f"Persons: {counts[EntityCategory.PERSON.value]}",
            f"Animals: {counts[EntityCategory.ANIMAL.value]}",
            f"Objects: {counts[EntityCategory.OBJECT.value]}",
        ]
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")

        line_h  = 22
        box_w   = 170
        box_h   = len(lines) * line_h + 12
        x_start = w - box_w - 10
        y_start = 10

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x_start, y_start), (w - 10, y_start + box_h), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, line in enumerate(lines):
            cv2.putText(
                canvas, line,
                (x_start + 8, y_start + 18 + i * line_h),
                self._FONT, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
            )


# ═══════════════════════════════════════════════════════════════════════════
# 7. IMAGE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class ImageProcessor:
    """
    Handles the end-to-end pipeline for a **single static image**:
      1. Load  →  2. Infer  →  3. Annotate  →  4. Save  →  5. Return JSON
    """

    def __init__(
        self,
        inference: ModelInference,
        visualizer: Visualizer,
        output_dir: Union[str, Path] = "output",
    ) -> None:
        self.inference  = inference
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, source: Union[str, Path]) -> Dict:
        """
        Run the full pipeline on a static image.

        Returns
        -------
        dict
            JSON-serialisable analysis containing all detections.
        """
        handler = InputHandler(source)
        if not handler.is_image:
            raise ValueError(f"'{source}' is not a recognised image file.")

        frame = handler.read_image()
        logger.info("Image loaded — size: %dx%d px", frame.shape[1], frame.shape[0])

        t0          = time.perf_counter()
        sv_dets     = self.inference.predict(frame)
        infer_ms    = (time.perf_counter() - t0) * 1000.0
        logger.info("Inference completed in [bold]%.1f ms[/bold].", infer_ms)

        analysis    = self.inference.parse_detections(sv_dets, frame_index=0, timestamp_ms=0.0)
        annotated   = self.visualizer.annotate(frame, analysis)

        # ── Save annotated image ─────────────────────────────────────────
        stem      = Path(source).stem
        out_path  = self.output_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("Annotated image saved → [bold]%s[/bold]", out_path)

        # ── Build JSON result ────────────────────────────────────────────
        result = {
            "source":          str(source),
            "output":          str(out_path),
            "inference_ms":    round(infer_ms, 2),
            "image_size":      {"width": frame.shape[1], "height": frame.shape[0]},
            "analysis":        analysis.to_dict(),
        }

        json_path = self.output_dir / f"{stem}_analysis.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("JSON analysis saved → [bold]%s[/bold]", json_path)

        return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """
    Handles the end-to-end pipeline for **video files and live streams**:
      1. Open capture  →  2. Infer per-frame  →  3. Track  →
      4. Annotate  →  5. Write MP4 (H.264)  →  6. Log per-frame JSON
    """

    # H.264 FourCC
    _FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # works universally; re-encode with ffmpeg for true H.264

    def __init__(
        self,
        inference: ModelInference,
        tracker:   ObjectTracker,
        visualizer: Visualizer,
        output_dir: Union[str, Path] = "output",
        max_frames: Optional[int]    = None,
        skip_frames: int             = 0,
    ) -> None:
        """
        Parameters
        ----------
        max_frames  : Stop after this many frames (None = process all).
        skip_frames : Process 1 in every (skip_frames+1) frames for speed.
                      E.g. skip_frames=1 processes every other frame.
        """
        self.inference   = inference
        self.tracker     = tracker
        self.visualizer  = visualizer
        self.output_dir  = Path(output_dir)
        self.max_frames  = max_frames
        self.skip_frames = skip_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ──────────────────────────────────────────────────

    def _frame_generator(
        self, cap: cv2.VideoCapture
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_index, timestamp_ms, bgr_frame) tuples."""
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if self.skip_frames > 0 and idx % (self.skip_frames + 1) != 0:
                idx += 1
                continue
            yield idx, ts_ms, frame
            idx += 1
            if self.max_frames is not None and idx >= self.max_frames:
                break

    @staticmethod
    def _get_video_meta(cap: cv2.VideoCapture) -> Tuple[int, int, float, int]:
        """Return (width, height, fps, total_frames)."""
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return w, h, fps, total

    # ── Public API ───────────────────────────────────────────────────────

    def process(self, source: Union[str, Path]) -> None:
        """
        Run the full tracking pipeline on a video file or stream.
        Annotated output is written to `output_dir`.
        Per-frame analysis is emitted via logging at DEBUG level
        and summarised at INFO level.
        """
        handler = InputHandler(source)
        if not handler.is_video:
            raise ValueError(f"'{source}' is not a recognised video/stream source.")

        self.tracker.reset()
        cap = handler.open_capture()

        try:
            w, h, fps, total_frames = self._get_video_meta(cap)
            logger.info(
                "Video meta — %dx%d @ %.2f FPS, ~%d frames total.",
                w, h, fps, total_frames,
            )

            # ── Writer setup ─────────────────────────────────────────────
            stem     = Path(str(source)).stem
            out_path = self.output_dir / f"{stem}_tracked.mp4"
            writer   = cv2.VideoWriter(str(out_path), self._FOURCC, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {out_path}")

            # ── Progress bar ─────────────────────────────────────────────
            progress_total = (
                min(total_frames, self.max_frames) if self.max_frames else total_frames
            )
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=_console,
            )

            fps_timer   = time.perf_counter()
            fps_counter = 0
            live_fps    = 0.0
            all_counts: Dict[str, int] = {cat.value: 0 for cat in EntityCategory}

            with progress:
                task_id = progress.add_task(
                    f"Processing [cyan]{stem}[/cyan]", total=progress_total
                )
                for frame_idx, ts_ms, frame in self._frame_generator(cap):
                    try:
                        sv_dets  = self.inference.predict(frame)
                        sv_dets  = self.tracker.update(sv_dets)
                        analysis = self.inference.parse_detections(sv_dets, frame_idx, ts_ms)
                        canvas   = self.visualizer.annotate(frame, analysis, live_fps)
                    except Exception as exc:
                        logger.warning("Frame %d inference error: %s — skipping.", frame_idx, exc)
                        canvas = frame

                    writer.write(canvas)

                    # ── Per-frame log ─────────────────────────────────────
                    logger.debug("Frame %d: %s", frame_idx, json.dumps(analysis.to_dict()))

                    # ── Aggregate counts ──────────────────────────────────
                    for cat, n in analysis.counts.items():
                        all_counts[cat] += n

                    # ── Live FPS ──────────────────────────────────────────
                    fps_counter += 1
                    elapsed = time.perf_counter() - fps_timer
                    if elapsed >= 2.0:
                        live_fps    = fps_counter / elapsed
                        fps_timer   = time.perf_counter()
                        fps_counter = 0

                    progress.advance(task_id)

            writer.release()

        finally:
            cap.release()

        logger.info(
            "Video processing complete → [bold]%s[/bold]\n"
            "  Aggregate detections — Persons: %d | Animals: %d | Objects: %d",
            out_path,
            all_counts.get(EntityCategory.PERSON.value, 0),
            all_counts.get(EntityCategory.ANIMAL.value, 0),
            all_counts.get(EntityCategory.OBJECT.value, 0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. PIPELINE FACTORY  (convenience entry-point)
# ═══════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    High-level factory that wires up all components and exposes a single
    `run(source)` method that automatically dispatches to the correct
    sub-processor based on the input type.
    """

    def __init__(
        self,
        model_path:            str   = "yolov8x.pt",
        confidence_threshold:  float = 0.35,
        iou_threshold:         float = 0.45,
        device:                Optional[str] = None,
        output_dir:            str   = "output",
        show_hud:              bool  = True,
        lost_track_buffer:     int   = 30,
        min_match_threshold:   float = 0.8,
        max_frames:            Optional[int] = None,
        skip_frames:           int   = 0,
        log_level:             str   = "INFO",
    ) -> None:
        logging.getLogger("cv_pipeline").setLevel(log_level.upper())

        self._inference = ModelInference(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        self._tracker = ObjectTracker(
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=min_match_threshold,
        )
        self._visualizer = Visualizer(show_hud=show_hud)

        self._image_proc = ImageProcessor(
            inference=self._inference,
            visualizer=self._visualizer,
            output_dir=output_dir,
        )
        self._video_proc = VideoProcessor(
            inference=self._inference,
            tracker=self._tracker,
            visualizer=self._visualizer,
            output_dir=output_dir,
            max_frames=max_frames,
            skip_frames=skip_frames,
        )

    def run(self, source: Union[str, Path]) -> Optional[Dict]:
        """
        Automatically detect input type and run the appropriate processor.

        Returns
        -------
        dict | None
            JSON-serialisable result for images; None for videos
            (video results are written to disk and logged).
        """
        handler = InputHandler(source)
        if handler.is_image:
            return self._image_proc.process(source)
        else:
            self._video_proc.process(source)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# 10. CLI  &  __main__
# ═══════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Production-grade human / animal / object detection, "
            "tracking, and classification pipeline.\n\n"
            "Examples:\n"
            "  python pipeline.py --source photo.jpg\n"
            "  python pipeline.py --source clip.mp4 --skip-frames 1\n"
            "  python pipeline.py --source rtsp://192.168.1.10:554/live\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--source", "-s", required=True,
        help="Path to an image, video file, or stream URL (RTSP/HTTP).",
    )
    parser.add_argument(
        "--output-dir", "-o", default="output",
        help="Directory where annotated outputs are saved. (default: output/)",
    )

    # Model
    parser.add_argument(
        "--model", "-m", default="yolov8x.pt",
        help="Ultralytics model weights file or name. (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="Torch device override: cuda / mps / cpu. (default: auto-detect)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Minimum detection confidence. (default: 0.35)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS. (default: 0.45)",
    )

    # Tracker
    parser.add_argument(
        "--track-buffer", type=int, default=30,
        help="Frames to keep a lost track alive. (default: 30)",
    )
    parser.add_argument(
        "--match-thresh", type=float, default=0.8,
        help="ByteTrack IoU matching threshold. (default: 0.8)",
    )

    # Video
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Process only the first N frames of a video. (default: all)",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Skip N frames between processed frames for speed. (default: 0)",
    )

    # Misc
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Disable the HUD overlay on annotated frames.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    logging.getLogger("cv_pipeline").setLevel(args.log_level.upper())
    # Silence ultralytics' own verbose output at INFO; set to WARNING.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("CV Pipeline starting …")
    logger.info("  source    : %s", args.source)
    logger.info("  model     : %s", args.model)
    logger.info("  output    : %s", args.output_dir)
    logger.info("=" * 60)

    try:
        pipeline = Pipeline(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            output_dir=args.output_dir,
            show_hud=not args.no_hud,
            lost_track_buffer=args.track_buffer,
            min_match_threshold=args.match_thresh,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
            log_level=args.log_level,
        )

        result = pipeline.run(args.source)

        if result is not None:
            # Static image — pretty-print JSON to console
            _console.print_json(json.dumps(result))

        logger.info("Pipeline finished successfully.")
        return 0

    except FileNotFoundError as exc:
        logger.error("Input not found: %s", exc)
        return 2
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


# ── Programmatic usage example (also serves as integration smoke-test) ───

def _demo_programmatic() -> None:
    """
    Shows how to embed the pipeline directly in another Python module
    without the CLI.  Replace the paths with your own assets.
    """
    pipeline = Pipeline(
        model_path="yolov8x.pt",
        confidence_threshold=0.40,
        iou_threshold=0.45,
        output_dir="output",
        show_hud=True,
        lost_track_buffer=60,   # longer buffer for dense scenes
        log_level="INFO",
    )

    # ── Static image ─────────────────────────────────────────────────────
    image_result = pipeline.run("assets/sample_image.jpg")
    if image_result:
        logger.info(
            "Image analysis complete — %d detections.",
            len(image_result["analysis"]["detections"]),
        )

    # ── Video file ───────────────────────────────────────────────────────
    pipeline.run("assets/sample_video.mp4")

    # ── Live RTSP stream (uncomment to use) ──────────────────────────────
    # pipeline.run("rtsp://username:password@192.168.1.10:554/stream1")


if __name__ == "__main__":
    sys.exit(main())
