"""Session-scoped local inference and FFmpeg decoding."""
import asyncio
import multiprocessing as mp
import queue
import time
from pathlib import Path

LABELS = ("deer", "bobcat", "coyote", "raccoon", "rabbit", "squirrel", "cat", "dog")
PROMPTS = tuple(f"a {name}" for name in LABELS)
# MegaDetector V6 is trained on camera-trap images (day and infrared) with one animal class.
# 2026-10-31 on 12 recorded S340 clips it boxed the animal in far more frames than the
# YOLO-World prompts above, at about half the CPU time; person and vehicle boxes are dropped.
MEGADETECTOR_CLASSES = ("animal", "person", "vehicle")
ANIMAL = "animal"
ANIMAL_LABELS = LABELS + (ANIMAL,)
WIDTH, HEIGHT = 640, 360
FRAME_BYTES = WIDTH * HEIGHT * 3


def model_labels(names):
    """Detection label per class index of a supported checkpoint; None marks a class that is not an animal."""
    names = list(names.values())
    if names == list(PROMPTS):
        return LABELS
    if names == list(MEGADETECTOR_CLASSES):
        return (ANIMAL, None, None)
    raise ValueError("Prepared model class set mismatch")


def _worker(inputs, outputs, model_path, confidence):
    try:
        from ultralytics import YOLO
        from .motion import CameraMotion, MotionBlobs, is_grayscale, prepare
        model = YOLO(model_path)
        camera = CameraMotion()
        blobs = MotionBlobs()
        labels = model_labels(model.names)
        outputs.put(("ready", None))
        while True:
            frame = inputs.get()
            if frame is None:
                break
            if frame == b"":
                # Session boundary for a warm model: forget the previous session's last view.
                camera = CameraMotion()
                blobs.reset()
                outputs.put(("reset", None))
                continue
            import numpy as np
            image = np.frombuffer(frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            result = model.predict(image, imgsz=640, conf=confidence, verbose=False, device="cpu")[0]
            detections = []
            for box in result.boxes:
                index = int(box.cls[0])
                if 0 <= index < len(labels) and labels[index]:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    detections.append({"label": labels[index], "score": round(float(box.conf[0]), 3),
                                       "center": [round((x1 + x2) / (2 * WIDTH), 3), round((y1 + y2) / (2 * HEIGHT), 3)],
                                       "area": round((x2 - x1) * (y2 - y1) / (WIDTH * HEIGHT), 3)})
            try:
                motion = camera.update(prepare(frame, WIDTH, HEIGHT))
            except Exception:
                motion = {"shift": None, "response": None, "moving": None}
            try:
                # Moving regions locate animals the model cannot name (small, infrared, half hidden).
                detections += blobs.update(frame, WIDTH, HEIGHT, motion)
                motion["grayscale"] = is_grayscale(frame, WIDTH, HEIGHT)
            except Exception:
                motion["grayscale"] = None
            outputs.put(("detections", {"detections": detections, "camera": motion}))
    except Exception as error:
        outputs.put(("error", type(error).__name__))


class Detector:
    def __init__(self, model_path, confidence):
        self.model_path = model_path
        self.confidence = confidence
        self.context = mp.get_context("spawn")
        self.inputs = self.context.Queue(maxsize=1)
        self.outputs = self.context.Queue(maxsize=2)
        self.process = None
        self.camera = None  # Whole-frame motion for the most recent result; None means unknown.

    async def start(self, timeout=90):
        if self.process is not None:
            if self.process.is_alive():
                return
            raise RuntimeError("Model worker exited before session")
        if not Path(self.model_path).is_file():
            raise FileNotFoundError("Model absent; run explicit model-install first")
        self.process = self.context.Process(target=_worker, args=(self.inputs, self.outputs, self.model_path, self.confidence), daemon=True)
        self.process.start()
        try:
            kind, detail = await self._poll(timeout)
            if kind != "ready":
                raise RuntimeError(f"Model load failed: {detail}")
        except BaseException:
            self.close()
            raise

    def submit(self, frame):
        try:
            if self.inputs.full():
                self.inputs.get_nowait()
            self.inputs.put_nowait(frame)
        except queue.Full:
            pass

    async def _poll(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.outputs.get_nowait()
            except queue.Empty:
                if self.process and not self.process.is_alive():
                    raise RuntimeError("Model worker exited")
                await asyncio.sleep(0.05)
        raise TimeoutError("Model worker deadline")

    def alive(self):
        return self.process is not None and self.process.is_alive()

    async def reset(self, timeout=15):
        """Warm-model session boundary: discard stale results and the previous session's view."""
        if not self.alive():
            raise RuntimeError("Model worker not running")
        self.camera = None
        self.submit(b"")  # Replaces any unconsumed frame; every result before the ack is stale.
        deadline = time.monotonic() + timeout
        while True:
            kind, detail = await self._poll(max(.01, deadline - time.monotonic()))
            if kind == "reset":
                return
            if kind == "error":
                raise RuntimeError(f"Model worker failed: {detail}")

    async def result(self, timeout):
        kind, detail = await self._poll(timeout)
        if kind != "detections":
            raise RuntimeError(f"Inference failed: {detail}")
        self.camera = detail["camera"]
        return detail["detections"]

    def close(self):
        if self.process is not None:
            try:
                self.inputs.put_nowait(None)
            except queue.Full:
                pass
            self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.process = None
        self.inputs.close()
        self.outputs.close()


class Decoder:
    def __init__(self, codec, fps=3):
        fmt = {"H264": "h264", "H265": "hevc", "HEVC": "hevc"}.get(str(codec).upper())
        if fmt is None:
            raise ValueError("Unsupported video codec")
        if fps not in (3, 5):
            raise ValueError("Decode rate is 3 FPS (observer) or 5 FPS (probe)")
        self.fmt = fmt
        self.fps = fps
        self.process = None

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-probesize", "32",
            "-analyzeduration", "0", "-f", self.fmt,
            "-i", "pipe:0", "-an", "-vf", f"fps={self.fps},scale={WIDTH}:{HEIGHT}",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def feed(self, chunk):
        self.process.stdin.write(chunk)
        await asyncio.wait_for(self.process.stdin.drain(), 3)

    async def frame(self, timeout):
        return await asyncio.wait_for(self.process.stdout.readexactly(FRAME_BYTES), timeout)

    async def close(self):
        if self.process:
            if self.process.returncode is None:
                self.process.kill()
            # A paused/full stdout pipe can keep asyncio.wait() pending even
            # after FFmpeg exits. Drain the bounded residual pipe after kill.
            await asyncio.wait_for(self.process.communicate(), 3)
            self.process = None
