"""Whole-frame camera-motion estimate between consecutive decoded frames.

Observation only: it tells the tracker when the view itself moved (native
tracking, wind, or a future verified step) and by how much. Runs inside the
session-scoped model worker so the idle parent never loads OpenCV/numpy.
"""
import math

SMALL = (160, 90)  # width, height of the grayscale correlation image


def prepare(frame, width, height):
    import cv2
    import numpy as np
    image = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, SMALL, interpolation=cv2.INTER_AREA).astype(np.float32)


class CameraMotion:
    """Phase correlation; shift is content displacement in normalized image units.

    Positive dx means scene content moved right in the image (the view turned
    left). A weak correlation peak (low texture, night, large scene change)
    reports shift=None/moving=None: unknown is never reported as still.
    """
    def __init__(self, minimum_response=.2, moving_threshold=.02):
        if not 0 < minimum_response < 1 or not 0 < moving_threshold < .5:
            raise ValueError("Invalid camera-motion thresholds")
        self.minimum_response = minimum_response
        self.moving_threshold = moving_threshold
        self.previous = None
        self.window = None

    def update(self, small):
        import cv2
        if self.window is None:
            self.window = cv2.createHanningWindow(SMALL, cv2.CV_32F)
        previous, self.previous = self.previous, small
        if previous is None:
            return {"shift": None, "response": None, "moving": None}
        (dx, dy), response = cv2.phaseCorrelate(previous, small, self.window)
        if not all(math.isfinite(v) for v in (dx, dy, response)) or response < self.minimum_response:
            return {"shift": None, "response": round(response, 3) if math.isfinite(response) else None, "moving": None}
        shift = (round(dx / SMALL[0], 4), round(dy / SMALL[1], 4))
        return {"shift": shift, "response": round(response, 3),
                "moving": max(abs(shift[0]), abs(shift[1])) >= self.moving_threshold}


MOTION_LABEL = "motion"
MOTION_SCORE = .35  # starts a track (≥0.3) but cannot turn the camera on score alone (<0.4)


def is_grayscale(frame, width, height):
    """Infrared night frames carry no colour; used to hold the stream through Eufy's recording.

    Mean |B-G|+|G-R| on recorded S340 clips: infrared 0–6 (encoder chroma offset), daylight 40–51.
    """
    import numpy as np
    image = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3))[::8, ::8].astype(np.int16)
    return float(np.abs(image[..., 0] - image[..., 1]).mean() + np.abs(image[..., 1] - image[..., 2]).mean()) < 15


class MotionBlobs:
    """Moving regions in a still view (background subtraction): class-agnostic animal evidence.

    Recorded S340 night footage: a small deer in infrared was invisible to the model
    but a continuous moving blob. The top band holds Eufy's changing timestamp and
    is ignored. The background is re-learned whenever the view itself moves, and a
    frame whose foreground is mostly the whole picture (IR switch, exposure jump)
    yields nothing.
    """
    def __init__(self, ignore_top=.07, min_area=.0004, max_area=.25, max_foreground=.2, settle_frames=3):
        self.ignore_top = ignore_top
        self.min_area = min_area
        self.max_area = max_area
        self.max_foreground = max_foreground
        self.settle_frames = settle_frames
        self.reset()

    def reset(self):
        self.subtractor = None
        self.seen = 0

    def update(self, frame, width, height, camera=None):
        import cv2
        import numpy as np
        shift = (camera or {}).get("shift")
        if (camera or {}).get("moving") or (shift is not None and max(abs(shift[0]), abs(shift[1])) > .004):
            self.reset()  # A moved view invalidates the learned background.
        if self.subtractor is None:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=False)
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3))
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = self.subtractor.apply(gray)
        self.seen += 1
        if self.seen <= self.settle_frames:
            return []
        mask[: int(height * self.ignore_top)] = 0
        if mask.mean() / 255 > self.max_foreground:
            return []
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel), self.kernel, iterations=2)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        blobs = []
        for i in range(1, count):
            area = stats[i, cv2.CC_STAT_AREA] / (width * height)
            if self.min_area <= area <= self.max_area:
                blobs.append({"label": MOTION_LABEL, "score": MOTION_SCORE, "area": round(area, 4),
                              "center": [round(centroids[i][0] / width, 3), round(centroids[i][1] / height, 3)]})
        return sorted(blobs, key=lambda b: -b["area"])[:3]
