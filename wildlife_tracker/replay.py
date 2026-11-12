"""Offline replay of recorded clips through the live pipeline: decode → model → tracker → step plan.

No WS, no camera and no network. Clips are user-downloaded Eufy recordings
placed under the data mount (default /data/clips). Video time drives the
tracker and planner clocks, so policy timing matches real time. Nothing is
written; results are candidates for human review, not species labels.
"""
import asyncio
import shutil
from contextlib import aclosing
from pathlib import Path

from .core import log
from .ptz import StepPlanner, StepPolicy
from .tracking import AnimalTracker
from .vision import Detector, FRAME_BYTES, HEIGHT, WIDTH

SUFFIXES = {".mp4", ".mov", ".mkv", ".ts", ".h264", ".h265", ".hevc"}
FPS = 3  # same as the live observer


def clips(path):
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if path.is_file():
        return [path]
    raise FileNotFoundError("Replay path absent")


async def frames(path, fps=FPS):
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
        "-vf", f"fps={fps},scale={WIDTH}:{HEIGHT}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        index = 0
        while True:
            try:
                frame = await process.stdout.readexactly(FRAME_BYTES)
            except asyncio.IncompleteReadError:
                break
            yield index / fps, frame
            index += 1
    finally:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.communicate(), 5)


async def replay_clip(detector, path, sample_every=None):
    """sample_every: also keep one decoded frame per this many video seconds (training candidates)."""
    tracker = AnimalTracker()
    planner = StepPlanner(StepPolicy(closed_loop=False))  # recorded view never responds to a plan
    labels, count, tracking, proposals, steps, moving, first = {}, 0, 0, 0, 0, 0, None
    active = []  # video seconds where an animal is evidently present (for archive trimming)
    samples, next_sample = [], 0.0
    seconds = 0.0
    async with aclosing(frames(path)) as stream:
        async for seconds, frame in stream:
            count += 1
            if sample_every and seconds >= next_sample:
                samples.append((round(seconds, 2), frame))
                next_sample = seconds + sample_every
            detector.submit(frame)
            found = await detector.result(60)
            camera = getattr(detector, "camera", None)
            moving += bool(camera and camera.get("moving"))
            for item in found:
                entry = labels.setdefault(item["label"], {"detections": 0, "max_score": 0.0, "first_seconds": round(seconds, 2)})
                entry["detections"] += 1
                entry["max_score"] = max(entry["max_score"], item["score"])
            if found and first is None:
                first = round(seconds, 2)
            decision = tracker.update(found, now=seconds, observed_at=seconds, camera=camera)
            target = tracker.target
            if ((target is not None and target["last_seen"] == seconds and target["matched"] >= 2)
                    or any(item["label"] != "motion" and item["score"] >= .3 for item in found)):
                active.append(round(seconds, 2))
            tracking += decision["state"] == "TRACKING"
            proposals += decision["action"] == "move"
            steps += planner.plan(decision, seconds, camera)["command"] is not None
    tracker.finish()
    dominant = max(labels, key=lambda k: labels[k]["detections"]) if labels else None
    return {"clip": path.name, "seconds": round(seconds + (1 / FPS if count else 0), 2), "frames": count,
            "labels": labels, "dominant_label": dominant, "first_detection_seconds": first,
            "tracking_frames": tracking, "tracking_proposals": proposals, "planned_steps": steps,
            "camera_moving_frames": moving, "active_seconds": active, "samples": samples,
            "review": "candidate_only" if labels else "no_target_detected"}


async def replay(config, path):
    selected = clips(path)
    if not selected:
        raise FileNotFoundError("No supported clips in replay path")
    detector = Detector(config.model_path, config.confidence)
    totals = {"clips": 0, "clips_with_detections": 0, "clips_tracked": 0, "planned_steps": 0}
    try:
        await detector.start()
        for clip in selected:
            await detector.reset()  # Each clip starts with no previous view.
            result = await replay_clip(detector, clip)
            totals["clips"] += 1
            totals["clips_with_detections"] += bool(result["labels"])
            totals["clips_tracked"] += result["tracking_frames"] > 0
            totals["planned_steps"] += result["planned_steps"]
            result.pop("active_seconds")
            result.pop("samples")
            log("replay_clip", source="recorded_clip", **result)
        log("replay_complete", source="recorded_clip", **totals, automatic_ptz=False)
        return totals
    finally:
        detector.close()


def segments(active, duration, pad=2.0, gap=3.0, lead=10.0):
    """Merge animal-present seconds into padded [start, end] ranges within the clip.

    The first `lead` seconds are always kept: an Eufy recording starts at the motion that
    triggered it, and detection can miss there (2026-10-15 03:49: a coyote in heavy rain at
    0–3 s was not detected, and trimming kept only 7–11 s).
    """
    ranges = [[0.0, max(0.0, min(lead, duration) - pad)]] if duration > 0 and lead > 0 else []
    for t in sorted(active):
        if ranges and t - ranges[-1][1] <= gap:
            ranges[-1][1] = t
        else:
            ranges.append([t, t])
    merged = []
    for start, end in ranges:
        start, end = max(0.0, start - pad), min(duration, end + pad)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(round(a, 2), round(b, 2)) for a, b in merged if b > a]


async def cut(source, start, end, target):
    """Copy one range without re-encoding: original quality, always smaller than the source.

    Re-encoding Eufy's 3K HEVC to H.264 made segments larger than the whole recording.
    Stream copy starts at the keyframe before `start`, which the 2 s padding absorbs.
    """
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", str(source),
        "-t", f"{end - start:.2f}", "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-tag:v", "hvc1"
        if await codec(source) == "hevc" else "avc1", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart", str(target),
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, error = await process.communicate()
    if process.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg cut failed: {error.decode(errors='replace')[-200:]}")


async def codec(source):
    process = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
        "-of", "csv=p=0", str(source), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    output, _ = await process.communicate()
    return output.decode().strip()


CANDIDATE_SECONDS = 10   # one training-candidate frame per this much video outside the kept segments
MAX_CANDIDATES = 12      # per clip


def write_candidates(clip, samples, ranges, directory):
    """Save frames from outside the kept segments as JPEGs for later labelling.

    The model saw no animal there, which is not proof of none (it missed a coyote in rain), so
    these are review candidates for negatives (bush, rain, lights, empty yard), never labels.
    """
    import json
    import cv2
    import numpy as np
    outside = [(t, f) for t, f in samples if not any(a <= t <= b for a, b in ranges)][:MAX_CANDIDATES]
    if not outside:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for seconds, frame in outside:
        name = f"{clip.stem}_t{seconds:06.1f}.jpg"
        ok, data = cv2.imencode(".jpg", np.frombuffer(frame, np.uint8).reshape(HEIGHT, WIDTH, 3),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        (directory / name).write_bytes(data.tobytes())
        names.append({"file": name, "seconds": seconds})
    (directory / f"{clip.stem}_candidates.json").write_text(json.dumps(
        {"source": clip.name, "frames": names, "model_detection": False, "review": "unlabelled_candidate"}, indent=1))
    return names


async def archive(config, path, out, candidates=None, keep_undetected=False):
    """Keep only the animal segments of each recording. Never deletes or modifies a source.

    A clip with no detected animal is left untouched and reported (detection can miss); with
    keep_undetected it is also copied whole into `out`, since a recording the user chose to
    download most likely shows an animal the model missed. `candidates`: a directory for frames
    from the discarded parts (see write_candidates).
    Already-archived clips (sidecar present) are skipped, so reruns are safe.
    """
    import json
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = Path(candidates) if candidates else None
    detector = Detector(config.model_path, config.confidence)
    report = {"archived": [], "no_animal_kept": [], "skipped_existing": []}
    try:
        await detector.start()
        for clip in clips(path):
            sidecar = out / f"{clip.stem}.json"
            if sidecar.exists():
                report["skipped_existing"].append(clip.name)
                continue
            await detector.reset()
            result = await replay_clip(detector, clip, CANDIDATE_SECONDS if candidates else None)
            active = result.pop("active_seconds")
            samples = result.pop("samples")
            # Nothing detected anywhere: leave the source alone for the user (detection can miss).
            ranges = segments(active, result["seconds"]) if active else []
            frames = write_candidates(clip, samples, ranges, candidates) if candidates else []
            if not ranges:
                report["no_animal_kept"].append(clip.name)
                if keep_undetected:
                    whole = out / f"{clip.stem}_undetected{clip.suffix.lower()}"
                    shutil.copyfile(clip, whole)
                    sidecar.write_text(json.dumps({"source": clip.name, "source_bytes": clip.stat().st_size,
                        "seconds": result["seconds"], "segments": [], "undetected_copy": whole.name,
                        "labels": result["labels"], "candidate_frames": len(frames)}, indent=1))
                log("archive_clip", clip=clip.name, segments=0, kept_source=True,
                    undetected_copy=keep_undetected, candidate_frames=len(frames))
                continue
            outputs = []
            for index, (start, end) in enumerate(ranges, 1):
                target = out / f"{clip.stem}_{index:02d}_{int(start):03d}-{int(end):03d}s.mp4"
                await cut(clip, start, end, target)
                outputs.append({"start": start, "end": end, "file": target.name, "bytes": target.stat().st_size})
            meta = {"source": clip.name, "source_bytes": clip.stat().st_size, "seconds": result["seconds"],
                    "segments": outputs, "labels": result["labels"], "tracking_frames": result["tracking_frames"],
                    "kept_seconds": round(sum(o["end"] - o["start"] for o in outputs), 2),
                    "candidate_frames": len(frames)}
            sidecar.write_text(json.dumps(meta, indent=1))
            report["archived"].append(clip.name)
            log("archive_clip", clip=clip.name, segments=len(outputs), kept_seconds=meta["kept_seconds"],
                seconds=result["seconds"], source_bytes=meta["source_bytes"],
                archive_bytes=sum(o["bytes"] for o in outputs), candidate_frames=len(frames))
        log("archive_complete", **{k: len(v) for k, v in report.items()}, no_animal_kept_clips=report["no_animal_kept"])
        return report
    finally:
        detector.close()
