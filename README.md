# eufy-wildlife-follow

When an animal walks past an Eufy SoloCam S340 (or E42), this local service recognizes it and can turn the camera to keep it in view. It runs on your own machine. No cloud AI and no subscription are involved.

- Listens for camera events through [eufy-security-ws](https://github.com/bropat/eufy-security-ws).
- Starts a short live stream and decodes it at 3 FPS.
- Finds animals with [MegaDetector V6](https://github.com/microsoft/CameraTraps), a detector trained on millions of day and infrared camera-trap images.
- Optionally pans the camera toward the animal, checks every step against the video, and returns to where it started.

> **Status: experimental.** The Eufy SoloCam S340 (model `T8170`) has run unattended in one backyard since 2026-10, at night under infrared and in daylight. The SoloCam E42 (`T8173`) is accepted for observation; it follows only after you measure its pan step (see below). Other models are refused. Expect to tune it for your yard.

Not affiliated with Anker or Eufy. The Eufy side uses an unofficial API that can change at any time.

## How it works

```
camera event (motion / pet / radar)
  → start live stream (only if no one else is streaming)
  → FFmpeg 640×360 @ 3 FPS
  → MegaDetector: animal boxes      ─┐
  → whole-frame motion estimate      ├→ tracker: one confirmed target, held across crossings
  → moving-region detector          ─┘
  → follow (optional): one pan step at a time, each verified by measured view shift
  → target lost → step back to the start view → stop the stream it started
```

**Day and night are handled differently.** The service decides by checking whether the frames are grayscale, which means infrared.

| | Daylight | Night (infrared) |
|---|---|---|
| Session length | up to 30 s | held up to 130 s from stream start |
| No animal in the first 12 s | stop | keep watching |
| Animal lost for 8 s | step back, end the session | step back, keep watching |

At night the stream stays open anyway while Eufy records the clip, and stopping it early switched the infrared light off. So the service keeps watching (and following) for the whole recording.

### Following, and how it fails safe

Following is off unless `WILDLIFE_FOLLOW=1`. It also stays off unless the bridge reports the verified versions (server 3.1.0, driver 4.1.0).

- **Pan only, by a measured step.** One S340 pan step moves the image by about 11.8% of its width; that value is built in. Any other camera follows only after you measure its step with `ptz-probe` and set `WILDLIFE_PAN_STEP`; until then it logs `follow_mode` as `pan_step_unmeasured` and only observes. Tilt is used only after you measure it with `ptz-probe --axis tilt`.
- **Write before send.** Every step goes into a durable ledger before it is sent. Afterwards, the measured shift of the whole frame must match the expected size and direction.
- **Limits.** At most 16 forward steps per session, and at most 5 steps away from the start view.
- **People.** A native person event from the camera means no forward steps in that session.
- **Movement alone is not enough.** A track seen only by movement, never by the model (insects, rain), never turns the camera.
- **Failed checks.** If a step does not check out, the position is no longer known, so the session stops sending steps and does not try to return. The next session treats wherever the camera points as its new start. `ptz-ledger-clear` exists for manual recovery.

The service never writes camera settings and never sends `ROTATE360`, presets or calibration. When following is off, it sends no movement commands at all.

## Requirements

| | |
|---|---|
| Camera | Eufy SoloCam S340 (`T8170`), or SoloCam E42 (`T8173`) after measuring its pan step |
| Bridge | eufy-security-ws 3.1.0, logged in with a separate Eufy account that the camera is shared with |
| Host | Docker; 2 CPU cores and about 1.5 GB RAM. It runs at about 120 ms per frame on an Apple M4 Mac mini under OrbStack. Any x86-64 or arm64 Linux host should work. No GPU is needed. |
| Network | The host reaches Eufy's cloud and the camera (P2P) |

## Quick start

```sh
git clone https://github.com/<you>/eufy-wildlife-follow && cd eufy-wildlife-follow
cp .env.example .env            # fill in EUFY_USERNAME / EUFY_PASSWORD
docker compose build
docker compose up -d eufy-security-ws
```

The first login may require a verification code. See the [eufy-security-ws docs](https://bropat.github.io/eufy-security-ws/) and check `docker compose logs eufy-security-ws`.

Then work through these steps with one-off containers (`docker compose run --rm wildlife-follow python -m wildlife_tracker <action>`):

1. **`model-install`**: downloads MegaDetector V6 compact (5.8 MB) from Zenodo, verifies its pinned SHA-256 and class names, and saves it under `data/wildlife/models/`.
2. **`inventory --show-serials`**: lists cameras on the account. Serials are printed only to your terminal. Copy the camera's serial into `WILDLIFE_DEVICE_SERIAL` in `.env`. One container watches one camera; for a second camera run a second service with its own serial and data directory.
3. **`preflight`**: runs read-only checks of the connection, camera identity, versions, settings and stream state.
4. **`trial --seconds 120`**: one supervised session. Close live view in the Eufy app first. Type `START` when it prints `READY`, then walk through the view. It checks the full pipeline and cleans up.
5. Set `WILDLIFE_ENABLED=1` in `.env` and run `docker compose up -d wildlife-follow`.
6. Once a few nights of observation look right, set `WILDLIFE_FOLLOW=1` and recreate the container.

While `WILDLIFE_ENABLED=0`, the `run` action refuses to start. Use one-off `run --rm` containers until step 5.

Watch it with `docker compose logs -f wildlife-follow`. Every line is a JSON event, such as `animal_observed`, `follow_step`, `follow_step_measured`, `target_lost` and `session_complete`.

## Commands

| Action | What it does | Talks to the camera? |
|---|---|---|
| `run` | the service (default command) | yes |
| `model-install [--model world]` | download and verify the model | no |
| `model-check` | load the model, time inference on blank frames | no |
| `inventory [--show-serials]` | list devices and supported commands | read-only |
| `preflight` | read-only readiness checks | read-only |
| `capture-events --seconds N` | log redacted events, no streaming | read-only |
| `probe-stream --seconds N` | start and decode a short stream, no inference | streams |
| `trial` / `pilot` | supervised single or few-session acceptance | streams |
| `ptz-probe [--axis pan\|tilt]` | supervised step measurement, one confirmed step at a time | moves |
| `reconcile` | clear a stale stream-ownership marker once the stream is confirmed stopped | read-only |
| `ptz-ledger-clear` | manual follow recovery | no |
| `replay --path clip.mp4` | run detection and tracking on recorded clips | no |
| `archive-clips --path in --out out` | keep only the animal parts of recordings | no |

`replay` is the main tool for tuning: drop exported recordings into `data/wildlife/clips/` and compare results before and after a change.

## Configuration

All settings live in `.env`; see [`.env.example`](.env.example). The ones you are likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `WILDLIFE_ENABLED` | `0` | must be `1` for the service to run |
| `WILDLIFE_FOLLOW` | `0` | `1` lets it pan the camera |
| `WILDLIFE_MODE` | `observe` | `listen` logs events and never streams |
| `WILDLIFE_SESSION_SECONDS` | `30` | daylight session cap |
| `WILDLIFE_NIGHT_HOLD_SECONDS` | `130` | infrared session hold |
| `WILDLIFE_CONFIDENCE` | `0.1` | detector floor; the tracker decides what counts |
| `WILDLIFE_MIN_BATTERY` | `25` | skip sessions below this battery % |
| `WILDLIFE_MAX_NET_STEPS` | `5` | furthest pan from the start view |
| `WILDLIFE_PAN_STEP` | built in for S340 | measured pan step (fraction of image width) from `ptz-probe`; required to follow with any other model |

## Privacy and safety

- Everything runs locally. Camera frames stay in memory and are never written to disk, except by `archive-clips`, which you run on your own recordings.
- Logs redact serials. Device identity is stored only as a hash.
- The service starts a stream only when the camera is idle, and stops only the stream it started. If a command outcome is ambiguous, an ownership marker blocks new sessions until `reconcile` confirms the stream is stopped.
- Streaming uses battery. Sessions are bounded, a cooldown follows each session, and low battery skips sessions.

## Known limitations

- Only the S340 and E42 are accepted, and only the S340 is field-tested. Another pan-and-tilt model needs a code change to accept it and a `ptz-probe` measurement to follow; see `SUPPORTED_MODELS` in [`__main__.py`](wildlife_tracker/__main__.py) and `PAN_STEPS` in [`follow.py`](wildlife_tracker/follow.py).
- MegaDetector says "animal", not which species.
- Small or distant animals at night are sometimes missed. Low-scoring static false boxes are filtered by the tracker, not removed.
- Daytime pan steps measure smaller than at night. The check tolerance is widened to cover this, but the cause is not yet understood.

## Development

```sh
pip install -r requirements-test.txt   # plus ffmpeg on PATH
PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_*.py'
```

The unit tests mock the model and the WebSocket server, so they need neither a camera nor model weights. `tests/operational_acceptance.py` runs the real model and FFmpeg inside the container, with no camera.

## License and credits

AGPL-3.0; see [LICENSE](LICENSE). This follows from the dependencies below.

- [MegaDetector](https://github.com/microsoft/CameraTraps) V6 (MDV6-yolov10-c, [weights on Zenodo](https://zenodo.org/records/15398270)), from Microsoft AI for Good and the PyTorch-Wildlife team.
- [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), used to run the detector.
- [eufy-security-ws](https://github.com/bropat/eufy-security-ws) and [eufy-security-client](https://github.com/bropat/eufy-security-client) by bropat.
