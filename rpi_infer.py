```python
#!/usr/bin/env python3
"""
rpi_infer.py
============
Real-time waste classification (metal / plastic / organic).

Green conveyor detection:
- If the center-crop region is >= 95% green, skip classification.
- Displays "CONVEYOR - NO ITEM".
- White/black blank-scene detection is preserved.
- Green detection can be disabled with --no-green-check.

Usage:
    python3 rpi_infer.py --model waste_classifier_float32_tf.tflite
    python3 rpi_infer.py --green-fraction 0.90
    python3 rpi_infer.py --no-green-check
    python3 rpi_infer.py --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# TFLite runtime
OpResolverType = None

try:
    from ai_edge_litert.interpreter import Interpreter
    try:
        from ai_edge_litert.interpreter import OpResolverType
    except ImportError:
        pass
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        try:
            from tflite_runtime.interpreter import OpResolverType
        except ImportError:
            pass
    except ImportError:
        sys.exit(
            "No TFLite runtime found. Install one of:\n"
            "  pip install ai-edge-litert\n"
            "  pip install tflite-runtime"
        )

LABELS = ["metal", "plastic", "organic"]
DEFAULT_MODEL = "waste_classifier_float32_tf.tflite"
UNCERTAIN_TEXT = "Uncertain / Scanning..."

COLOR_OK = (60, 200, 60)
COLOR_UNCERTAIN = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)

CAPTURE_KEYS = {
    ord("m"): "metal",
    ord("p"): "plastic",
    ord("o"): "organic",
}


# ----------------------------------------------------------------------
# Shared center-crop helper
# ----------------------------------------------------------------------
def center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


# ----------------------------------------------------------------------
# Green conveyor detection
# ----------------------------------------------------------------------
def check_green_conveyor(frame_bgr: np.ndarray) -> float:
    """
    Return the fraction of center-crop pixels classified as green.

    HSV color detection is used so green detection is less sensitive
    to brightness variations than checking RGB values directly.
    """
    roi = center_crop(frame_bgr)
    small = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # OpenCV HSV hue range is 0-179.
    # Adjust these values if the conveyor's green differs significantly.
    lower_green = np.array([35, 60, 40], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    return float(np.count_nonzero(mask)) / mask.size


# ----------------------------------------------------------------------
# White / black blank-scene detection
# ----------------------------------------------------------------------
def check_blank_scene(
    frame_bgr: np.ndarray,
    white_level: int,
    black_level: int,
    blank_fraction: float,
) -> Tuple[Optional[str], float, float]:
    roi = center_crop(frame_bgr)
    small = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    white_frac = float(np.mean(gray >= white_level))
    black_frac = float(np.mean(gray <= black_level))

    if white_frac >= blank_fraction:
        return "white", white_frac, black_frac

    if black_frac >= blank_fraction:
        return "black", white_frac, black_frac

    return None, white_frac, black_frac


# ----------------------------------------------------------------------
# Camera sources
# ----------------------------------------------------------------------
class PiCamera2Source:
    def __init__(
        self,
        width: int,
        height: int,
        hflip: bool = False,
        vflip: bool = False,
    ):
        from libcamera import Transform
        from picamera2 import Picamera2

        self._cam = Picamera2()
        config = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            transform=Transform(hflip=hflip, vflip=vflip),
        )
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.5)

    def read(self) -> Optional[np.ndarray]:
        frame = self._cam.capture_array("main")
        if frame is None:
            return None
        return np.ascontiguousarray(frame[:, :, :3])

    def release(self) -> None:
        try:
            self._cam.stop()
        finally:
            self._cam.close()


class OpenCVSource:
    def __init__(self, index: int, width: int, height: int):
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = cv2.VideoCapture(index)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


def open_camera(
    backend: str,
    index: int,
    width: int,
    height: int,
    hflip: bool = False,
    vflip: bool = False,
):
    if backend in ("picamera2", "auto"):
        try:
            cam = PiCamera2Source(width, height, hflip, vflip)
            print("[camera] Using Picamera2")
            return cam
        except Exception as exc:
            if backend == "picamera2":
                raise RuntimeError(
                    f"Picamera2 failed to start: {exc}"
                ) from exc
            print(
                f"[camera] Picamera2 unavailable ({exc}); "
                "falling back to OpenCV"
            )

    cam = OpenCVSource(index, width, height)
    print(f"[camera] Using OpenCV (index {index})")
    return cam


# ----------------------------------------------------------------------
# TFLite classifier
# ----------------------------------------------------------------------
class WasteClassifier:
    def __init__(self, model_path: Path, num_threads: int):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.interpreter = Interpreter(
                model_path=str(model_path),
                num_threads=num_threads,
            )
            self.interpreter.allocate_tensors()
        except RuntimeError as exc:
            if OpResolverType is None:
                raise

            print(
                f"[model] Default delegate failed ({exc}); "
                "retrying without default delegates"
            )
            self.interpreter = Interpreter(
                model_path=str(model_path),
                num_threads=num_threads,
                experimental_op_resolver_type=(
                    OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
            self.interpreter.allocate_tensors()

        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]

        self._in_index = inp["index"]
        self._out_index = out["index"]
        self._in_dtype = np.dtype(inp["dtype"])
        self._out_dtype = np.dtype(out["dtype"])

        self._in_scale, self._in_zero_point = inp["quantization"]
        self._out_scale, self._out_zero_point = out["quantization"]

        shape = [int(d) for d in inp["shape"]]

        if len(shape) != 4:
            raise RuntimeError(f"Unexpected input shape {shape}")

        self._nchw = shape[1] == 3 and shape[3] != 3

        if self._nchw:
            self.height, self.width = shape[2], shape[3]
        else:
            self.height, self.width = shape[1], shape[2]

        num_outputs = int(out["shape"][-1])
        if num_outputs != len(LABELS):
            raise RuntimeError(
                f"Model has {num_outputs} outputs but "
                f"{len(LABELS)} labels are defined"
            )

        self._quantized_in = (
            np.issubdtype(self._in_dtype, np.integer)
            and self._in_scale > 0
        )
        self._quantized_out = (
            np.issubdtype(self._out_dtype, np.integer)
            and self._out_scale > 0
        )

        print(
            f"[model] {model_path.name}: "
            f"{'quantized input' if self._quantized_in else 'float input'}, "
            f"{'quantized output' if self._quantized_out else 'float output'}"
        )

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        crop = center_crop(frame_bgr)
        resized = cv2.resize(
            crop,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Model input is assumed to use RGB values scaled to [0, 1].
        x = rgb.astype(np.float32) / 255.0

        if self._quantized_in:
            info = np.iinfo(self._in_dtype)
            x = np.clip(
                np.rint(x / self._in_scale + self._in_zero_point),
                info.min,
                info.max,
            )

        x = x.astype(self._in_dtype)
        x = x[np.newaxis, ...]

        if self._nchw:
            x = np.transpose(x, (0, 3, 1, 2))

        return np.ascontiguousarray(x)

    def _dequantize_output(self, raw: np.ndarray) -> np.ndarray:
        raw = raw.astype(np.float32)

        if self._quantized_out:
            raw = (
                raw - self._out_zero_point
            ) * self._out_scale

        return raw

    def predict(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        tensor = self.preprocess(frame_bgr)

        t0 = time.perf_counter()
        self.interpreter.set_tensor(self._in_index, tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self._out_index)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = self._dequantize_output(raw)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        return probs, latency_ms

    def warmup(self, runs: int = 3) -> None:
        dummy = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )
        for _ in range(runs):
            self.predict(dummy)

    def benchmark(self, runs: int = 50) -> float:
        dummy = np.random.randint(
            0, 256,
            (self.height, self.width, 3),
            dtype=np.uint8,
        )
        self.warmup()
        times = [self.predict(dummy)[1] for _ in range(runs)]
        return float(np.mean(times))


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    text: str,
    color: Tuple[int, int, int],
    info: str,
    status: str = "",
) -> None:
    h, w = frame.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2

    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + side - 1, y0 + side - 1),
        color,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2

    (tw, th), base = cv2.getTextSize(text, font, scale, thick)

    cv2.rectangle(
        frame,
        (0, 0),
        (min(w - 1, tw + 20), th + base + 20),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame, text, (10, th + 10),
        font, scale, color, thick, cv2.LINE_AA,
    )

    cv2.putText(
        frame, info, (10, h - 12),
        font, 0.55, COLOR_TEXT, 1, cv2.LINE_AA,
    )

    if status:
        cv2.putText(
            frame, status, (10, th + base + 45),
            font, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )


def save_capture(
    frame: np.ndarray,
    capture_dir: Path,
    label: str,
) -> str:
    folder = capture_dir / label
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{label}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(
        str(path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    count = sum(1 for _ in folder.glob("*.jpg"))
    return f"saved {label} ({count} total)"


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi real-time waste classifier."
    )

    parser.add_argument(
        "--model", type=Path, default=Path(DEFAULT_MODEL)
    )
    parser.add_argument(
        "--camera",
        choices=["auto", "picamera2", "opencv"],
        default="auto",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--vflip", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--capture-dir", type=Path, default=Path("captures")
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    # White/black blank detection.
    parser.add_argument("--no-blank-check", action="store_true")
    parser.add_argument("--white-level", type=int, default=200)
    parser.add_argument("--black-level", type=int, default=40)
    parser.add_argument("--blank-fraction", type=float, default=0.95)

    # Green conveyor detection.
    parser.add_argument(
        "--no-green-check",
        action="store_true",
        help="Disable green conveyor detection.",
    )
    parser.add_argument(
        "--green-fraction",
        type=float,
        default=0.95,
        help="Green pixel fraction required to skip classification.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    classifier = WasteClassifier(args.model, args.threads)

    if args.benchmark:
        print(
            f"[benchmark] {args.model.name}: "
            f"{classifier.benchmark():.1f} ms/inference "
            f"({args.threads} threads)"
        )
        return

    classifier.warmup()

    camera = open_camera(
        args.camera,
        args.camera_index,
        args.width,
        args.height,
        args.hflip,
        args.vflip,
    )

    window = "Waste Classifier (press 'q' to quit)"
    ema_latency: Optional[float] = None
    last_print = 0.0
    failed_reads = 0
    status = ""
    status_until = 0.0

    try:
        while True:
            frame = camera.read()

            if frame is None:
                failed_reads += 1
                if failed_reads >= 30:
                    print(
                        "[camera] Too many failed reads, exiting.",
                        file=sys.stderr,
                    )
                    break
                time.sleep(0.03)
                continue

            failed_reads = 0

            # 1. Check for a fully green conveyor FIRST.
            green_frac = 0.0

            if not args.no_green_check:
                green_frac = check_green_conveyor(frame)

            if green_frac >= args.green_fraction:
                text = "CONVEYOR - NO ITEM"
                color = COLOR_OK
                info = (
                    f"green {green_frac * 100:.0f}% - "
                    "classification skipped"
                )

            else:
                # 2. Existing white/black blank-scene check.
                blank_kind: Optional[str] = None
                white_frac = black_frac = 0.0

                if not args.no_blank_check:
                    blank_kind, white_frac, black_frac = (
                        check_blank_scene(
                            frame,
                            args.white_level,
                            args.black_level,
                            args.blank_fraction,
                        )
                    )

                if blank_kind is not None:
                    text = UNCERTAIN_TEXT
                    color = COLOR_UNCERTAIN
                    info = (
                        f"blank {blank_kind} background "
                        f"(white {white_frac * 100:.0f}% / "
                        f"black {black_frac * 100:.0f}%) - no item"
                    )

                else:
                    # 3. Run waste classification only when needed.
                    probs, latency_ms = classifier.predict(frame)

                    ema_latency = (
                        latency_ms
                        if ema_latency is None
                        else 0.9 * ema_latency + 0.1 * latency_ms
                    )

                    best = int(np.argmax(probs))
                    confidence = float(probs[best])

                    if confidence >= args.threshold:
                        text = (
                            f"{LABELS[best].upper()} "
                            f"{confidence * 100:.1f}%"
                        )
                        color = COLOR_OK
                    else:
                        text = UNCERTAIN_TEXT
                        color = COLOR_UNCERTAIN

                    info = (
                        f"inference {latency_ms:.1f} ms "
                        f"(avg {ema_latency:.1f} ms) | "
                        + " ".join(
                            f"{name}:{p * 100:.0f}%"
                            for name, p in zip(LABELS, probs)
                        )
                    )

            if args.headless:
                now = time.time()

                if now - last_print >= 0.5:
                    print(f"{text:<26} | {info}")
                    last_print = now

            else:
                display = frame.copy()

                shown_status = (
                    status if time.time() < status_until else ""
                )

                draw_overlay(
                    display, text, color, info, shown_status
                )
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key in CAPTURE_KEYS:
                    status = save_capture(
                        frame,
                        args.capture_dir,
                        CAPTURE_KEYS[key],
                    )
                    status_until = time.time() + 1.5
                    print(f"[capture] {status}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        camera.release()

        if not args.headless:
            cv2.destroyAllWindows()

        print("Resources released. Bye.")


if __name__ == "__main__":
    main()
```
