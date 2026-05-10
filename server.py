import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Flutter app-аас холбогдохыг зөвшөөрөх
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTO_DIR = Path(__file__).resolve().parent / "photos"
PHOTO_DIR.mkdir(parents=True, exist_ok=True)
NUMBA_CACHE_DIR = Path(__file__).resolve().parent / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

PRINT_DPI = 300
PRINT_PHOTO_WIDTH_CM = 3
PRINT_PHOTO_HEIGHT_CM = 4
# Printer/driver scaling calibration. Your printer currently enlarges a 3x4 cm
# generated photo to about 3.5x4.5 cm, so render smaller to get 3x4 cm on paper.
PRINT_RENDER_PHOTO_WIDTH_CM = float(os.environ.get("URJIN_RENDER_PHOTO_WIDTH_CM", "2.57"))
PRINT_RENDER_PHOTO_HEIGHT_CM = float(os.environ.get("URJIN_RENDER_PHOTO_HEIGHT_CM", "3.56"))
PRINT_SHEET_WIDTH_CM = 7
PRINT_SHEET_HEIGHT_CM = 10
PRINT_GAP_CM = 0
PRINT_CUT_LINE_WIDTH_PX = 1
HEAD_TARGET_HEIGHT_RATIO = 0.8
PRINT_PRINTER_NAME = os.environ.get("URJIN_PRINTER_NAME")
PRINT_MEDIA = os.environ.get("URJIN_PRINT_MEDIA", "Custom.70x100mm")
PRINT_INPUT_SLOT = os.environ.get("URJIN_PRINT_INPUT_SLOT", "Rear")
PRINT_MEDIA_TYPE = os.environ.get("URJIN_PRINT_MEDIA_TYPE", "PremiumGlossy")
PRINT_QUALITY = os.environ.get("URJIN_PRINT_QUALITY", "5")


def lp_print_args(filepath: Path) -> list[str]:
    args = ["lp"]
    if PRINT_PRINTER_NAME:
        args.extend(["-d", PRINT_PRINTER_NAME])

    # These CUPS option names are common for inkjet photo printers, but printer
    # drivers can expose different values. Override them with URJIN_PRINT_* envs.
    args.extend(
        [
            "-o",
            f"media={PRINT_MEDIA}",
            "-o",
            f"InputSlot={PRINT_INPUT_SLOT}",
            "-o",
            f"MediaType={PRINT_MEDIA_TYPE}",
            "-o",
            f"print-quality={PRINT_QUALITY}",
            "-o",
            "fit-to-page",
            str(filepath),
        ]
    )
    return args


def release_macos_camera_service() -> None:
    if sys.platform != "darwin":
        return

    # macOS often starts PTPCamera when a USB camera is attached. It claims the
    # PTP interface and prevents gphoto2 from capturing, so stop it before use.
    subprocess.run(
        ["killall", "PTPCamera"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def run_gphoto2(args: list[str]) -> subprocess.CompletedProcess[str]:
    release_macos_camera_service()

    try:
        return subprocess.run(
            ["gphoto2", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="gphoto2 is not installed or is not available in PATH",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Camera command timed out",
        ) from exc


def command_detail(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip() or "Camera command failed"


def capture_photo(filepath: Path) -> subprocess.CompletedProcess[str]:
    result = run_gphoto2(["--capture-image-and-download", f"--filename={filepath}"])
    if result.returncode == 0 and filepath.exists():
        return result

    fallback_errors = [command_detail(result)]
    unsupported_capture = (
        "Unsupported operation" in result.stderr
        or "does not support generic capture" in result.stderr
    )
    if not unsupported_capture:
        return result

    trigger_result = run_gphoto2(["--trigger-capture"])
    if trigger_result.returncode != 0:
        fallback_errors.append(command_detail(trigger_result))
        return subprocess.CompletedProcess(
            args=trigger_result.args,
            returncode=trigger_result.returncode,
            stdout=trigger_result.stdout,
            stderr="\n".join(fallback_errors),
        )

    download_result = run_gphoto2(
        ["--wait-event-and-download=15s", f"--filename={filepath}"]
    )
    if download_result.returncode == 0 and filepath.exists():
        return download_result

    fallback_errors.append(command_detail(download_result))
    return subprocess.CompletedProcess(
        args=download_result.args,
        returncode=download_result.returncode,
        stdout=download_result.stdout,
        stderr="\n".join(fallback_errors),
    )


def normalized_photo_id(photo_id: str) -> str:
    photo_name = Path(photo_id).name
    if photo_name.endswith("-bg-removed.png"):
        photo_name = photo_name[:-15]
    elif photo_name.endswith("-print.jpg"):
        photo_name = photo_name[:-10]
    elif photo_name.endswith("-print.pdf"):
        photo_name = photo_name[:-10]
    elif photo_name.endswith(".jpg"):
        photo_name = photo_name[:-4]

    try:
        uuid.UUID(photo_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid photo id") from exc

    return photo_name


def photo_response(photo_id: str) -> FileResponse:
    photo_name = normalized_photo_id(photo_id)
    filepath = PHOTO_DIR / f"{photo_name}.jpg"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(filepath, media_type="image/jpeg")


def background_removed_response(photo_id: str) -> FileResponse:
    photo_name = normalized_photo_id(photo_id)
    filepath = PHOTO_DIR / f"{photo_name}-bg-removed.png"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Background removed photo not found")
    return FileResponse(filepath, media_type="image/png")


def print_pdf_response(photo_id: str) -> FileResponse:
    photo_name = normalized_photo_id(photo_id)
    filepath = PHOTO_DIR / f"{photo_name}-print.pdf"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Print file not found")
    return FileResponse(filepath, media_type="application/pdf")


def send_print_job(photo_id: str) -> dict[str, str]:
    photo_name = normalized_photo_id(photo_id)
    filepath = PHOTO_DIR / f"{photo_name}-print.pdf"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Print file not found")

    try:
        result = subprocess.run(
            lp_print_args(filepath),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Printer command is not available. Install or enable CUPS/lp.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Printer command timed out") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Print failed"
        raise HTTPException(status_code=500, detail=detail)

    return {
        "id": photo_name,
        "message": result.stdout.strip() or "Print job submitted",
    }


def print_preview_response(photo_id: str) -> FileResponse:
    photo_name = normalized_photo_id(photo_id)
    filepath = PHOTO_DIR / f"{photo_name}-print.jpg"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Print preview not found")
    return FileResponse(filepath, media_type="image/jpeg")


def cm_to_px(cm: float) -> int:
    return round(cm / 2.54 * PRINT_DPI)


def center_crop_to_ratio(image, width_ratio: int, height_ratio: int):
    target_ratio = width_ratio / height_ratio
    image_ratio = image.width / image.height

    if image_ratio > target_ratio:
        new_width = round(image.height * target_ratio)
        left = (image.width - new_width) // 2
        return image.crop((left, 0, left + new_width, image.height))

    new_height = round(image.width / target_ratio)
    top = (image.height - new_height) // 2
    return image.crop((0, top, image.width, top + new_height))


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def detect_face_bounds(image) -> Optional[Tuple[int, int, int, int]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        return None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    return x, y, width, height


def face_aware_crop_to_ratio(image, width_ratio: int, height_ratio: int):
    crop_box, face_detected = face_aware_crop_box_to_ratio(image, width_ratio, height_ratio)
    return image.crop(crop_box), face_detected


def face_aware_crop_box_to_ratio(image, width_ratio: int, height_ratio: int):
    target_ratio = width_ratio / height_ratio
    face_bounds = detect_face_bounds(image)

    if face_bounds is None:
        cropped = center_crop_to_ratio(image, width_ratio, height_ratio)
        left = (image.width - cropped.width) // 2
        top = (image.height - cropped.height) // 2
        return (left, top, left + cropped.width, top + cropped.height), False

    face_x, face_y, face_width, face_height = face_bounds
    head_left = face_x - round(face_width * 0.25)
    head_top = face_y - round(face_height * 0.45)
    head_width = round(face_width * 1.5)
    head_height = round(face_height * 1.45)
    head_center_x = head_left + head_width // 2
    head_center_y = head_top + head_height // 2

    crop_height = round(head_height / HEAD_TARGET_HEIGHT_RATIO)
    crop_width = round(crop_height * target_ratio)

    if crop_width > image.width:
        crop_width = image.width
        crop_height = round(crop_width / target_ratio)
    if crop_height > image.height:
        crop_height = image.height
        crop_width = round(crop_height * target_ratio)

    left = clamp(head_center_x - crop_width // 2, 0, image.width - crop_width)
    # Place the estimated head near the top of the ID photo frame.
    top = clamp(head_top - round(crop_height * -0.1), 0, image.height - crop_height)

    return (left, top, left + crop_width, top + crop_height), True




def create_print_files(photo_id: str) -> tuple[Path, Path, bool, bool]:
    photo_name = normalized_photo_id(photo_id)
    background_removed_path = PHOTO_DIR / f"{photo_name}-bg-removed.png"
    original_path = PHOTO_DIR / f"{photo_name}.jpg"
    input_path = background_removed_path if background_removed_path.exists() else original_path
    preview_path = PHOTO_DIR / f"{photo_name}-print.jpg"
    pdf_path = PHOTO_DIR / f"{photo_name}-print.pdf"
    uses_background_removed = input_path == background_removed_path

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Print generation needs Pillow. Install it with: python3 -m pip install pillow",
        ) from exc

    photo_width = cm_to_px(PRINT_RENDER_PHOTO_WIDTH_CM)
    photo_height = cm_to_px(PRINT_RENDER_PHOTO_HEIGHT_CM)
    sheet_width = cm_to_px(PRINT_SHEET_WIDTH_CM)
    sheet_height = cm_to_px(PRINT_SHEET_HEIGHT_CM)
    gap = cm_to_px(PRINT_GAP_CM)

    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        detection_image = image.convert("RGB")
        crop_box, face_detected = face_aware_crop_box_to_ratio(detection_image, 3, 4)
        cropped = image.crop(crop_box)
        printed_photo = cropped.resize((photo_width, photo_height), Image.Resampling.LANCZOS)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    block_width = photo_width * 2 + gap
    block_height = photo_height * 2 + gap
    start_x = (sheet_width - block_width) // 2
    start_y = 120

    for row in range(2):
        for col in range(2):
            x = start_x + col * (photo_width + gap)
            y = start_y + row * (photo_height + gap)
            if printed_photo.mode == "RGBA":
                sheet.paste(printed_photo, (x, y), printed_photo)
            else:
                sheet.paste(printed_photo.convert("RGB"), (x, y))
            draw.rectangle(
                (x, y, x + photo_width - 1, y + photo_height - 1),
                outline="black",
                width=PRINT_CUT_LINE_WIDTH_PX,
            )

    sheet.save(preview_path, "JPEG", quality=95, dpi=(PRINT_DPI, PRINT_DPI))
    sheet.save(pdf_path, "PDF", resolution=PRINT_DPI)

    return preview_path, pdf_path, face_detected, uses_background_removed


def photo_item(photo_path: Path, request: Request) -> dict[str, Union[str, int]]:
    photo_id = photo_path.stem
    item = {
        "id": photo_id,
        "filename": photo_path.name,
        "url": str(request.url_for("get_photo_by_filename", photo_id=photo_id)),
        "size": photo_path.stat().st_size,
    }
    background_removed_path = PHOTO_DIR / f"{photo_id}-bg-removed.png"
    if background_removed_path.exists():
        item["background_removed_url"] = str(
            request.url_for("get_background_removed_by_filename", photo_id=photo_id)
        )
    print_path = PHOTO_DIR / f"{photo_id}-print.pdf"
    if print_path.exists():
        item["print_url"] = str(request.url_for("get_print_by_filename", photo_id=photo_id))
        item["print_preview_url"] = str(
            request.url_for("get_print_preview_by_filename", photo_id=photo_id)
        )
    return item

@app.get("/status")
def status():
    result = run_gphoto2(["--auto-detect"])
    connected = result.returncode == 0 and "Canon" in result.stdout
    return {"connected": connected}

@app.post("/capture")
def capture():
    photo_id = str(uuid.uuid4())
    filename = f"{photo_id}.jpg"
    filepath = PHOTO_DIR / filename

    result = capture_photo(filepath)

    if result.returncode != 0:
        detail = command_detail(result)
        raise HTTPException(status_code=500, detail=detail)

    if not filepath.exists():
        detail = command_detail(result)
        raise HTTPException(status_code=500, detail=detail)

    return {"id": photo_id, "filename": filename, "path": str(filepath)}

@app.get("/photos")
def get_photos(request: Request):
    photos = sorted(
        (
            photo_path
            for photo_path in PHOTO_DIR.glob("*.jpg")
            if "-print" not in photo_path.stem
        ),
        key=lambda photo_path: photo_path.stat().st_mtime,
        reverse=True,
    )
    return {"photos": [photo_item(photo_path, request) for photo_path in photos]}

@app.post("/photo/{photo_id}/remove-background")
def remove_background(photo_id: str, request: Request):
    photo_name = normalized_photo_id(photo_id)
    input_path = PHOTO_DIR / f"{photo_name}.jpg"
    output_path = PHOTO_DIR / f"{photo_name}-bg-removed.png"

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")

    try:
        from rembg import remove
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Background removal needs rembg. Install it with: python3 -m pip install rembg",
        ) from exc

    output_path.write_bytes(remove(input_path.read_bytes()))

    return {
        "id": photo_name,
        "filename": output_path.name,
        "url": str(request.url_for("get_background_removed_by_filename", photo_id=photo_name)),
        "path": str(output_path),
    }

@app.post("/photo/{photo_id}/make-print")
def make_print(photo_id: str, request: Request):
    photo_name = normalized_photo_id(photo_id)
    preview_path, pdf_path, face_detected, uses_background_removed = create_print_files(photo_name)

    return {
        "id": photo_name,
        "filename": pdf_path.name,
        "url": str(request.url_for("get_print_by_filename", photo_id=photo_name)),
        "preview_filename": preview_path.name,
        "preview_url": str(request.url_for("get_print_preview_by_filename", photo_id=photo_name)),
        "path": str(pdf_path),
        "preview_path": str(preview_path),
        "photo_width_cm": PRINT_PHOTO_WIDTH_CM,
        "photo_height_cm": PRINT_PHOTO_HEIGHT_CM,
        "copies": 4,
        "dpi": PRINT_DPI,
        "face_detected": face_detected,
        "uses_background_removed": uses_background_removed,
    }

@app.post("/photo/{photo_id}/print")
def print_photo(photo_id: str):
    return send_print_job(photo_id)

@app.get("/photo/{photo_id}")
def get_photo(photo_id: str):
    return photo_response(photo_id)

@app.get("/photo/{photo_id}/background-removed")
def get_background_removed(photo_id: str):
    return background_removed_response(photo_id)

@app.get("/photo/{photo_id}/print")
def get_print(photo_id: str):
    return print_pdf_response(photo_id)

@app.get("/photo/{photo_id}/print-preview")
def get_print_preview(photo_id: str):
    return print_preview_response(photo_id)

@app.get("/{photo_id}-bg-removed.png")
def get_background_removed_by_filename(photo_id: str):
    return background_removed_response(photo_id)

@app.get("/{photo_id}-print.pdf")
def get_print_by_filename(photo_id: str):
    return print_pdf_response(photo_id)

@app.get("/{photo_id}-print.jpg")
def get_print_preview_by_filename(photo_id: str):
    return print_preview_response(photo_id)

@app.get("/{photo_id}.jpg")
def get_photo_by_filename(photo_id: str):
    return photo_response(photo_id)
