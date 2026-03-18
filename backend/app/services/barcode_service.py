import cv2
import numpy as np

# Optional imports
HAS_PYZBAR = False
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except Exception:
    pass

# Helper to convert to gray
def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

PREPROCESS_PIPELINE = [
    ("Original",          lambda img: img),
    ("Grayscale",         lambda img: to_gray(img)),
    ("Sharpened",         lambda img: cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))),
    ("CLAHE",             lambda img: cv2.createCLAHE(3.0, (8,8)).apply(to_gray(img))),
    ("AdaptiveThreshold", lambda img: cv2.adaptiveThreshold(to_gray(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)),
    ("OtsuThreshold",     lambda img: cv2.threshold(cv2.GaussianBlur(to_gray(img),(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]),
    ("Morphology",        lambda img: cv2.morphologyEx(to_gray(img), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))),
    ("Inverted",          lambda img: cv2.bitwise_not(to_gray(img))),
]

def detect_pyzbar(img):
    if not HAS_PYZBAR:
        return []
    results = []
    try:
        for obj in pyzbar_decode(img):
            results.append({
                "data": obj.data.decode("utf-8", errors="replace"),
                "type": obj.type,
                "rect": {"x": obj.rect.left, "y": obj.rect.top,
                          "w": obj.rect.width, "h": obj.rect.height},
                "engine": "pyzbar"
            })
    except Exception:
        pass
    return results

def detect_opencv_qr(img):
    results = []
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    if data:
        rect = {}
        if bbox is not None:
            pts = bbox[0]
            x1, y1 = int(min(p[0] for p in pts)), int(min(p[1] for p in pts))
            x2, y2 = int(max(p[0] for p in pts)), int(max(p[1] for p in pts))
            rect = {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1}
        results.append({"data": data, "type": "QRCODE", "rect": rect, "engine": "opencv_qr"})
    try:
        det2 = cv2.QRCodeDetectorAruco()
        ok, info, pts, _ = det2.detectAndDecodeMulti(img)
        if ok and info:
            existing = {r["data"] for r in results}
            for d in info:
                if d and d not in existing:
                    results.append({"data": d, "type": "QRCODE", "rect": {}, "engine": "opencv_qr_aruco"})
    except AttributeError:
        pass
    return results

def detect_opencv_barcode(img):
    results = []
    try:
        det = cv2.barcode.BarcodeDetector()
        retvals = det.detectAndDecode(img)
        if len(retvals) == 4:
            ok, info, btype, pts = retvals
        elif len(retvals) == 3:
            ok, info, btype = retvals
        else:
            return results
        if ok and info:
            if isinstance(info, str):
                info, btype = [info], [btype]
            for d, t in zip(info, btype):
                if d:
                    results.append({"data": d, "type": t or "BARCODE", "rect": {}, "engine": "opencv_barcode"})
    except (AttributeError, cv2.error, Exception):
        pass
    return results

ALL_DETECTORS = [
    ("pyzbar", detect_pyzbar),
    ("opencv_qr", detect_opencv_qr),
    ("opencv_barcode", detect_opencv_barcode),
]

def sliding_window_scan(img, window_size=400, overlap=0.3):
    results = []
    h, w = img.shape[:2]
    step = int(window_size * (1 - overlap))
    for y in range(0, h - window_size // 2, step):
        for x in range(0, w - window_size // 2, step):
            crop = img[y:min(y+window_size, h), x:min(x+window_size, w)]
            for _, dfn in ALL_DETECTORS:
                for r in dfn(crop):
                    if r.get("rect"):
                        r["rect"]["x"] = r["rect"].get("x", 0) + x
                        r["rect"]["y"] = r["rect"].get("y", 0) + y
                    r["strategy"] = f"SlidingWindow_{window_size}"
                    results.append(r)
    return results

def rotation_scan(img):
    results = []
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [90, 180, 270, 45, 135]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        for _, dfn in ALL_DETECTORS:
            for r in dfn(rotated):
                r["strategy"] = f"Rotated_{angle}°"
                results.append(r)
    return results

def dedup(results):
    seen = set()
    unique = []
    for r in results:
        key = (r["data"], r["type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

def full_scan(img_cv):
    """Run full 4-stage barcode scan pipeline."""
    all_results = []

    # Stage 1: Multi-engine x Multi-preprocess
    for _, prep_fn in PREPROCESS_PIPELINE:
        try:
            processed = prep_fn(img_cv)
        except Exception:
            continue
        for _, det_fn in ALL_DETECTORS:
            try:
                found = det_fn(processed)
                for r in found:
                    r.setdefault("strategy", "Direct")
                all_results.extend(found)
            except Exception:
                pass

    # Stage 2: Multi-scale
    if not all_results:
        for scale in [1.5, 2.0, 2.5, 3.0]:
            scaled = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for _, dfn in ALL_DETECTORS:
                found = dfn(scaled)
                for r in found:
                    r["strategy"] = f"Scale_x{scale}"
                all_results.extend(found)
            if all_results:
                break

    # Stage 3: Sliding window
    if not all_results:
        for ws in [300, 500]:
            found = sliding_window_scan(img_cv, ws, 0.3)
            if found:
                all_results.extend(found)
                break

    # Stage 4: Rotation
    if not all_results:
        all_results.extend(rotation_scan(img_cv))

    return dedup(all_results)

def scan_file_for_barcodes(file_path: str):
    """Helper to detect barcodes from a given file path."""
    try:
        img_array = cv2.imread(file_path)
        if img_array is None:
            return []
        return full_scan(img_array)
    except Exception as e:
        print(f"Barcode scanning failed: {e}")
        return []
