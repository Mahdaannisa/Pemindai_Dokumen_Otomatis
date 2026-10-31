import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io

st.set_page_config(
    page_title="Pemindai Dokumen Otomatis",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Pemindai Dokumen Otomatis")
st.write(
    "Unggah foto dokumen. Sistem akan mendeteksi empat sudut, "
    "meluruskan dokumen, lalu menghasilkan citra hitam-putih."
)

def read_image(path):
    """Baca gambar yang juga mendukung nama file dengan karakter khusus."""
    path = Path(path)
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def resize_for_detection(image, max_side=1100):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
    return resized, scale


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)],
                     pts[np.argmax(sums)], pts[np.argmax(diffs)]], dtype=np.float32)


# Deteksi 4 sudut
A4_RATIO = 297 / 210
def _sudut_internal_quad(pts):
    """Hitung 4 sudut internal quad (derajat). pts harus sudah urut."""
    angles = []
    for i in range(4):
        p0, p1, p2 = pts[(i-1) % 4], pts[i], pts[(i+1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    return np.array(angles)

def _skor_quad_v2(corners_small, area_ratio, image_w, image_h):
    """
    Skor berbasis BENTUK, bukan rasio A4.
    Cocok untuk foto miring karena proyeksi perspektif mengubah rasio.
    """
    pts = corners_small.reshape(4, 2).astype(np.float32)

    # 1. Harus convex
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.int32)):
        return -1

    # 2. Sudut internal wajar (hindari quad degenerasi / garis)
    angles = _sudut_internal_quad(pts)
    if np.any(angles < 45) or np.any(angles > 135):
        return -1
    angle_score = np.clip(1 - np.std(angles) / 45, 0, 1)

    # 3. Solidity: seberapa "rectangle" quad ini
    quad_area = cv2.contourArea(pts.reshape(-1, 1, 2))
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    rect_area = rect[1][0] * rect[1][1]
    if rect_area < 1:
        return -1
    solidity = quad_area / rect_area  # 1.0 = rectangle sempurna

    # 4. Area: tidak terlalu kecil / tidak menutupi seluruh foto
    if area_ratio < 0.05:
        return -1
    if area_ratio > 0.95:
        area_score = 0.3   # kemungkinan frame foto, bukan kertas
    elif area_ratio > 0.85:
        area_score = 0.7
    else:
        area_score = 1.0

    # 5. Penalti jika ≥3 titik menempel tepi frame
    m = 2
    n_touch = sum(1 for x, y in pts
                  if x <= m or x >= image_w - m - 1
                  or y <= m or y >= image_h - m - 1)
    border_penalty = 0.7 if n_touch >= 3 else 1.0

    return (0.40 * solidity + 0.30 * angle_score + 0.30 * area_score) * border_penalty


def _preprocess_variant(image, variant="clahe_normal"):
    """Grayscale -> peningkatan kontras opsional -> Gaussian blur -> Canny -> close."""
    if image is None:
        raise ValueError("Gambar tidak berhasil dibaca.")
    resized, scale = resize_for_detection(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    if variant == "clahe_strong":
        gray_eq = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    elif variant == "clahe_tile_small":
        gray_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    elif variant == "bilateral":
        gray_eq = cv2.bilateralFilter(gray, 9, 75, 75)
    elif variant == "tophat":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        gray_eq = cv2.normalize(cv2.subtract(cv2.add(gray, top_hat), black_hat), None, 0, 255, cv2.NORM_MINMAX)
    else:
        gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    if variant == "adaptive_canny":
        # Adaptive threshold membantu saat kontras tepi rendah; Canny tetap dipakai.
        binary = cv2.adaptiveThreshold(
            gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 10
        )
        edges = cv2.Canny(binary, 50, 150)
    else:
        # Canny dengan ambang otomatis yang dibatasi agar tidak terlalu ekstrem.
        median = float(np.median(blurred))
        low = int(max(20, 0.66 * median))
        high = int(min(240, max(low + 30, 1.33 * median)))
        edges = cv2.Canny(blurred, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return resized, scale, gray_eq, blurred, edges, edges_closed

def _skor_quad(corners_small, area_ratio, image_w=None, image_h=None):
    """
    Skor LUNAK. Tolak:
    - Quad degenerasi (sisi < 5 px)
    - Quad ≥2 titik di frame (kertas menyatu dengan frame)
    - Quad area > 0.92 (kemungkinan frame/meja, bukan kertas)
    Beri skor rendah (tapi tidak -1) untuk area 0.85–0.92.
    """
    pts = corners_small.reshape(4, 2)
    w_top    = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    h_left   = np.linalg.norm(pts[3] - pts[0])
    h_right  = np.linalg.norm(pts[2] - pts[1])
    w = (w_top + w_bottom) / 2
    h = (h_left + h_right) / 2
    if w < 5 or h < 5:
        return -1

    # HARD REJECT: area > 0.92 = kemungkinan frame/meja
    # Kertas jarang sekali >92% frame, dan kalau memang iya, itu tanda
    # kontur salah (frame atau bayangan di tepi). Lebih baik fallback.
    if area_ratio > 0.92:
        return -1

    # HARD REJECT: ≥2 titik di frame = kertas menyatu frame
    if image_w is not None and image_h is not None:
        margin = 0.02
        x_min = margin * image_w
        x_max = (1 - margin) * image_w
        y_min = margin * image_h
        y_max = (1 - margin) * image_h
        n_touch = sum(
            1 for x, y in pts
            if x <= x_min or x >= x_max or y <= y_min or y >= y_max
        )
        if n_touch >= 2:
            return -1

    ratio = max(w, h) / min(w, h)
    rasio_score = max(0.0, 1.0 - abs(ratio - A4_RATIO) / A4_RATIO)

    if area_ratio < 0.05:
        area_score = 0.0
    elif area_ratio < 0.30:
        area_score = area_ratio / 0.30
    elif area_ratio <= 0.85:
        area_score = 1.0
    else:
        # 0.85–0.92: transisi turun
        area_score = 1.0 - (area_ratio - 0.85) / 0.07 * 0.7

    return 0.55 * rasio_score + 0.45 * area_score

def _paksa_quad(contour):
    """
    Kalau approxPolyDP biasa gagal dapat 4 titik,
    coba convex hull + approxPolyDP berulang.
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            return ap.reshape(4, 2).astype(np.float32)
    return None

def _kumpulkan_kandidat(edges_closed, image_area,
                        min_area_ratio=0.02, max_area_ratio=0.95,
                        epsilon_values=(0.01, 0.015, 0.02, 0.025, 0.03,
                                        0.04, 0.05, 0.06, 0.08)):
    """Kumpulkan kandidat quad dari kontur Canny; jangan otomatis menganggap frame foto sebagai kertas."""
    h, w = edges_closed.shape[:2]
    pad = max(8, int(min(h, w) * 0.015))
    padded = cv2.copyMakeBorder(edges_closed, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape[:2]
    padded_area = ph * pw
    candidates = []

    for mode in (cv2.RETR_LIST, cv2.RETR_EXTERNAL):
        contours, _ = cv2.findContours(padded.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:80]:
            area = cv2.contourArea(contour)
            ratio_padded = area / padded_area
            if ratio_padded < min_area_ratio or ratio_padded > max_area_ratio:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 80:
                continue

            quads = []
            for eps in epsilon_values:
                approx = cv2.approxPolyDP(contour, eps * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quads.append((approx.reshape(4, 2).astype(np.float32), eps))
            # Convex hull membantu bila kontur sedikit bergerigi/terputus.
            if not quads:
                hull = cv2.convexHull(contour)
                hp = cv2.arcLength(hull, True)
                for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
                    approx = cv2.approxPolyDP(hull, eps * hp, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        quads.append((approx.reshape(4, 2).astype(np.float32), f"hull-{eps}"))
                        break

            for pts_pad, eps_used in quads:
                pts = pts_pad.copy()
                pts[:, 0] -= pad
                pts[:, 1] -= pad
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                pts = order_points(pts)
                quad_area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
                area_ratio = quad_area / image_area
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue

                sides = np.array([
                    np.linalg.norm(pts[1]-pts[0]),
                    np.linalg.norm(pts[2]-pts[1]),
                    np.linalg.norm(pts[3]-pts[2]),
                    np.linalg.norm(pts[0]-pts[3])
                ])
                if sides.min() < 12:
                    continue

                angles = _sudut_internal_quad(pts)
                if np.any(angles < 25) or np.any(angles > 155):
                    continue

                # Hindari kontur frame: penalti berdasarkan titik yang sangat dekat tepi.
                margin_x, margin_y = max(3, int(w*0.008)), max(3, int(h*0.008))
                touches = sum(
                    (x <= margin_x or x >= w-1-margin_x or
                     y <= margin_y or y >= h-1-margin_y)
                    for x, y in pts
                )
                border_score = 1.0 if touches == 0 else (0.65 if touches == 1 else 0.15)

                # Bentuk quadrilateral dan area menjadi skor utama; rasio A4 bukan syarat keras
                # karena perspektif dan jenis dokumen dapat mengubah proporsi yang terlihat.
                rect = cv2.minAreaRect(pts.reshape(-1, 1, 2).astype(np.float32))
                rect_area = max(rect[1][0] * rect[1][1], 1.0)
                solidity = min(1.0, quad_area / rect_area)
                area_score = min(1.0, area_ratio / 0.30)
                angle_score = max(0.0, 1.0 - np.std(angles) / 55.0)
                score = (0.35*solidity + 0.30*angle_score + 0.35*area_score) * border_score

                candidates.append({
                    "skor": float(score), "corners": pts, "contour": contour,
                    "epsilon": eps_used, "area": quad_area,
                    "area_ratio": area_ratio
                })

    # Hilangkan kandidat duplikat yang hampir sama, lalu urutkan berdasarkan skor.
    candidates.sort(key=lambda item: item["skor"], reverse=True)
    unique = []
    for item in candidates:
        if not any(np.mean(np.linalg.norm(item["corners"] - old["corners"], axis=1)) < 12
                   for old in unique):
            unique.append(item)
        if len(unique) >= 20:
            break
    return unique

def detect_document_corners(image, min_area_ratio=0.02, max_area_ratio=0.95,
                            epsilon_values=(0.01, 0.015, 0.02, 0.025, 0.03,
                                            0.04, 0.05, 0.06, 0.08)):
    """Deteksi otomatis 4 sudut menggunakan preprocessing + Canny + contours + approxPolyDP."""
    if image is None:
        raise ValueError("Gambar tidak berhasil dibaca.")

    variants = ["clahe_normal", "clahe_strong", "clahe_tile_small",
                "bilateral", "tophat", "adaptive_canny"]
    all_candidates = []
    last_state = None

    for variant in variants:
        resized, scale, gray_eq, blurred, edges, edges_closed = _preprocess_variant(image, variant)
        last_state = (resized, scale, gray_eq, blurred, edges, edges_closed)
        image_area = resized.shape[0] * resized.shape[1]

        candidates = _kumpulkan_kandidat(
            edges_closed, image_area, min_area_ratio, max_area_ratio, epsilon_values
        )
        for candidate in candidates:
            candidate = candidate.copy()
            candidate["variant"] = variant
            candidate["corners_small"] = candidate["corners"].copy()
            candidate["corners"] = candidate["corners"] / scale
            candidate.update({
                "resized": resized, "gray": gray_eq, "blurred": blurred,
                "edges": edges, "edges_closed": edges_closed
            })
            all_candidates.append(candidate)

    if all_candidates:
        # Variasi preprocessing dapat menemukan kontur yang sama; pilih skor terbaik.
        all_candidates.sort(key=lambda item: item["skor"], reverse=True)
        best = all_candidates[0]
        if best["skor"] >= 0.38:
            best["epsilon_used"] = best["epsilon"]
            return best

    resized, scale, gray_eq, blurred, edges, edges_closed = last_state
    return {
        "corners": None, "corners_small": None, "resized": resized,
        "gray": gray_eq, "blurred": blurred, "edges": edges,
        "edges_closed": edges_closed, "contour": None,
        "epsilon_used": None, "area_ratio": None, "variant": None, "skor": -1
    }



# Warp + Threshold + Pipeline

def warp_kertas(image_bgr, points, lebar_output=1000):
    pts = order_points(points)

    # Validasi: quad harus convex & sudut wajar
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.int32)):
        return None
    angles = _sudut_internal_quad(pts)
    if np.any(angles < 40) or np.any(angles > 140):
        return None

    # Estimasi lebar & tinggi quad sumber
    w = (np.linalg.norm(pts[1]-pts[0]) + np.linalg.norm(pts[2]-pts[3])) / 2
    h = (np.linalg.norm(pts[3]-pts[0]) + np.linalg.norm(pts[2]-pts[1])) / 2
    if w < 10 or h < 10:
        return None

    # Orientasi: portrait default, landscape hanya jika jelas lebih lebar
    is_landscape = w > h * 1.15
    lebar = int(lebar_output)
    tinggi = int(round(lebar / A4_RATIO)) if is_landscape else int(round(lebar * A4_RATIO))

    dst = np.array([[0, 0], [lebar-1, 0],
                    [lebar-1, tinggi-1], [0, tinggi-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(image_bgr, M, (lebar, tinggi),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # Sanity check: tolak hasil warp yang kosong/rusak
    if warped.std() < 5:
        return None
    return warped

def threshold_otsu(gray):
    """Threshold global Otsu."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def threshold_adaptive_mean(gray, block_size=31, C=10):
    """Adaptive threshold dengan MEAN."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, C)


def threshold_adaptive_gaussian(gray, block_size=31, C=10):
    """Adaptive threshold dengan GAUSSIAN."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)


def threshold_camscanner(gray):
    """
    'Mode CamScanner': bilateral (halus tapi tepi teks tetap tajam)
    + adaptive threshold + pembersihan noise kecil.
    """
    # 1. Bilateral: mengurangi noise/toner fotokopi, teks tetap tajam
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # 2. CLAHE ringan: atasi bayangan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    smooth = clahe.apply(smooth)
    # 3. Adaptive threshold
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 10)
    # 4. Bersihkan bintik kecil
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # 5. Tutup celah kecil di teks
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # 6. Median blur halus (opsional, bikin tepi teks lebih rapi)
    binary = cv2.medianBlur(binary, 3)
    return binary


# =========================
# Antarmuka Streamlit
# =========================

uploaded_file = st.file_uploader(
    "Pilih gambar dokumen",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

metode = st.selectbox(
    "Metode thresholding",
    options=["camscanner", "adaptive_gaussian", "adaptive_mean", "otsu"],
    format_func=lambda x: {
        "camscanner": "CamScanner (Adaptive Gaussian + pembersihan noise)",
        "adaptive_gaussian": "Adaptive Gaussian",
        "adaptive_mean": "Adaptive Mean",
        "otsu": "Otsu"
    }[x]
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("File tidak dapat dibaca sebagai gambar. Coba unggah gambar lain.")
    else:
        with st.spinner("Mendeteksi dan memproses dokumen..."):
            detection = detect_document_corners(image_bgr)

            if detection["corners"] is None:
                warped = None
                result = None
            else:
                warped = warp_kertas(
                    image_bgr,
                    detection["corners"],
                    lebar_output=1000
                )

                if warped is not None:
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    if metode == "otsu":
                        result = threshold_otsu(gray)
                    elif metode == "adaptive_mean":
                        result = threshold_adaptive_mean(gray)
                    elif metode == "adaptive_gaussian":
                        result = threshold_adaptive_gaussian(gray)
                    else:
                        result = threshold_camscanner(gray)
                else:
                    result = None

        st.divider()

        if result is None:
            st.error(
                "Empat sudut dokumen belum berhasil ditemukan atau proses "
                "pelurusan gagal. Coba gambar yang memperlihatkan seluruh "
                "sisi kertas dengan lebih jelas."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gambar asli")
                st.image(
                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            with col2:
                st.subheader("Titik deteksi")
                preview = image_bgr.copy()
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
        else:
            st.subheader("Perbandingan hasil")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Gambar asli**")
                st.image(
                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            with col2:
                st.markdown("**Hasil pemindaian**")
                st.image(result, clamp=True, use_container_width=True)

            with st.expander("Lihat hasil pelurusan berwarna"):
                st.image(
                    cv2.cvtColor(warped, cv2.COLOR_BGR2RGB),
                    caption="Dokumen setelah perspective transformation",
                    use_container_width=True
                )

            st.subheader("Informasi hasil")
            tinggi, lebar = result.shape[:2]
            info1, info2, info3 = st.columns(3)
            info1.metric("Status deteksi", "Berhasil")
            info2.metric("Lebar hasil", f"{lebar} px")
            info3.metric("Tinggi hasil", f"{tinggi} px")

            info4, info5 = st.columns(2)
            info4.metric("Skor kandidat", f"{detection['skor']:.3f}")
            info5.metric("Preprocessing", str(detection["variant"]))

            # Siapkan hasil PNG untuk diunduh.
            success, encoded = cv2.imencode(".png", result)
            if success:
                st.download_button(
                    label="⬇️ Download hasil pemindaian",
                    data=encoded.tobytes(),
                    file_name="hasil_pemindaian.png",
                    mime="image/png"
                )
else:
    st.info("Silakan unggah foto dokumen untuk memulai pemindaian.")

st.caption("Pemindai Dokumen Otomatis | UTS Praktikum Visi Komputer | Kelompok 03")
