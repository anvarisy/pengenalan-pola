from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = e


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "output"


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# A) LIST OPERATIONS (basic Python)
# -----------------------------------------------------------------------------
def demo_list_operations() -> None:
    header("A) DEMO LIST OPERATIONS")

    a = [3, 1, 4]
    b = [1, 5, 9]
    print("List a:", a)
    print("List b:", b)

    # 1) Gabungkan list
    c = a + b
    print("\n1) Gabungkan (a + b):", c)

    # 2) Ambil nilai tertentu
    print("2) Ambil nilai c[0]:", c[0])
    print("   Ambil nilai c[-1]:", c[-1])

    # 3) Sisipkan nilai
    c.insert(2, 99)
    print("\n3) Insert 99 pada index 2:", c)

    # 4) Hapus nilai (pop & remove)
    popped = c.pop()  # hapus elemen terakhir
    print("\n4) Pop elemen terakhir:", popped, "->", c)

    if 99 in c:
        c.remove(99)  # hapus elemen pertama yang bernilai 99
        print("   Remove nilai 99 ->", c)

    # 5) Urutkan
    c.sort()
    print("\n5) Sort ascending:", c)
    c.sort(reverse=True)
    print("   Sort descending:", c)


# -----------------------------------------------------------------------------
# B) LOAD CSV & TEXT DATA
# -----------------------------------------------------------------------------
def load_csv_data() -> None:
    header("B) LOAD CSV DATA (Data.csv ; delimiter)")

    csv_path = DATA_DIR / "Data.csv"
    if not csv_path.exists():
        print(f"[SKIP] File tidak ditemukan: {csv_path}")
        return

    df = pd.read_csv(csv_path, delimiter=";")
    print("Path:", csv_path)
    print("Shape:", df.shape)
    print("\nHead():")
    print(df.head(10))
    print("\nInfo():")
    print(df.info())


def load_text_data() -> None:
    header("C) LOAD TEXT DATA (Data.txt tab delimiter)")

    txt_path = DATA_DIR / "Data.txt"
    if not txt_path.exists():
        print(f"[SKIP] File tidak ditemukan: {txt_path}")
        return

    # Tab-delimited file
    df = pd.read_csv(txt_path, delimiter="\t")
    print("Path:", txt_path)
    print("Shape:", df.shape)
    print("\nHead():")
    print(df.head(10))


def load_data_from_url() -> None:
    header("D) LOAD DATA FROM URL (optional)")

    # Kalau dosen minta contoh baca data dari URL,
    # pakai URL CSV yang stabil dan gampang dicek.
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
        print("URL:", url)
        print("Shape:", df.shape)
        print("\nHead():")
        print(df.head(10))
    except Exception as e:
        print("[FAIL] Gagal baca dari URL:", e)


# -----------------------------------------------------------------------------
# E) PLOT WEB TRAFFIC (optional) - depends on web_traffic.tsv existing
# -----------------------------------------------------------------------------
def plot_web_traffic() -> None:
    header("E) PLOT WEB TRAFFIC (web_traffic.tsv)")

    tsv_path = DATA_DIR / "web_traffic.tsv"
    if not tsv_path.exists():
        print(f"[SKIP] File tidak ditemukan: {tsv_path}")
        print("       Jika modul kamu punya web_traffic.tsv, taruh di folder data/.")
        return

    # File biasanya: day \t hits
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["day", "hits"])
    df["hits"] = pd.to_numeric(df["hits"], errors="coerce")
    clean = df.dropna()

    print("Path:", tsv_path)
    print("Original rows:", len(df), "| Clean rows:", len(clean))
    print("Sample rows:")
    print(clean.head(10))

    ensure_out_dir()
    plt.figure()
    plt.scatter(clean["day"], clean["hits"])
    plt.title("Web Traffic")
    plt.xlabel("Day")
    plt.ylabel("Hits")
    out_path = OUT_DIR / "web_traffic_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Plot disimpan ke: {out_path}")


# -----------------------------------------------------------------------------
# F) IMAGE PROCESSING (OpenCV)
# -----------------------------------------------------------------------------
def require_cv2() -> bool:
    if cv2 is None:
        header("F) IMAGE PROCESSING (OpenCV)")
        print("[FAIL] OpenCV (cv2) tidak bisa diimport.")
        print("Error:", CV2_IMPORT_ERROR)
        print("Fix: pastikan kamu install opencv-python lewat requirements.txt")
        return False
    return True


def image_info_and_pixel_access() -> None:
    header("F1) IMAGE: 5.png (shape/size + pixel access)")

    if not require_cv2():
        return

    img_path = IMG_DIR / "5.png"
    if not img_path.exists():
        print(f"[SKIP] File tidak ditemukan: {img_path}")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # grayscale langsung
    if img is None:
        print("[FAIL] cv2.imread gagal membaca image.")
        return

    print("Path:", img_path)
    print("Shape (H,W):", img.shape)
    print("Size (total pixel):", img.size)

    # Pixel access (aman: ambil tengah gambar)
    h, w = img.shape
    y, x = h // 2, w // 2
    pixel_val = int(img[y, x])
    print(f"Pixel value at center [{y},{x}]:", pixel_val)

    ensure_out_dir()
    out_path = OUT_DIR / "five_grayscale.png"
    cv2.imwrite(str(out_path), img)
    print(f"[OK] Grayscale image disimpan ke: {out_path}")


def logo_to_grayscale_and_save() -> None:
    header("F2) IMAGE: logo_ipb.png (color -> grayscale)")

    if not require_cv2():
        return

    img_path = IMG_DIR / "logo_ipb.png"
    if not img_path.exists():
        print(f"[SKIP] File tidak ditemukan: {img_path}")
        return

    img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_color is None:
        print("[FAIL] cv2.imread gagal membaca image.")
        return

    print("Path:", img_path)
    print("Color shape (H,W,C):", img_color.shape)
    print("Color size (total values):", img_color.size)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    print("Gray shape (H,W):", img_gray.shape)
    print("Gray size (total pixel):", img_gray.size)

    # Pixel access contoh: titik (100,100) kalau ada
    y, x = 100, 100
    if y < img_gray.shape[0] and x < img_gray.shape[1]:
        print(f"Pixel value gray[{y},{x}]:", int(img_gray[y, x]))
    else:
        # fallback ke center
        h, w = img_gray.shape
        y2, x2 = h // 2, w // 2
        print(f"Pixel value gray center[{y2},{x2}]:", int(img_gray[y2, x2]))

    ensure_out_dir()
    out_gray = OUT_DIR / "logo_ipb_grayscale.png"
    cv2.imwrite(str(out_gray), img_gray)
    print(f"[OK] Grayscale logo disimpan ke: {out_gray}")


# -----------------------------------------------------------------------------
def main() -> int:
    print("Repo root:", ROOT)
    print("Python:", sys.version.replace("\n", " "))

    demo_list_operations()
    load_csv_data()
    load_text_data()
    load_data_from_url()
    plot_web_traffic()
    image_info_and_pixel_access()
    logo_to_grayscale_and_save()

    header("DONE")
    print("Jika ada bagian yang [SKIP], berarti file-nya belum ada di folder yang tepat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
