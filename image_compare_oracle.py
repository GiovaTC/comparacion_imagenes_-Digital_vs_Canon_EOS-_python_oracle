import os
import sys
import io
import csv
from datetime import datetime
from dotenv import load_dotenv
import oracledb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

#cargar .env
load_dotenv()

ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASS = os.getenv("ORACLE_PASS")
ORACLE_DSN = os.getenv("ORACLE_DSN") # por ejemplo :"localhost:puerto/orclpdb1"
POOL_MIN = int(os.getenv("ORACLE_POOL_MIN", "1"))
POOL_MAX = int(os.getenv("ORACLE_POOL_MAX", "4"))

#rutas por defecto ( puedes cambiar ) .
DIGITAL_IMAGE_PATH = os.getenv("DIGITAL_IMAGE_PATH", "digital_image.png")
CANON_IMAGE_PATH = os.getenv("CANON_IMAGE_PATH", "canon_eos.jpg")

BACKUP_DIR = "backup"
METADATA_CSV = "scores.csv"

def ensure_backup_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)

def save_backup(path, name):
    ensure_backup_dir()
    dest = os.path.join(BACKUP_DIR, name)
    with open(path, "rb") as rf, open(dest, "wb") as wf:
        wf.write(rf.read())
    return dest 

def append_metadata_csv(row):
    header = ["timestamp","nombre","descripcion","formato","localpath"]
    exists = os.path.exists(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

def connect_oracle():
    #intentamos crear un pool si hay credenciales . :
    if not ORACLE_USER or not ORACLE_PASS or not ORACLE_DSN:
        raise ValueError("faltan credenciales de oracle en variables de entorno .")
    pool = oracledb.create_pool(user=ORACLE_USER, password=ORACLE_PASS, dsn=ORACLE_DSN,
                                min=POOL_MIN, max=POOL_MAX, increment=1, encoding="utf-8")
    return pool

def insert_image(pool, nombre, descripcion, formato, image_bytes):
    conn = pool.acquire()
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO imagenes (nombre, descripcion, formato, imagen) VALUES (:1, :2, :3, :4) RETURNING id INTO :5"
        out_id = cursor.var(int)
        cursor.execute(sql, [nombre, descripcion, formato, image_bytes, out_id])
        conn.commit()
        return int(out_id.getvalue()[0])
    finally:
        pool.release(conn)

def fetch_image_by_id(pool, img_id):
    conn = pool.acquire()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT nombre, descripcion, formato, imagen FROM imagenes WHERE id = :id", [img_id])
        row = cursor.fetchone()
        if row:
            nombre, descripcion, formato, blob = row
            image_bytes = blob.read() if blob is not None else None
            return {"id": img_id, "nombre": nombre, "descripcion": descripcion, "formato": formato, "bytes": image_bytes}
        else:
            return None
    finally:
        pool.release(conn)

def load_file_bytes(path):
    with open(path, "rb") as f:
        return f.read() 

def bytes_to_cv2_image(b):
    #lee bytes y convierte a imagen BGR ( openCV ) .
    arr = np.frombuffer(b,  dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  

def pil_to_cv2(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def show_images_grid(img1_bgr, img2_bgr, title1= "imagen 1", title2= "imagen 2"):
    #convert BGR -> RGB para matplotlib
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB) if img1_bgr is not None else None
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB) if img2_bgr is not None else None

    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title(title1)   
    if img1 is not None:
        plt.imshow(img1)
    else:
        plt.text(0.5,0.5,"no  disponible", ha='center')
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title(title2)   
    if img2 is not None:
        plt.imshow(img2)
    else:
        plt.text(0.5,0.5,"no disponible", ha='center')
    
    plt.tight_layout()
    plt.show()   

def compute_psnr(imgA_bgr, imgB_bgr):
    # Convertir a Y (luminance) o usar BGR; usaremos BGR en escala flotante
    if imgA_bgr.shape != imgB_bgr.shape:
        # redimensionar la segunda a la primera manteniendo proporción ( simple )
        imgB_bgr = cv2.resize(imgB_bgr, (imgA_bgr.shape[1], imgA_bgr.shape[0]))
    mse = np.mean((imgA_bgr.astype(np.float64) - imgB_bgr.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr_val = 10 * np.log10((PIXEL_MAX ** 2) / mse)
    return psnr_val

def compute_ssim(imgA_bgr, imgB_bgr):
    # Convertir a gris para SSIM
    if imgA_bgr.shape != imgB_bgr.shape:
        imgB_bgr = cv2.resize(imgB_bgr, (imgA_bgr.shape[1], imgA_bgr.shape[0]))
    grayA = cv2.cvtColor(imgA_bgr, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB_bgr, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim(grayA, grayB, data_range=grayA.max() - grayA.min())
    return ssim_val

def main():
    print("=== Programa: comparar imagenes (digital vs Canon EOS) con Oracle ===")
    # Cargar imágenes desde disco
    if not os.path.exists(DIGITAL_IMAGE_PATH):
        print(f"Error: no se encontró la imagen digital en {DIGITAL_IMAGE_PATH}")
        sys.exit(1)
    if not os.path.exists(CANON_IMAGE_PATH):
        print(f"Error: no se encontró la imagen Canon en {CANON_IMAGE_PATH}")
        sys.exit(1)

    digital_bytes = load_file_bytes(DIGITAL_IMAGE_PATH)
    canon_bytes   = load_file_bytes(CANON_IMAGE_PATH)

    # Intentar conexión a Oracle
    try:
        pool = connect_oracle()
        print("conexion a oracle: OK ( pool creado )")
        # Insertar imágenes
        id_digital = insert_image(pool, os.path.basename(DIGITAL_IMAGE_PATH),
                                  "imagen digital generada", os.path.splitext(DIGITAL_IMAGE_PATH)[1].lstrip('.'),
                                  digital_bytes)
        print(f"imagen digital insertada con id = {id_digital}")

        id_canon = insert_image(pool, os.path.basename(CANON_IMAGE_PATH),
                                 "imagen tomada con Canon EOS", os.path.splitext(CANON_IMAGE_PATH)[1].lstrip('.'),
                                 canon_bytes)
        print(f"imagen Canon insertada con id = {id_canon}")

        # Recuperar para mostrar / comparar
        rec_digital = fetch_image_by_id(pool, id_digital)
        rec_canon   = fetch_image_by_id(pool, id_canon)

        if not rec_digital or not rec_canon:
            print("error: no se pudieron recuperar una o ambas imagenes desde la BD .")
            sys.exit(1)

        img1 = bytes_to_cv2_image(rec_digital["bytes"])
        img2 = bytes_to_cv2_image(rec_canon["bytes"])

        # Mostrar en ventana (matplotlib)
        show_images_grid(img1, img2, title1=f"digital (id {id_digital})", title2=f"canon EOS (id {id_canon})")

        # Calcular métricas
        psnr_val = compute_psnr(img1, img2)
        ssim_val = compute_ssim(img1, img2)
        print(f"PSNR (dB): {psnr_val:.2f}")
        print(f"SSIM (0-1): {ssim_val:.4f}")

        # Resumen
        print("\nresumen:")
        if psnr_val == float('inf'):
            print("- las imágenes son exactamente iguales (PSNR = inf).")
        else:
            print(f"- PSNR indica diferencia relativa en luminancia/ruido (mayor = más parecido) .")
            print(f"- SSIM indica similaridad estructural (1.0 = idénticas) .")

    except Exception as e:
        # Si falla la conexión a Oracle, guardamos localmente como respaldo
        print("no se pudo conectar a Oracle o ocurrió un error. Se realizará respaldo local .")
        print("error:", str(e))
        ensure_backup_dir()
        b_digital = save_backup(DIGITAL_IMAGE_PATH, f"backup_{os.path.basename(DIGITAL_IMAGE_PATH)}")
        b_canon   = save_backup(CANON_IMAGE_PATH, f"backup_{os.path.basename(CANON_IMAGE_PATH)}")
        timestamp = datetime.utcnow().isoformat()
        append_metadata_csv([timestamp, os.path.basename(DIGITAL_IMAGE_PATH), "Imagen digital (backup)", os.path.splitext(DIGITAL_IMAGE_PATH)[1].lstrip('.'), b_digital])
        append_metadata_csv([timestamp, os.path.basename(CANON_IMAGE_PATH), "Imagen Canon (backup)", os.path.splitext(CANON_IMAGE_PATH)[1].lstrip('.'), b_canon])
        # Mostrar localmente las imágenes (usando OpenCV)
        img1 = cv2.imread(b_digital)
        img2 = cv2.imread(b_canon)
        if img1 is None or img2 is None:
            print("error: no se pudieron leer los archivos de respaldo. ")
            sys.exit(1)
        show_images_grid(img1, img2, title1="Digital (backup)", title2="Canon EOS (backup)")
        psnr_val = compute_psnr(img1, img2)
        ssim_val = compute_ssim(img1, img2)
        print(f"PSNR (dB) [backup]: {psnr_val:.2f}")
        print(f"SSIM (0-1) [backup]: {ssim_val:.4f}")
        print(f"se guardaron respaldos en {BACKUP_DIR} y metadata en {METADATA_CSV}")

if __name__ == "__main__":
    main()
