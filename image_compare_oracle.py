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