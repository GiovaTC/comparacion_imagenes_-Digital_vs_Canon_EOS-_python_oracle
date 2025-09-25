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
