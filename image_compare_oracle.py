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