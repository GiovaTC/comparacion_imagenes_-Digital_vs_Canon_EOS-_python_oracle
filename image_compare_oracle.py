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