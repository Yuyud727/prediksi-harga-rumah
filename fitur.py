import numpy as np
import pandas as pd

KOLOM_NUMERIK = ["Kamar_Tidur", "Kamar_Mandi", "Garasi", "Luas_Tanah", "Luas_Bangunan"]
FITUR_BARU = ["Log_Luas_Tanah", "Log_Luas_Bangunan", "Rasio_LB_LT", "Total_Kamar", "Ada_Garasi"]
KOLOM_KATEGORI = ["Kecamatan"]
KOLOM_INPUT = KOLOM_NUMERIK + KOLOM_KATEGORI


def rekayasa_fitur(X):
    """Tambahkan fitur turunan. Nilai kosong tetap NaN (diimputasi di dalam pipeline)."""
    X = X.copy()
    lt = X["Luas_Tanah"].astype(float)
    lb = X["Luas_Bangunan"].astype(float)
    X["Log_Luas_Tanah"] = np.log1p(lt)
    X["Log_Luas_Bangunan"] = np.log1p(lb)
    X["Rasio_LB_LT"] = (lb / lt.where(lt > 0)).clip(upper=6)
    X["Total_Kamar"] = X["Kamar_Tidur"].astype(float) + X["Kamar_Mandi"].astype(float)
    X["Ada_Garasi"] = (X["Garasi"].astype(float) > 0).astype(float)
    return X
