import pandas as pd
import numpy as np
from pathlib import Path

# Usamos la ruta exacta que encontró tu terminal
file_path = Path(r"D:\TUSZ_DataLake\03_TUSZ_Clean\3\s002_2015\aaaaatvk_s002_t002.parquet")

print(f"--- 🕵️ Analizando: {file_path.name} ---")

try:
    # Pandas lee tanto archivos .parquet como carpetas de parquet automáticamente
    df = pd.read_parquet(file_path)
    
    print(f"\n✅ Archivo cargado con éxito. Dimensiones: {df.shape}")
    print("\n📋 Columnas y sus tipos de datos:")
    print(df.dtypes)
    
    # 🚨 BUSCAMOS EL CULPABLE DEL 'str - str'
    text_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if text_cols:
        print(f"\n⚠️ ¡LO ENCONTRÉ! Estas columnas NO son números: {text_cols}")
        print("\nEjemplo de los valores que tienen:")
        print(df[text_cols].head())
    else:
        print("\n🤔 Qué extraño... todas las columnas son numéricas según Pandas.")
        print("Revisemos si los números están 'disfrazados' de texto (object).")
        for col in df.columns:
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, str):
                print(f"  - La columna {col} contiene STRINGS (ej: '{sample_val}')")

except Exception as e:
    print(f"❌ Error al intentar leer el archivo: {e}")