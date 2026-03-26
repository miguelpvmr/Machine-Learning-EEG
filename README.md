# Diseño de un flujo analítico para la clasificación de crisis epilépticas en EEG de superficie: Evaluación sobre el Corpus TUSZ

---

**Autores:**
* **Paula Andrea Gómez Vargas** (apaulag@uninorte.edu.co)
* **Juan Camilo Mendoza Arango** (cjarango@uninorte.edu.co)
* **Miguel Ángel Pérez Vargas** (vargasmiguel@uninorte.edu.co)

---

# Project Overview

Este proyecto de investigación se centra en el diseño e implementación de un pipeline de **Machine Learning** para la clasificación automática de crisis epilépticas utilizando señales de EEG de superficie. La evaluación se realiza sobre el **Temple University Hospital EEG Seizure Corpus (TUSZ)**, uno de los repositorios públicos más grandes y complejos en el área de la neuroingeniería.

A diferencia de los enfoques convencionales, este flujo integra técnicas de **Escalado Robusto** para mitigar artefactos biomédicos y **Aprendizaje de Variedades (UMAP)** para la visualización tridimensional de fenotipos ictales. El objetivo es identificar patrones espaciotemporales que permitan distinguir entre actividad **Basal**, **Focal** y **Generalizada** con alta precisión y explicabilidad clínica.

## Data Source

El dataset utilizado es el **TUH EEG Seizure Corpus (TUSZ) v1.5.2/v2.0.0**, el cual contiene registros clínicos anotados por expertos neurólogos. 
* **Origen:** [The Temple University Hospital EEG Resources](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)
* **Referencia:** Shah, V., et al. (2018). "The Temple University Hospital EEG Seizure Corpus."

# Project Structure

Para garantizar la reproducibilidad, el proyecto sigue una arquitectura de **Data Lake** local y modularización de código en la carpeta `src/`.

```text
TUSZ_project/
├── data/
│   └── (Archivos procesados de TUSZ)
├── notebooks/
│   └── Análisis Exploratorio y Caracterización de Fenotipos Ictales.ipynb
├── src/
│   ├── visualization_utils.py  # Aquí reside plot_umap_3d
│   └── preprocessing.py
├── TUSZ_DataLake/
│   ├── 00_Raw_Data/           # Archivos .edf originales
│   ├── 01_Raw_Consolidated/   # Metadata extraída
│   ├── 02_Metadata/           # Etiquetas y anotaciones
│   ├── 03_TUSZ_Clean/         # Señales filtradas
│   └── 04_TUSZ_Features_ML/   # Características (ranking_features)
├── .gitignore
├── README.md
└── environment.yml
```

Objectives
El objetivo central es construir un modelo de clasificación multiclase capaz de identificar el tipo de crisis en ventanas de tiempo cortas.

Target Definition
0 - Basal (Reposo): Actividad eléctrica normal o interictal.

1 - Focal: Crisis localizadas en una región específica del cerebro.

2 - Generalizada: Crisis que involucran ambos hemisferios desde el inicio.
