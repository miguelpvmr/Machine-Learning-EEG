# Diseño de un flujo analítico para la clasificación de crisis epilépticas en EEG de superficie: Evaluación sobre el Corpus TUSZ

---

**Autores:**
* **Paula Andrea Gómez Vargas** (apaulag@uninorte.edu.co)
* **Juan Camilo Mendoza Arango** (cjarango@uninorte.edu.co)
* **Miguel Ángel Pérez Vargas** (vargasmiguel@uninorte.edu.co)

---

# Descripción general del proyecto

Este proyecto de investigación se centra en el diseño e implementación de un flujo de trabajo (pipeline) de **aprendizaje automático** para la clasificación automática de crisis epilépticas utilizando señales de EEG de superficie. La evaluación se realiza sobre el **Corpus de Crisis de EEG del Hospital de la Universidad de Temple (TUSZ)**, uno de los repositorios públicos más grandes y complejos en el área de la neuroingeniería.

A diferencia de los enfoques convencionales, este flujo integra técnicas de **escalado robusto** para mitigar artefactos biomédicos y **aprendizaje de variedades (UMAP)** para la visualización tridimensional de fenotipos ictales. El objetivo es identificar patrones espaciotemporales que permitan distinguir entre actividad **basal**, **focal** y **generalizada** con alta precisión y explicabilidad clínica.

## Fuente de datos

El conjunto de datos utilizado es el **Corpus de Crisis de EEG TUH (TUSZ) v1.5.2/v2.0.0**, el cual contiene registros clínicos anotados por expertos neurólogos.  

* **Origen:** https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ 
* **Referencia:** Shah, V., Weltin, E. von, Lopez, S., McHugh, J. R., Veloso, L., Golmohammadi, M., Obeid, I., & Picone, J. (2018). The Temple University Hospital Seizure Detection Corpus. *Frontiers in Neuroinformatics, 12*, 83. https://doi.org/10.3389/fninf.2018.00083

# Estructura del proyecto

Para garantizar la reproducibilidad, el proyecto sigue una arquitectura de **Data Lake** local y una modularización del código en la carpeta `src/`.

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
