# Diseño de un flujo analítico para la clasificación de crisis epilépticas en EEG de superficie

# Descripción general del proyecto

Este proyecto documenta y ejecuta un flujo analítico integral de Machine Learning diseñado para la clasificación automática de crisis epilépticas utilizando señales de electroencefalografía (EEG) de superficie. Toda la documentación técnica, los hallazgos del análisis exploratorio, los experimentos de selección de características, los procedimientos de optimización y los resultados de validación se encuentran disponibles y renderizados para su consulta en la página oficial del proyecto: https://miguelpvmr.github.io/Machine-Learning-EEG/

A diferencia de un ejercicio puramente estadístico, este repositorio abarca desde el acondicionamiento de señales EEG clínicas y la extracción de características tiempo-frecuencia, hasta la construcción de arquitecturas de ensamble basadas en *Stacking* e interpretación mediante valores SHAP. El flujo de trabajo integra estrategias avanzadas de balanceo de clases, optimización de hiperparámetros y validación rigurosa a nivel de pacientes, garantizando que los resultados obtenidos reflejen escenarios de generalización clínicamente realistas.

# Fuente de datos

El conjunto de datos utilizado corresponde al **Temple University Hospital Seizure Corpus (TUSZ) v1.5.2/v2.0.0**, uno de los repositorios públicos de EEG clínico más extensos disponibles para investigación en epilepsia.

El corpus contiene registros de electroencefalografía anotados por neurólogos expertos e incluye segmentos de actividad basal y múltiples tipos de crisis epilépticas registradas bajo condiciones hospitalarias reales. Estas características convierten a TUSZ en un entorno particularmente desafiante para el desarrollo de sistemas automáticos de clasificación.

**Características generales del corpus:**

* Registros clínicos de EEG de superficie.
* Anotaciones temporales realizadas por especialistas.
* Actividad basal y múltiples categorías ictales.
* Variabilidad interpaciente e intersesión.
* Presencia de artefactos fisiológicos y ambientales propios de escenarios clínicos reales.

**Origen:** https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

**Cita y referencia:**

* Shah, V., von Weltin, E., Lopez, S., McHugh, J. R., Veloso, L., Golmohammadi, M., Obeid, I., & Picone, J. (2018). *The Temple University Hospital Seizure Detection Corpus*. Frontiers in Neuroinformatics, 12, 83. https://doi.org/10.3389/fninf.2018.00083

# Arquitectura y tecnologías utilizadas

Para garantizar la reproducibilidad experimental y la escalabilidad de los análisis, el proyecto hace uso de las siguientes herramientas:

* **MNE-Python:** Lectura, procesamiento y manipulación de registros EEG en formato EDF.
* **NumPy & Pandas:** Manipulación eficiente de datos tabulares y matrices numéricas.
* **Scikit-Learn:** Construcción de pipelines de aprendizaje automático y validación cruzada.
* **XGBoost:** Implementación de modelos basados en Gradient Boosting.
* **Optuna:** Optimización bayesiana de hiperparámetros.
* **DEAP:** Optimización mediante algoritmos evolutivos.
* **SHAP:** Interpretabilidad y explicación de modelos predictivos.
* **Matplotlib & Plotly:** Visualización de señales, métricas y resultados experimentales.
* **Joblib:** Serialización de modelos y objetos de aprendizaje.
* **Jupyter Notebook:** Desarrollo reproducible de experimentos y documentación.

# Flujo analítico

El pipeline experimental desarrollado en este proyecto comprende las siguientes etapas:

1. Consolidación y limpieza de registros EEG.
2. Re-referenciación digital mediante montaje bipolar longitudinal (*Double Banana*).
3. Filtrado digital en cascada (Butterworth + Notch).
4. Segmentación temporal mediante ventanas deslizantes.
5. Escalado robusto de cada ventana EEG.
6. Extracción de características temporales, espectrales y tiempo-frecuencia.
7. Balanceo de clases mediante técnicas de sobremuestreo y ajuste de pesos.
8. Selección de características.
9. Optimización de hiperparámetros.
10. Entrenamiento de modelos base y arquitecturas de ensamble.
11. Interpretación de modelos mediante SHAP.
12. Evaluación final sobre pacientes no observados durante el entrenamiento.

# Estrategias de balanceo y optimización

Con el objetivo de mitigar el fuerte desbalance presente entre actividad basal y eventos ictales, se evaluaron múltiples estrategias de balanceo:

* Pesos de clase (*Class Weights*).
* SMOTE.
* ADASYN.

De forma complementaria, la búsqueda de hiperparámetros se realizó mediante diferentes enfoques:

* Grid Search.
* Random Search.
* Optimización Bayesiana mediante Optuna.
* Algoritmos Evolutivos mediante DEAP.

Estas estrategias fueron integradas dentro de esquemas de validación cruzada para garantizar una estimación robusta del rendimiento de los modelos.

# Modelos evaluados

Se realizó un benchmarking exhaustivo entre múltiples algoritmos de aprendizaje automático y arquitecturas de ensamble.

Los modelos evaluados incluyen:

* Regresión Logística.
* K-Nearest Neighbors (KNN).
* Gaussian Naive Bayes (GNB).
* Árboles de Decisión.
* Random Forest.
* Support Vector Machines (SVM).
* XGBoost.
* Arquitecturas de Stacking.

Los modelos fueron comparados bajo protocolos homogéneos de validación cruzada y optimización de hiperparámetros, con el objetivo de identificar aquellas configuraciones capaces de generalizar adecuadamente sobre pacientes no observados durante el entrenamiento.

Los resultados completos de cada experimento, junto con los análisis comparativos, procedimientos de selección de características y configuraciones óptimas de hiperparámetros, pueden consultarse en la documentación oficial del proyecto.

# Estructura del repositorio

```text
TUSZ_project/
├── data/
│   └── (Archivos procesados de TUSZ)
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_test_models.ipynb
│   └── 3_model_explainability.ipynb
├── src/
├── results/
├── models/
├── TUSZ_DataLake/
│   ├── 00_Raw_Data/
│   ├── 01_Raw_Consolidated/
│   ├── 02_Metadata/
│   ├── 03_TUSZ_Clean/
│   └── 04_TUSZ_Features_ML/
├── .gitignore
├── README.md
└── environment.yml
```

# Selección del modelo y métricas

Dada la naturaleza altamente desbalanceada del corpus TUSZ, la métrica rectora del proyecto es el Macro F2-Score, ya que asigna una mayor importancia al Recall y penaliza con mayor severidad los falsos negativos. Esta propiedad resulta especialmente relevante en aplicaciones clínicas, donde la omisión de una crisis epiléptica puede tener consecuencias significativas.

Para cada categoría clínica se calcula inicialmente un F2-Score binario bajo un esquema one-vs-rest:

$$
\frac{(1+\beta^2)\cdot Precision_i \cdot Recall_i}
{\beta^2 \cdot Precision_i + Recall_i}
\qquad \text{con } \beta = 2
$$

Posteriormente, el valor final reportado corresponde al promedio aritmético de los F2 obtenidos para todas las clases:

$$
\frac{1}{K}
\sum_{i=1}^{K}
F_{2,i}
$$

donde (K) representa el número total de categorías consideradas en el problema de clasificación.

Como métrica complementaria se emplea el G-Index, definido como la media geométrica de los recalls obtenidos para cada clase:

$$
\left(
\prod_{i=1}^{K}
Recall_i
\right)^{\frac{1}{K}}
$$

Esta métrica resulta especialmente útil en escenarios multiclase desbalanceados, ya que penaliza fuertemente los modelos que presentan un bajo rendimiento en alguna de las categorías, incluso cuando las demás clases muestran desempeños elevados.

Adicionalmente, se reportan métricas tradicionales de clasificación:

* Accuracy.
* Precision.
* Recall.
* F1-Score.
* Matrices de confusión.
* Curvas ROC.
* Curvas Precision–Recall.

La selección final de modelos se fundamenta principalmente en el desempeño alcanzado sobre Macro-F2 y G-Index bajo esquemas de validación cruzada estratificada y evaluación independiente sobre pacientes no observados durante el entrenamiento.
