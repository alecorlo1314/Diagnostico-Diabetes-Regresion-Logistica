Resumen del Proyecto de Predicción de Diabetes
Este proyecto tiene como objetivo desarrollar un modelo de clasificación capaz de predecir la probabilidad de que una persona padezca diabetes, basándose en un conjunto de características de salud. El problema se aborda como un tarea de clasificación binaria.

1. Entendimiento del Problema
Variable Objetivo: label (Diabetes: 1, No Diabetes: 0).
Características Clave: glucose, insulin, bmi, age.
Tipo de Problema: Clasificación binaria.
Impacto del Modelo: Proporcionar una probabilidad de diabetes, útil para la detección temprana y la intervención.
Métricas Importantes: Accuracy, Precision, Recall, y ROC AUC, para evaluar el rendimiento del modelo.
Restricciones: Posibilidad de datasets pequeños o inconsistentes.
2. Importación y Carga de Datos
Se importan las librerías necesarias (pandas, numpy, matplotlib, seaborn, sklearn, imblearn, mlxtend) y se carga el dataset 'Pima Indians Diabetes Database' desde KaggleHub. Las columnas son renombradas para mayor claridad.

3. Exploración Inicial de Datos
Estructura y Tipos de Datos: Se examinaron las primeras filas (.head()), la información general (.info()) y las estadísticas descriptivas (.describe()) del dataset.
Valores Duplicados y Faltantes: Se verificó la ausencia de filas duplicadas y valores nulos explícitos.
Identificación de Outliers: Se realizó una inspección inicial de valores atípicos utilizando el método del Rango Intercuartílico (IQR) para las variables glucose, insulin, bmi y age, detectando la presencia de ceros en glucose, insulin, y bmi que funcionalmente representan valores faltantes, así como valores extremos en insulin y age.
4. Limpieza de Datos
Imputación de Valores Faltantes: Los valores 0 en glucose, insulin y bmi (que son biológicamente imposibles o altamente improbables en un contexto normal y por lo tanto tratados como nulos) se identificaron. Aunque en la ejecución actual no se implementó explícitamente la sustitución por np.nan antes de la imputación con la mediana, la intención era tratar estos ceros. La estrategia propuesta es imputar estos valores con la mediana de sus respectivas columnas.
Manejo de Outliers: Para la columna insulin, que presentaba valores atípicos significativos, se aplicó un RobustScaler. Esto ayuda a mitigar el impacto de los outliers sin eliminarlos, ya que el escalador robusto es menos sensible a ellos que StandardScaler o MinMaxScaler.
5. Análisis Exploratorio Avanzado (EDA)
Visualización de Distribuciones y Relaciones:
Se utilizaron scatterplotmatrix para visualizar las relaciones por pares entre las variables y sus distribuciones marginales.
Histogramas fueron generados para cada característica para entender sus distribuciones.
Balance de Clases: Se analizó el balance entre las clases 'Diabetes' y 'No Diabetes', observando un desbalance (267 casos de diabetes vs. 500 de no diabetes).
Matriz de Correlación: Se generó un mapa de calor (heatmap) de la matriz de correlación para identificar las relaciones lineales entre las características y con la variable objetivo. La glucose mostró la correlación más alta con la label.
6. Preparación de Datos (Feature Engineering)
Selección de Características: Se seleccionaron glucose, insulin, bmi y age como las características (X) para el modelo, siendo label la variable objetivo (y).
Manejo del Desbalance de Clases: Para abordar el desbalance de clases, se aplicó la técnica de sobremuestreo SMOTE (Synthetic Minority Over-sampling Technique) a los datos de entrenamiento, lo que creó muestras sintéticas de la clase minoritaria (Diabetes) para equilibrar el dataset.
División del Dataset: El conjunto de datos (después de SMOTE) se dividió en conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test) con una proporción del 70% para entrenamiento y 30% para prueba, utilizando stratify=y_res para asegurar que la proporción de clases se mantuviera en ambos conjuntos.
Escalamiento de Características: Se utilizó StandardScaler para escalar las características en los conjuntos de entrenamiento y prueba. Esto es crucial para modelos basados en distancia como la regresión logística.
7. Modelo y Entrenamiento
Selección del Modelo: Se eligió un modelo de LogisticRegression debido a su interpretabilidad y buen rendimiento en problemas de clasificación binaria. Se configuró max_iter=1000 para asegurar la convergencia y class_weight='balanced' para manejar el desbalance de clases (aunque ya se usó SMOTE, esto añade una capa extra de robustez).
Entrenamiento: El modelo se entrenó con los datos escalados de entrenamiento (X_train, y_train).
8. Evaluación del Modelo
Predicciones: Se generaron predicciones (prediccion_diabetes) sobre el conjunto de prueba (X_test).
Coeficientes del Modelo: Se examinaron los coeficientes de la regresión logística para entender la importancia relativa de cada característica. La Glucosa fue la característica con el coeficiente positivo más alto, indicando su fuerte relación con la probabilidad de diabetes.
Matriz de Confusión: Se visualizó la matriz de confusión para entender los verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos del modelo. Se obtuvo un resultado de [[117, 33], [40, 110]] para el umbral por defecto (0.5).
Métricas de Rendimiento (Umbral por defecto):
Accuracy: 0.76
Classification Report: Se obtuvo un reporte detallado con Precision, Recall y F1-score para ambas clases, mostrando un rendimiento equilibrado (Precision ~0.76, Recall ~0.76).
Curva ROC y Área Bajo la Curva (AUC): Se calculó y graficó la curva ROC, obteniendo un AUC de 0.83, lo que indica una buena capacidad de discriminación del modelo.
Optimización del Umbral: Se experimentó con un umbral de clasificación ajustado a 0.41. Con este umbral, la precisión para 'Sin diabetes' fue 0.80 y el recall 0.67; para 'Con diabetes', la precisión fue 0.71 y el recall 0.83. Esto muestra un compromiso entre falsos positivos y falsos negativos que podría ser deseable dependiendo del contexto de la aplicación.
