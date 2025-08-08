# Implementación de Learning to Rank

Este proyecto implementa y compara tres enfoques de Learning to Rank: pointwise, pairwise y listwise, utilizando el conjunto de datos MSLR-WEB10K.

## Miembros del equipo


## Estructura del proyecto
```
├── train_models.py          # Script de entrenamiento para los tres enfoques
├── evaluate_models.py       # Script de evaluación con métricas de ranking
├── requirements.txt         # Dependencias de Python
├── README.md               # Este archivo
└── MSLR-WEB10K/           # Directorio del conjunto de datos
    └── Fold1/
        ├── train.txt       # Datos de entrenamiento
        ├── vali.txt        # Datos de validación
        └── test.txt        # Datos de prueba
```

## Instalación
Asegúrate de tener Python 3.7+ instalado.

Instala los paquetes requeridos:
```bash
pip install -r requirements.txt
```

## Formato del conjunto de datos
El conjunto de datos MSLR-WEB10K utiliza el formato LETOR:

- **Primera columna:** Etiqueta de relevancia (0-4, donde 4 es perfectamente relevante)
- **Segunda columna:** ID de consulta (`qid:X`)
- **Las 136 columnas restantes:** Características (`feature_id:valor`)

## Uso

### Entrenamiento de modelos
Entrena los tres modelos (pointwise, pairwise, listwise) usando el script de entrenamiento:
```bash
python train_models.py MSLR-WEB10K/Fold1/train.txt --output_dir ./models
```

**Argumentos:**
- `train_file`: Ruta al archivo de datos de entrenamiento (**obligatorio**)
- `--output_dir`: Directorio para guardar los modelos entrenados (por defecto: directorio actual)

**Salida:**
- `pointwise_model.joblib`: Modelo de regresión pointwise entrenado
- `pairwise_model.joblib`: Modelo SVM pairwise entrenado
- `listwise_model.joblib`: Modelo XGBoost listwise entrenado

### Evaluación de modelos
Evalúa los modelos entrenados en el conjunto de validación:
```bash
python evaluate_models.py MSLR-WEB10K/Fold1/vali.txt --model_dir ./models
```

**Argumentos:**
- `validation_file`: Ruta al archivo de datos de validación (**obligatorio**)
- `--model_dir`: Directorio que contiene los modelos entrenados (por defecto: directorio actual)

**Salida:**
El script imprime métricas de evaluación que incluyen:
- nDCG@5 y nDCG@10 (Ganancia Acumulativa Descontada Normalizada)
- MAP (Precisión Promedio Media)
- Análisis de rendimiento por características de la consulta

## Detalles de implementación

### Enfoque Pointwise
- **Modelo:** Regresión lineal con ajuste de hiperparámetros
- **Variantes:** Regresión Lineal, Ridge, Lasso
- **Hiperparámetros:** Valores de Alpha `[0.1, 1.0, 10.0]` para modelos regularizados
- **Evaluación:** Validación cruzada con error cuadrático medio negativo (negative MSE)

### Enfoque Pairwise
- **Modelo:** Máquina de Vectores de Soporte (SVM)
- **Generación de datos:** Crea pares de preferencia a partir de los datos pointwise
- **Preprocesamiento:** Escalado de características usando `StandardScaler`
- **Hiperparámetros:**
  - `C`: `[0.1, 1.0, 10.0]`
  - `Kernel`: `['linear', 'rbf']`
  - `Gamma`: `['scale', 'auto']`
- **Predicción:** Utiliza `decision_function()` para puntuaciones de ranking continuas

### Enfoque Listwise
- **Modelo:** XGBoost con objetivo `rank:ndcg`
- **Función objetivo:** `rank:ndcg` (optimiza para NDCG)
- **Información de grupo:** Utiliza grupos de consultas para un entrenamiento listwise adecuado
- **Hiperparámetros:**
  - **Tasa de aprendizaje:** `[0.05, 0.1, 0.2]`
  - **Profundidad máxima:** `[4, 6, 8]`
  - **Submuestra:** `[0.7, 0.8, 0.9]`

## Métricas de evaluación
- **nDCG@k (Ganancia Acumulativa Descontada Normalizada)**  
  Mide la calidad del ranking considerando tanto la relevancia como la posición:
  - nDCG@5: Se centra en los 5 primeros resultados
  - nDCG@10: Se centra en los 10 primeros resultados
- **MAP (Precisión Promedio Media)**  
  Mide la precisión en todos los niveles de recuperación, tratando la relevancia como binaria.

## Tiempo de ejecución esperado
- **Entrenamiento:** 30-60 minutos (dependiendo del hardware)
- **Evaluación:** 5-10 minutos

Para conjuntos de datos grandes, el enfoque pairwise puede tardar más debido al número cuadrático de pares generados.

## Solución de problemas

### Problemas de memoria
Si encuentras errores de memoria:
- El enfoque pairwise utiliza un subconjunto para el ajuste de hiperparámetros cuando los datos son grandes.
- Considera reducir el tamaño del conjunto de datos para las pruebas iniciales.
- Monitoriza el uso de la memoria del sistema durante el entrenamiento.

### Optimización del rendimiento
- La implementación utiliza procesamiento paralelo cuando es posible (`n_jobs=-1`).
- La búsqueda en cuadrícula (grid search) se realiza con validación cruzada.
- Los conjuntos de datos grandes se manejan con muestreo adecuado.

## Dependencias de archivos
- `joblib`: Para la persistencia del modelo
- `scikit-learn`: Para los modelos pointwise y pairwise
- `xgboost`: Para el ranking listwise
- `numpy`: Para cálculos numéricos

## Archivos de salida
Después del entrenamiento, tendrás tres archivos de modelo:
- `pointwise_model.joblib`: Contiene el modelo entrenado, metadatos e hiperparámetros.
- `pairwise_model.joblib`: Contiene el modelo SVM, el escalador y los metadatos.
- `listwise_model.joblib`: Contiene el modelo XGBoost y los metadatos.

Cada archivo se puede cargar de forma independiente para su evaluación o uso posterior.
