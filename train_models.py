#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def load_letor_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    print(f"Cargando archivos en {file_path}...")
    
    labels = []
    query_ids = []
    features = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            

            label = int(parts[0])
            labels.append(label)

            qid_part = parts[1]  
            qid = int(qid_part.split(':')[1])
            query_ids.append(qid)

            feature_vector = np.zeros(136)
            for i in range(2, len(parts)):
                if ':' in parts[i]:
                    feat_idx, feat_val = parts[i].split(':')
                    feat_idx = int(feat_idx) - 1 
                    feat_val = float(feat_val)
                    if 0 <= feat_idx < 136:
                        feature_vector[feat_idx] = feat_val
            
            features.append(feature_vector)
    
    return np.array(labels), np.array(query_ids), np.array(features)

def create_pairwise_data(labels: np.ndarray, query_ids: np.ndarray, 
                        features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    print("Creando datos de entrenamiento por pares...")
    
    pairwise_features = []
    pairwise_labels = []
    
    unique_queries = np.unique(query_ids)
    
    for qid in unique_queries:
        query_mask = query_ids == qid
        query_labels = labels[query_mask]
        query_features = features[query_mask]

        n_docs = len(query_labels)
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if query_labels[i] != query_labels[j]:
                    if query_labels[i] > query_labels[j]:
                        feature_diff = query_features[i] - query_features[j]
                        pair_label = 1
                    else:

                        feature_diff = query_features[j] - query_features[i]
                        pair_label = 1
                    
                    pairwise_features.append(feature_diff)
                    pairwise_labels.append(pair_label)
    
    return np.array(pairwise_features), np.array(pairwise_labels)

def get_query_groups(query_ids: np.ndarray) -> List[int]:

    unique_queries, counts = np.unique(query_ids, return_counts=True)
    return counts.tolist()

def train_pointwise_model(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:

    print("Entrenando modelo pointwise...")

    models_params = {
        'linear': (LinearRegression(), {}),
        'ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
        'lasso': (Lasso(), {'alpha': [0.1, 1.0, 10.0]})
    }
    
    best_model = None
    best_score = -np.inf
    best_params = None
    best_model_name = None
    
    for model_name, (model, param_grid) in models_params.items():
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(features, labels)
            current_model = grid_search.best_estimator_
            current_score = grid_search.best_score_
            current_params = grid_search.best_params_
        else:
            current_model = model
            current_model.fit(features, labels)
            predictions = current_model.predict(features)
            current_score = -np.mean((labels - predictions) ** 2)
            current_params = {}
        
        print(f"{model_name}: Score = {current_score:.4f}, Parametros = {current_params}")
        
        if current_score > best_score:
            best_score = current_score
            best_model = current_model
            best_params = current_params
            best_model_name = model_name
    
    print(f"Mejor modelo Pointwise: {best_model_name} con Score {best_score:.4f}")
    
    return {
        'model': best_model,
        'model_type': 'pointwise',
        'model_name': best_model_name,
        'best_params': best_params,
        'best_score': best_score
    }

def train_pairwise_model(pairwise_features: np.ndarray, pairwise_labels: np.ndarray) -> Dict[str, Any]:

    print("Entrenando modelo por pares...")

    unique_labels = np.unique(pairwise_labels)
    if len(unique_labels) < 2:
        print(f"Advertencia: Los datos por pares tienen solo {len(unique_labels)} etiqueta(s) única(s). Saltando entrenamiento por pares.")
        return None
    

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pairwise_features)
    

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    if len(pairwise_features) > 50000:
        print("Se ha detectado un gran conjunto de datos. Usando un subconjunto para la sintonización de hiperparámetros...")
        subset_size = 50000
        indices = np.random.choice(len(pairwise_features), subset_size, replace=False)
        tune_features = scaled_features[indices]
        tune_labels = pairwise_labels[indices]
    else:
        tune_features = scaled_features
        tune_labels = pairwise_labels
    
    try:
        svm = SVC(probability=False)  
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(tune_features, tune_labels)
        

        best_svm = SVC(**grid_search.best_params_)
        best_svm.fit(scaled_features, pairwise_labels)

        print(f"Mejor modelo por pares params: {grid_search.best_params_}")
        print(f"Mejor modelo por pares CV score: {grid_search.best_score_:.4f}")
        
        return {
            'model': best_svm,
            'scaler': scaler,
            'model_type': 'pairwise',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    except ValueError as e:
        print(f"Error : {e}")
        print("Esto puede ser debido a la falta de diversidad en las etiquetas por pares para la validación cruzada.")


        try:
            print("Intentando entrenar modelo por pares sin validación cruzada...")
            fallback_svm = SVC(C=1.0, kernel='rbf', gamma='scale')
            fallback_svm.fit(scaled_features, pairwise_labels)
            
            return {
                'model': fallback_svm,
                'scaler': scaler,
                'model_type': 'pairwise',
                'best_params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                'best_score': 0.0  
            }
        except Exception as fallback_e:
            print(f"Error : {fallback_e}")
            return None

def train_listwise_model(features: np.ndarray, labels: np.ndarray, 
                        query_groups: List[int]) -> Dict[str, Any]:

    print("Entrenando modelo por listas (XGBoost)...")

  
    dtrain = xgb.DMatrix(features, label=labels)
    dtrain.set_group(query_groups)

    param_grids = [
        {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8
        },
        {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'eta': 0.05,
            'max_depth': 8,
            'subsample': 0.9
        },
        {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@10',
            'eta': 0.2,
            'max_depth': 4,
            'subsample': 0.7
        }
    ]
    
    best_model = None
    best_score = -np.inf
    best_params = None
    
    for params in param_grids:
        print(f"Entrenando modelo por listas: {params}")

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            verbose_eval=False
        )

        predictions = model.predict(dtrain)

        score = np.mean(predictions)  
        
        print(f"Score del modelo: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
    
    print(f"Parametros del mejor modelo: {best_params}")
    
    return {
        'model': best_model,
        'model_type': 'listwise',
        'best_params': best_params,
        'best_score': best_score
    }

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos Learning to Rank')
    parser.add_argument('train_file', type=str, help='ruta al archivo de datos de entrenamiento')
    parser.add_argument('--output_dir', type=str, default='.', 
                       help='Directorio para guardar los modelos entrenados (por defecto: directorio actual)')
    
    args = parser.parse_args()
    
    print("=== Learning to Rank - Script de Entrenamiento ===")
    print(f"Archivo de entrenamiento: {args.train_file}")
    print(f"Directorio de salida: {args.output_dir}")


    try:
        labels, query_ids, features = load_letor_data(args.train_file)
        print(f"Se cargaron {len(labels)} muestras con {features.shape[1]} características")
        print(f"Número de consultas únicas: {len(np.unique(query_ids))}")
        print(f"Distribución de etiquetas: {np.bincount(labels)}")
    except Exception as e:
        print(f"Error al cargar los datos de entrenamiento: {e}")
        sys.exit(1)

    try:
        pointwise_result = train_pointwise_model(features, labels)
        joblib.dump(pointwise_result, f"{args.output_dir}/pointwise_model.joblib")
        print("✓ Modelo pointwise guardado con éxito")
    except Exception as e:
        print(f"Error al entrenar el modelo pointwise: {e}")


    try:
        pairwise_features, pairwise_labels = create_pairwise_data(labels, query_ids, features)
        print(f"Se crearon {len(pairwise_features)} muestras de entrenamiento por pares")

        pairwise_result = train_pairwise_model(pairwise_features, pairwise_labels)
        if pairwise_result is not None:
            joblib.dump(pairwise_result, f"{args.output_dir}/pairwise_model.joblib")
            print("✓ Modelo pairwise guardado con éxito")
        else:
            print("⚠ Entrenamiento del modelo pairwise omitido debido a la falta de diversidad en los datos")
    except Exception as e:
        print(f"Error al entrenar el modelo pairwise: {e}")

    try:
        query_groups = get_query_groups(query_ids)
        print(f"Tamaños de grupos de consultas: min={min(query_groups)}, max={max(query_groups)}, mean={np.mean(query_groups):.2f}")

        listwise_result = train_listwise_model(features, labels, query_groups)
        joblib.dump(listwise_result, f"{args.output_dir}/listwise_model.joblib")
        print("✓ Modelo listwise guardado con éxito")
    except Exception as e:
        print(f"Error al entrenar el modelo listwise: {e}")

    print("\n=== Entrenamiento completado ===")

if __name__ == "__main__":
    main()
