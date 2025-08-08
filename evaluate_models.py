#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import ndcg_score
import xgboost as xgb

def load_letor_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    print(f"Cargando datos de : {file_path}...")
    
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

def calculate_dcg(relevances: np.ndarray, k: int = None) -> float:

    if k is not None:
        relevances = relevances[:k]
    
    if len(relevances) == 0:
        return 0.0
    
    dcg = relevances[0]
    for i in range(1, len(relevances)):
        dcg += relevances[i] / np.log2(i + 1)
    
    return dcg

def calculate_ndcg(true_relevances: np.ndarray, predicted_scores: np.ndarray, k: int = None) -> float:

    if len(true_relevances) == 0:
        return 0.0
    

    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_relevances = true_relevances[sorted_indices]

    dcg = calculate_dcg(sorted_relevances, k)
    

    ideal_relevances = np.sort(true_relevances)[::-1]
    idcg = calculate_dcg(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def calculate_average_precision(true_relevances: np.ndarray, predicted_scores: np.ndarray) -> float:

    if len(true_relevances) == 0:
        return 0.0

    binary_relevances = (true_relevances > 0).astype(int)
    
    if np.sum(binary_relevances) == 0:
        return 0.0

    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_binary_relevances = binary_relevances[sorted_indices]
    

    precisions = []
    num_relevant = 0
    
    for i, is_relevant in enumerate(sorted_binary_relevances):
        if is_relevant:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precisions.append(precision_at_i)
    
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)

def evaluate_model(model_info: Dict[str, Any], features: np.ndarray, labels: np.ndarray, 
                  query_ids: np.ndarray) -> Dict[str, float]:

    model_type = model_info['model_type']
    model = model_info['model']
    
    print(f"Evaluating {model_type} model...")

    if model_type == 'pointwise':
        predictions = model.predict(features)
    
    elif model_type == 'pairwise':
        scaler = model_info['scaler']
        scaled_features = scaler.transform(features)
        predictions = model.decision_function(scaled_features)
    
    elif model_type == 'listwise':
        dtest = xgb.DMatrix(features)
        predictions = model.predict(dtest)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

    unique_queries = np.unique(query_ids)
    
    ndcg_5_scores = []
    ndcg_10_scores = []
    map_scores = []
    
    for qid in unique_queries:
        query_mask = query_ids == qid
        query_labels = labels[query_mask]
        query_predictions = predictions[query_mask]
        
        if len(query_labels) == 0:
            continue
        

        ndcg_5 = calculate_ndcg(query_labels, query_predictions, k=5)
        ndcg_10 = calculate_ndcg(query_labels, query_predictions, k=10)

        ap = calculate_average_precision(query_labels, query_predictions)
        
        ndcg_5_scores.append(ndcg_5)
        ndcg_10_scores.append(ndcg_10)
        map_scores.append(ap)

    mean_ndcg_5 = np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0
    mean_ndcg_10 = np.mean(ndcg_10_scores) if ndcg_10_scores else 0.0
    mean_ap = np.mean(map_scores) if map_scores else 0.0
    
    results = {
        'nDCG@5': mean_ndcg_5,
        'nDCG@10': mean_ndcg_10,
        'MAP': mean_ap,
        'num_queries': len(unique_queries)
    }
    
    return results

def analyze_performance_by_relevance(labels: np.ndarray, query_ids: np.ndarray) -> None:

    print("\n=== ANALISIS DE RENDIMIENTO POR RELEVANCIA ===")

    unique_queries = np.unique(query_ids)
    
    relevant_counts = []
    total_counts = []
    
    for qid in unique_queries:
        query_mask = query_ids == qid
        query_labels = labels[query_mask]
        
        num_relevant = np.sum(query_labels > 0)
        num_total = len(query_labels)
        
        relevant_counts.append(num_relevant)
        total_counts.append(num_total)
    
    relevant_counts = np.array(relevant_counts)
    total_counts = np.array(total_counts)
    
    print(f"Numero de queries: {len(unique_queries)}")
    print(f"Documentos por queries - Min: {np.min(total_counts)}, Max: {np.max(total_counts)}, Mean: {np.mean(total_counts):.2f}")
    print(f"Documentos relevantes por queries - Min: {np.min(relevant_counts)}, Max: {np.max(relevant_counts)}, Mean: {np.mean(relevant_counts):.2f}")


    low_relevant = np.sum(relevant_counts <= 2)
    medium_relevant = np.sum((relevant_counts > 2) & (relevant_counts <= 10))
    high_relevant = np.sum(relevant_counts > 10)
    
    print(f"Queries con menos de 2 documentos relevantes: {low_relevant} ({low_relevant/len(unique_queries)*100:.1f}%)")
    print(f"Queries con 3-10 documentos relevantes: {medium_relevant} ({medium_relevant/len(unique_queries)*100:.1f}%)")
    print(f"Queries con más de 10 documentos relevantes: {high_relevant} ({high_relevant/len(unique_queries)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Evaluar Learning to Rank models')
    parser.add_argument('validation_file', type=str, help='Donde esta el archivo de validacion')
    parser.add_argument('--model_dir', type=str, default='.', 
                       help='Directorio que contiene los modelos entrenados (default: directorio actual)')
    
    args = parser.parse_args()

    print("=== Learning to Rank - Script de Evaluación ===")
    print(f"Archivo de validación: {args.validation_file}")
    print(f"Directorio de modelos: {args.model_dir}")
    

    try:
        labels, query_ids, features = load_letor_data(args.validation_file)
        print(f"Cargado {len(labels)} muestras de validación con {features.shape[1]} características")
        print(f"úmero de consultas únicas: {len(np.unique(query_ids))}")
        print(f"Distribución de etiquetas: {np.bincount(labels)}")
    except Exception as e:
        print(f"Error : {e}")
        sys.exit(1)

    analyze_performance_by_relevance(labels, query_ids)

    results = {}

    try:
        pointwise_info = joblib.load(f"{args.model_dir}/pointwise_model.joblib")
        pointwise_results = evaluate_model(pointwise_info, features, labels, query_ids)
        results['Pointwise'] = pointwise_results
        print("✓ modelo pointwise evaluado con éxito")
    except Exception as e:
        print(f"Error : {e}")
        results['Pointwise'] = None
    

    try:
        pairwise_info = joblib.load(f"{args.model_dir}/pairwise_model.joblib")
        pairwise_results = evaluate_model(pairwise_info, features, labels, query_ids)
        results['Pairwise'] = pairwise_results
        print("✓ modelo pairwise evaluado con éxito")
    except Exception as e:
        print(f"Error : {e}")
        results['Pairwise'] = None
    
    try:
        listwise_info = joblib.load(f"{args.model_dir}/listwise_model.joblib")
        listwise_results = evaluate_model(listwise_info, features, labels, query_ids)
        results['Listwise'] = listwise_results
        print("✓ modelo listwise evaluado con éxito")
    except Exception as e:
        print(f"Error : {e}")
        results['Listwise'] = None

    print("\n" + "="*60)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("="*60)

    print(f"{'Modelo':<15} {'nDCG@5':<10} {'nDCG@10':<10} {'MAP':<10} {'Consultas':<10}")
    print("-" * 60)
    
    for model_name, model_results in results.items():
        if model_results is not None:
            print(f"{model_name:<15} {model_results['nDCG@5']:<10.4f} {model_results['nDCG@10']:<10.4f} "
                  f"{model_results['MAP']:<10.4f} {model_results['num_queries']:<10}")
        else:
            print(f"{model_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:

        best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['nDCG@10'])
        print(f"\nMejor modelo: {best_model} (nDCG@10 = {valid_results[best_model]['nDCG@10']:.4f})")

        print("\n=== Análisis de Rendimiento ===")
        print("Comparación de Modelos:")
        for model_name, model_results in valid_results.items():
            print(f"\n{model_name}:")
            print(f"  - nDCG@5:  {model_results['nDCG@5']:.4f}")
            print(f"  - nDCG@10: {model_results['nDCG@10']:.4f}")
            print(f"  - MAP:     {model_results['MAP']:.4f}")

    print("\n=== Evaluación completada ===")

if __name__ == "__main__":
    main()
