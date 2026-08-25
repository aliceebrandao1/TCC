"""
Pipeline Unificado: Redução de Multicolinearidade + Análise de Features
Baseado no protocolo de Son Gyo Jung et al.

Ordem de execução:
  1. Remoção de features constantes (VarianceThreshold)
  2. Filtro de Correlação de Pearson (threshold = 0.85)
  3. Clusterização Hierárquica (Spearman + Ward Linkage)
  4. Análise de Importância: F-Test (correlação linear)
  5. Análise de Importância: Mutual Information (correlação não-linear)
  6. Ranking final das features mais relevantes
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


def main():
    df = pd.read_csv('data/c2db_featurizado_artigo.csv')
    target = 'hform'
    features = [col for col in df.columns if col not in [target, 'formula']]
    print(f"Features iniciais extraídas do Matminer: {len(features)}")

    # ETAPA 1: Remoção de Features Constantes (VarianceThreshold)
    print("\n--- ETAPA 1: Remoção de features constantes ---")
    X = df[features].fillna(0)
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X)
    features_v1 = X.columns[vt.get_support()].tolist()
    removidas_const = len(features) - len(features_v1)
    print(f"Features removidas (sem variação): {removidas_const}")
    print(f"Features restantes: {len(features_v1)}")

    # ETAPA 2: Filtro de Correlação de Pearson
    print("\n--- ETAPA 2: Filtro de Correlação de Pearson ---")
    threshold_pearson = 0.85
    corr_matrix = df[features_v1].corr().abs()

    col_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold_pearson:
                col_corr.add(corr_matrix.columns[i])

    features_v2 = [f for f in features_v1 if f not in col_corr]
    print(f"Features removidas (correlação > {threshold_pearson}): {len(col_corr)}")
    print(f"Features restantes: {len(features_v2)}")

    # ETAPA 3: Clusterização Hierárquica (Spearman + Ward Linkage)
    print("\n--- ETAPA 3: Clusterização Hierárquica (Spearman + Ward) ---")
    print("Calculando correlação de Spearman...")

    corr_spearman = spearmanr(df[features_v2].fillna(0)).correlation
    corr_spearman[np.isnan(corr_spearman)] = 0
    corr_linkage = hierarchy.ward(corr_spearman)

    os.makedirs('results/figures', exist_ok=True)
    plt.figure(figsize=(15, 8))
    hierarchy.dendrogram(
        corr_linkage, labels=features_v2,
        orientation='top', leaf_rotation=90, leaf_font_size=8
    )
    plt.title("Dendrograma de Features (Distância de Ward)")
    plt.ylabel("Distância")
    plt.tight_layout()
    plt.savefig('results/figures/dendrograma_features.png', dpi=300)
    plt.close()
    print("Dendrograma salvo em: results/figures/dendrograma_features.png")

    threshold_linkage = 1.5
    cluster_ids = hierarchy.fcluster(corr_linkage, t=threshold_linkage, criterion='distance')

    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    selected_indices = [value[0] for value in cluster_id_to_feature_ids.values()]
    features_v3 = [features_v2[i] for i in selected_indices]
    print(f"Features finais após clustering (threshold {threshold_linkage}): {len(features_v3)}")

    os.makedirs('data', exist_ok=True)
    colunas_finais = ['formula', target] + features_v3
    df_limpo = df[colunas_finais]
    df_limpo.to_csv('data/c2db_features_selecionadas.csv', index=False)
    print(f"Dataset limpo salvo em: data/c2db_features_selecionadas.csv")

    # ETAPA 4: Análise F-Test (Correlação Linear) — Pós-Multicolinearidade
    print("\n--- ETAPA 4: F-Test (Correlação Linear) ---")
    X_limpo = df_limpo[features_v3].fillna(0)
    y = df_limpo[target].fillna(0)

    sel_f = SelectKBest(f_regression, k='all')
    sel_f.fit(X_limpo, y)

    df_f_test = pd.DataFrame({'feature': features_v3, 'f_score': sel_f.scores_})
    df_f_test = df_f_test.dropna().sort_values(by='f_score', ascending=False)

    scaler = MinMaxScaler()
    df_f_test['f_score_scaled'] = scaler.fit_transform(df_f_test[['f_score']])

    os.makedirs('results/feature_analysis', exist_ok=True)
    df_f_test.to_csv('results/feature_analysis/f_test_result_hform.csv', index=False)
    print("Resultado salvo em: results/feature_analysis/f_test_result_hform.csv")

    print("\nTop 10 Features (Relação Linear - F-Test):")
    print(df_f_test[['feature', 'f_score_scaled']].head(10).to_string(index=True))

    # ETAPA 5: Ranqueamento Final (Mutual Information Puro)
    print("\n--- ETAPA 5: Ranqueamento Final (Mutual Information) ---")
    print("(Calculando a redução de entropia física... isso pode demorar alguns segundos)")

    X_final = df_limpo[features_v3].fillna(0)
    y_final = df_limpo[target].fillna(0)

    # Calcula os valores absolutos reais (em nats) sem distorções de escala
    mi_scores_brutos = mutual_info_regression(X_final, y_final)

    df_ranking = pd.DataFrame({
        'feature': features_v3, 
        'mi_score_bruto': mi_scores_brutos
    })
    
    # Ordena com base puramente na teoria da informação
    df_ranking = df_ranking.sort_values(by='mi_score_bruto', ascending=False).reset_index(drop=True)

    os.makedirs('results/feature_analysis', exist_ok=True)
    df_ranking.to_csv('results/feature_analysis/ranking_features_finais.csv', index=False)

    print("\nTop 10 Features Mais Importantes (Sem MinMaxScaler):")
    print(df_ranking.head(10).to_string(index=True))
    print("\nRanking completo salvo em: results/feature_analysis/ranking_features_finais.csv")


    print("\n" + "="*65)
    print(" RESUMO DO PIPELINE")
    print("="*65)
    print(f"  Features iniciais (Matminer):          {len(features)}")
    print(f"  Após remover constantes:               {len(features_v1)} (-{removidas_const})")
    print(f"  Após Filtro de Pearson (>{threshold_pearson}):       {len(features_v2)} (-{len(col_corr)})")
    print(f"  Features Finais Independentes (t={threshold_linkage}): {len(features_v3)}")
    print("-" * 65)
    print(f"  Feature Mais Importante (Alvo: {target}):")
    print(f"  -> {df_ranking.iloc[0]['feature']} (Score: {df_ranking.iloc[0]['mi_score_bruto']:.4f} nats)")
    print("="*65)

if __name__ == "__main__":
    main()