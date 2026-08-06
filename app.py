"""Aplicação web simples para visualizar os resultados."""
import pandas as pd
import streamlit as st

from avaliacao import avaliar
from prep import carregar, janelar, SINAIS

st.set_page_config(page_title="Projeto Manutenção Preditiva", layout="wide")
st.title("Visualização do Pipeline")


df = carregar()
janelas = janelar(df)

feature_stats = ["median", "std", "slope", "range", "iqr"]
feature_names = [f"{stat}_{sinal}" for sinal in SINAIS for stat in feature_stats]

st.subheader("Resumo do pipeline")
st.info(
    "O fluxo atual inclui: carregamento ordenado, segmentação por rótulo bruto e intervalo, "
    "normalização de classes, seleção de 18 sinais e criação de janelas de 30 amostras com passo 15."
)

st.write("**Diferença entre segmento e janela:**")
st.write("- **Segmento**: é um bloco maior da série, formado por uma sequência contínua de amostras com o mesmo contexto ou mesma falha.")
st.write("- **Janela**: é uma fatia menor dentro do segmento, usada para extrair características locais e alimentar o modelo.")
st.write("- Em resumo: o segmento define o contexto; a janela é a unidade de análise que entra no modelo.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Linhas processadas", df.shape[0])
col2.metric("Classes finais", int(df["classe"].nunique()))
col3.metric("Segmentos", int(df["segment_id"].nunique()))
col4.metric("Janelas", int(janelas.shape[0]))

st.subheader("Etapa 1 — Carregamento e ordenação")
st.success("O CSV é lido com datas parseadas e ordenado por created_at.")

st.subheader("Etapa 2 — Segmentação")
if "created_at" in df.columns:
    dt = df["created_at"].diff().dt.total_seconds()
    long_intervals = df.loc[dt.gt(3600), ["created_at", "classe"]].copy()
    long_intervals["intervalo_s"] = dt[dt.gt(3600)].tolist()

    st.write("Novos segmentos são abertos quando o rótulo bruto muda ou quando o intervalo entre amostras excede 1 hora.")
    st.metric("Mediana do intervalo", f"{dt.median():.2f} s")
    st.metric("Intervalos > 1h", int((dt > 3600).sum()))
    st.dataframe(long_intervals.head(20), use_container_width=True)

st.subheader("Etapa 3 — Normalização de rótulos")
st.success("Os rótulos brutos foram limpos e padronizados para formar as classes finais usadas pelo modelo.")
st.write("Nesta etapa, labels inconsistentes, variantes escritas de forma diferente e rótulos irrelevantes foram unificados.")
st.write("Exemplos comuns incluem: 'desbalanceado', 'desabalanceado' e 'desbanlanceado' → 'desbalanceado'; 'baseline' e 'normal' passam a ser tratados como 'normal'.")
st.write("Rótulos que não representam falha útil, como 'teste' ou 'acelerando', são removidos da análise.")
st.write("O objetivo é reduzir ruído e garantir que o modelo veja uma mesma classe com o mesmo nome.")

st.subheader("Etapa 4 — Seleção de sinais")
st.write(f"Foram mantidos os {len(SINAIS)} sinais da lista SINAIS.")
st.dataframe(pd.DataFrame({"sinal": SINAIS}), use_container_width=True)

st.markdown("### Explorar falhas por segmento")
if "classe" in df.columns and "segment_id" in df.columns:
    falhas = sorted(df["classe"].dropna().unique().tolist())
    falha_selecionada = st.selectbox("Selecione a falha", falhas)

    segmentos = (
        df.loc[df["classe"] == falha_selecionada, ["segment_id", "classe"]]
        .drop_duplicates()
        .sort_values("segment_id")
    )

    st.metric("Quantidade de segmentos", len(segmentos))
    st.dataframe(segmentos, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Expandir todos os segmentos"):
            st.session_state["expand_all_segments"] = True
    with col2:
        if st.button("Recolher todos os segmentos"):
            st.session_state["expand_all_segments"] = False

    if "expand_all_segments" not in st.session_state:
        st.session_state["expand_all_segments"] = True

    colunas_disponiveis = [col for col in SINAIS if col in df.columns]
    colunas_selecionadas = st.multiselect(
        "Selecione as colunas para visualizar como série temporal",
        options=colunas_disponiveis,
        default=colunas_disponiveis[:3],
    )

    st.caption("Este gráfico mostra a evolução das séries temporais brutas do segmento selecionado. Ele é uma inspeção visual do comportamento do sinal, não a entrada direta do modelo.")

    for _, row in segmentos.iterrows():
        segment_id = int(row["segment_id"])
        with st.expander(f"Segmento {segment_id}", expanded=st.session_state["expand_all_segments"]):
            trecho = df.loc[df["segment_id"] == segment_id].copy()
            trecho = trecho.sort_values("created_at") if "created_at" in trecho.columns else trecho.reset_index(drop=True)
            st.caption(f"{len(trecho)} linhas de amostra")

            if not trecho.empty and colunas_selecionadas:
                eixo = "created_at" if "created_at" in trecho.columns else None
                dados_plot = trecho[colunas_selecionadas].copy()
                if eixo is not None:
                    dados_plot.index = trecho[eixo]
                st.line_chart(dados_plot)

    resumo_segmentos = (
        df.groupby(["classe", "segment_id"])
        .size()
        .reset_index(name="linhas")
        .groupby("classe")
        .agg(segmentos=("segment_id", "nunique"), linhas=("linhas", "sum"))
        .reset_index()
        .sort_values("segmentos", ascending=False)
    )

    st.bar_chart(resumo_segmentos.set_index("classe")["segmentos"])

st.subheader("Etapa 5 — Janelamento")
st.write("Cada janela contém 30 amostras, com sobreposição de 50% e sem cruzar a fronteira de um segmento.")
st.dataframe(janelas.head(10), use_container_width=True)

st.subheader("Transformação aplicada antes do modelo")
st.write("Aqui, 'série temporal' significa o trecho de sinais de um segmento ao longo do tempo. O segmento é o bloco maior; a janela é uma fatia menor dentro dele.")
st.write("Cada segmento é dividido em janelas de 30 amostras. Para cada janela, calculamos 5 estatísticas para cada um dos 18 sinais selecionados: mediana, desvio padrão, inclinação, amplitude e IQR.")
st.write("Como cada janela gera 18 sinais × 5 métricas, o resultado é um vetor com 90 features para cada janela.")
st.write("Ou seja: entrada da janela → 30 amostras × 18 sinais; saída para o modelo → 90 valores numéricos.")

st.info("Representação do problema: uma janela bruta vira um vetor de features e essa representação recebe uma classe alvo.")

if not janelas.empty:
    exemplo = janelas.iloc[0]
    col_a, col_b, col_c = st.columns([1.6, 2.0, 1.2])
    with col_a:
        st.subheader("Antes da transformação estatística")
        st.caption("Entrada bruta: uma janela de 30 amostras")
        st.dataframe(exemplo["janela_bruta"], use_container_width=True)
    with col_b:
        st.subheader("Depois da transformação estatística")
        st.caption("Saída para o modelo: 90 features")
        st.dataframe(
            pd.DataFrame({"feature": feature_names, "valor": exemplo["features"]}),
            use_container_width=True,
        )
    with col_c:
        st.subheader("Classe alvo")
        st.caption("Rótulo usado para supervisionar o modelo")
        st.metric("Classe", exemplo["classe"])
        st.metric("Segmento", int(exemplo["segment_id"]))

st.write("**Estrutura do dataset usado pelo modelo:**")
st.write("- cada linha = uma janela")
st.write("- colunas de entrada = as 90 features")
st.write("- coluna alvo = a classe do segmento")

st.dataframe(
    pd.DataFrame({"feature": feature_names}),
    use_container_width=True,
)
st.caption("Cada linha acima representa uma coluna de entrada do modelo. O vetor completo tem 90 colunas.")

st.subheader("Modelos treinados")
st.write("Modelo usado: Random Forest Classifier.")
st.write("Parâmetros principais: n_estimators=400, random_state=0, n_jobs=-1.")
st.write("Estratégias de validação: StratifiedKFold e StratifiedGroupKFold.")
st.write("A validação por segmento é importante porque evita vazamento entre janelas do mesmo segmento.")

st.write("**O que é um fold?**")
st.write("Fold é uma parte da validação cruzada. O conjunto de janelas é dividido em grupos, e em cada rodada um grupo é usado para teste enquanto os demais são usados para treino.")
st.write("No caso do projeto, cada fold representa uma rodada de treinamento e teste da validação cruzada.")

st.subheader("Conjuntos de treino e teste")
if "classe" in df.columns and "segment_id" in df.columns:
    treino = pd.DataFrame(janelas)
    teste = pd.DataFrame(janelas)
    st.download_button(
        label="Baixar treino",
        data=treino.to_csv(index=False),
        file_name="treino.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Baixar teste",
        data=teste.to_csv(index=False),
        file_name="teste.csv",
        mime="text/csv",
    )

st.subheader("Resultados dos modelos")
resultados = avaliar()
modelos = [
    ("Random Forest", "RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)"),
]
for nome, descricao in modelos:
    st.subheader(nome)
    st.caption(descricao)

for resultado in resultados:
    st.markdown(f"**{resultado['nome']}**")
    st.metric("Acurácia média", f"{resultado['media']:.3f}")
    for fold in resultado["folds"]:
        with st.expander(f"Fold {fold['fold']} — treino {fold['train_size']} / teste {fold['test_size']}"):
            st.write(f"Acurácia: {fold['accuracy']:.3f}")
            st.caption("Base de teste com classe real e classe prevista")
            teste_df = pd.DataFrame(
                {
                    "classe_real": fold["y_true"],
                    "classe_prevista": fold["y_pred"],
                    "segment_id": fold["teste_janelas"]["segment_id"].tolist(),
                    "classe_original": fold["teste_janelas"]["classe"].tolist(),
                }
            )
            st.dataframe(teste_df.head(20), use_container_width=True)
            st.caption("Os valores acima mostram um trecho da base de teste para este fold. O modelo prevê uma classe para cada janela e ela é comparada com a classe real.")

st.subheader("Pré-visualização do dataframe preparado")
st.dataframe(df.head(50), use_container_width=True)
