"""Aplicação web simples para visualizar os resultados."""
import streamlit as st

from prep import carregar

st.set_page_config(page_title="Projeto Manutenção Preditiva", layout="wide")
st.title("Visualização do Dataset")


df = carregar()

st.metric("Linhas", df.shape[0])
st.metric("Colunas", df.shape[1])
st.metric("Valores ausentes", int(df.isna().sum().sum()))
st.metric("Ordenado por created_at", "Sim" if df["created_at"].is_monotonic_increasing else "Não")

st.subheader("Distribuição por fault")
if "fault" in df.columns:
    counts = df["fault"].value_counts().head(10)
    st.bar_chart(counts)

st.subheader("Pré-visualização")
st.dataframe(df.head(50), use_container_width=True)
