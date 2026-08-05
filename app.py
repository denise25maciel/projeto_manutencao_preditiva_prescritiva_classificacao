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

st.subheader("Descrição da etapa de segmentação")
st.info(
    "A segmentação foi feita antes da limpeza de rótulos. O novo segmento abre quando o \n"
    "rótulo bruto muda ou quando o intervalo entre duas amostras ultrapassa 1 hora."
)

if "created_at" in df.columns:
    dt = df["created_at"].diff().dt.total_seconds()
    long_intervals = df.loc[dt.gt(3600), ["created_at", "fault"]].copy()
    long_intervals["intervalo_s"] = dt[dt.gt(3600)].tolist()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mediana do intervalo", f"{dt.median():.2f} s")
    col2.metric("Intervalos > 1h", int((dt > 3600).sum()))
    col3.metric("Segmentos brutos", int(df["segment_id"].nunique()))

    st.subheader("Intervalos reconhecidos")
    st.dataframe(long_intervals.head(20), use_container_width=True)

st.subheader("Segmentos agrupados")
segment_summary = (
    df.groupby("segment_id")["fault"].agg(["first", "nunique", "count"]).reset_index().head(20)
)
segment_summary.columns = ["segment_id", "rótulo_inicial", "quantidade_rótulos_distintos", "total_linhas"]
st.dataframe(segment_summary, use_container_width=True)

st.subheader("Distribuição por fault")
if "fault" in df.columns:
    counts = df["fault"].value_counts().head(10)
    st.bar_chart(counts)

st.subheader("Pré-visualização")
st.dataframe(df.head(50), use_container_width=True)
