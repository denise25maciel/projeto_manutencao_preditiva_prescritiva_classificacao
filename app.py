"""Interface didática do pipeline de manutenção preditiva.

Cada seção explica um passo em linguagem simples e mostra o dataset resultante
daquela transformação.
"""
import numpy as np
import pandas as pd
import streamlit as st

from avaliacao import avaliar
from otimizacao import PARAMS_JSON, carregar_melhores_params
from prep import (
    JANELA_PASSO,
    JANELA_TAMANHO,
    SINAIS,
    BANNER_CSV,
    classe_base,
    criar_amostras,
    preparar_dataframe,
)

st.set_page_config(page_title="Manutenção Preditiva — Passo a Passo", layout="wide")

ESTATISTICAS = ["mediana", "desvio_padrao", "inclinacao", "amplitude", "iqr"]
NOMES_FEATURES = [f"{stat}__{sinal}" for sinal in SINAIS for stat in ESTATISTICAS]


# ---------------------------------------------------------------------------
# Carga dos dados (em cache: só roda uma vez por sessão)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Lendo o arquivo de dados...")
def carregar_bruto():
    return pd.read_csv(BANNER_CSV, parse_dates=["created_at"])


@st.cache_data(show_spinner="Aplicando as transformações...")
def carregar_estagios():
    """Roda o pipeline guardando o que mudou em cada etapa."""
    bruto = carregar_bruto()

    # Etapa 1 — ordenar no tempo
    ordenado = bruto.sort_values("created_at").reset_index(drop=True)

    # Etapa 2 — separar em segmentos
    dt = ordenado["created_at"].diff().dt.total_seconds()
    mudou_rotulo = ordenado["fault"].ne(ordenado["fault"].shift())
    novo_segmento = mudou_rotulo | dt.gt(3600)
    com_segmento = ordenado.copy()
    com_segmento["segment_id"] = novo_segmento.cumsum()

    # Etapa 3 — limpar os nomes das falhas
    com_classe = com_segmento.copy()
    com_classe["classe"] = com_classe["fault"].apply(classe_base)
    limpo = com_classe.dropna(subset=["classe"]).reset_index(drop=True)

    # Etapa 4 — escolher os sinais (resultado final)
    final = preparar_dataframe(bruto)

    return {
        "bruto": bruto,
        "ordenado": ordenado,
        "intervalos": dt,
        "com_segmento": com_segmento,
        "com_classe": com_classe,
        "limpo": limpo,
        "final": final,
    }


@st.cache_data(show_spinner="Montando as amostras...")
def montar_amostras(modo):
    return criar_amostras(carregar_estagios()["final"], modo=modo)


@st.cache_data(show_spinner="Treinando os modelos... isso leva alguns minutos.")
def rodar_avaliacao(modo, params_serializados):
    params = dict(params_serializados) if params_serializados else None
    return avaliar(modo=modo, params=params)


def caixa_dataset(df, titulo, legenda="", linhas=10):
    """Mostra um dataset com suas dimensões."""
    st.markdown(f"**{titulo}**")
    if legenda:
        st.caption(legenda)
    col1, col2 = st.columns(2)
    col1.metric("Linhas", f"{len(df):,}".replace(",", "."))
    col2.metric("Colunas", df.shape[1])
    st.dataframe(df.head(linhas), width="stretch")


# ---------------------------------------------------------------------------
# Navegação
# ---------------------------------------------------------------------------

SECOES = [
    "Início — o que é este projeto",
    "Os dados brutos",
    "Passo 1 — Colocar em ordem de tempo",
    "Passo 2 — Separar em blocos (segmentos)",
    "Passo 3 — Limpar os nomes das falhas",
    "Passo 4 — Escolher quais medidas usar",
    "Passo 5 — Fatiar em janelas",
    "Passo 6 — Resumir cada janela em números",
    "Passo 7 — O modelo que aprende",
    "Passo 8 — Como sabemos se funciona",
    "Experimento — tamanho da janela",
    "Optuna — o que aquela mensagem queria dizer",
    "Resumo final e limitações",
]

st.sidebar.title("Roteiro")
st.sidebar.caption("Percorra na ordem. Cada passo mostra o dataset que ele produz.")
secao = st.sidebar.radio("Ir para:", SECOES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(
    f"Configuração atual\n\n"
    f"- Janela: **{JANELA_TAMANHO} leituras**\n"
    f"- Passo: **{JANELA_PASSO}**\n"
    f"- Sinais: **{len(SINAIS)}**\n"
    f"- Features: **{len(NOMES_FEATURES)}**"
)


# ---------------------------------------------------------------------------
# 0. Início
# ---------------------------------------------------------------------------

if secao == SECOES[0]:
    st.title("Prever falhas em um motor antes que elas parem a produção")

    st.markdown(
        """
        ### A ideia, em uma frase

        Um motor com defeito **vibra diferente** de um motor saudável. Se conseguirmos ler essa
        vibração e reconhecer o padrão, dá para dizer qual peça está com problema — antes que
        ela quebre de vez.

        ### A analogia

        Pense em um mecânico experiente. Ele encosta a mão no motor, escuta o ruído e diz:
        *"isso aí é rolamento"*. Ele não mediu nada — ele reconhece o padrão porque já ouviu
        centenas de motores.

        Este projeto tenta fazer a mesma coisa, só que com um sensor no lugar da mão e um
        programa de computador no lugar da experiência do mecânico.

        ### O que temos

        Um sensor preso ao motor mede vibração, temperatura e rotação, **a cada 2 segundos**.
        Durante os ensaios, alguém anotou qual defeito estava instalado no motor naquele momento.

        Então temos duas coisas:

        - **as medições** — os números do sensor;
        - **a resposta certa** — qual era o defeito.

        O computador estuda os dois juntos e tenta aprender a ligação entre eles.

        ### O caminho até lá

        Os dados do sensor não servem do jeito que chegam. Eles passam por 6 transformações antes
        de o computador conseguir aprender:
        """
    )

    st.markdown(
        """
        | Passo | O que faz | Por que precisa |
        |---|---|---|
        | 1 | Coloca as leituras em ordem de tempo | Sem isso, "antes" e "depois" ficam embaralhados |
        | 2 | Separa em blocos (segmentos) | Cada ensaio precisa ser tratado separadamente |
        | 3 | Limpa os nomes das falhas | Foram digitados à mão e vieram com erros |
        | 4 | Escolhe quais medidas usar | Algumas colunas são a mesma coisa repetida |
        | 5 | Fatia em pedaços de tamanho igual | Todos os exemplos precisam ter o mesmo formato |
        | 6 | Resume cada pedaço em números | O computador não lê gráfico, lê números |
        """
    )

    st.info(
        "Cada passo tem uma seção própria no menu à esquerda, com o dataset **antes** e "
        "**depois** da transformação. Você vê exatamente o que mudou."
    )

    st.markdown(
        """
        ### E funciona?

        Em parte. O sistema acerta **44% das vezes** quando testado em um motor/ensaio que ele
        nunca viu. Como existem 13 tipos de falha possíveis, chutar acertaria uns 8%. Então ele
        aprendeu alguma coisa real — mas está longe de ser confiável.

        A seção **Passo 8** explica de onde vem esse 44%, e por que existe outro número (92%) que
        parece muito melhor e **não** deve ser usado.
        """
    )


# ---------------------------------------------------------------------------
# 1. Dados brutos
# ---------------------------------------------------------------------------

elif secao == SECOES[1]:
    st.title("Os dados brutos")
    st.caption("O que sai do sensor, antes de qualquer tratamento.")

    estagios = carregar_estagios()
    bruto = estagios["bruto"]

    st.markdown(
        """
        ### Como ler esta tabela

        Cada **linha** é uma leitura do sensor em um instante. Cada **coluna** é uma coisa medida
        naquele instante.

        A tabela abaixo é o arquivo como ele chegou — nada foi alterado ainda.
        """
    )

    caixa_dataset(bruto, "Dataset bruto", "As 10 primeiras linhas do arquivo original.")

    st.markdown("### O que significa cada coluna")

    significados = {
        "id": "Número da linha. Não serve para nada além de identificar.",
        "created_at": "Data e hora exata da leitura.",
        "fault": "O defeito que estava instalado no motor. Digitado à mão por uma pessoa.",
        "rpm": "Rotações por minuto — a velocidade de giro do motor.",
        "temperature_c": "Temperatura em graus Celsius.",
        "temperature_f": "A mesma temperatura, em Fahrenheit.",
    }

    descricoes = []
    for coluna in bruto.columns:
        if coluna in significados:
            texto = significados[coluna]
        elif "velocity" in coluna:
            texto = "Velocidade da vibração — o quanto o motor balança."
        elif "acceleration" in coluna or "accel" in coluna:
            texto = "Aceleração da vibração — o quanto o balanço muda de intensidade."
        elif "freq" in coluna:
            texto = "Frequência da vibração — quantas vezes por segundo o motor balança."
        elif "kurtosis" in coluna:
            texto = "Curtose — indica se há picos bruscos no sinal (típico de rolamento ruim)."
        elif "crest" in coluna:
            texto = "Fator de crista — compara o pico da vibração com o valor médio."
        else:
            texto = "Medida de vibração."

        eixo = ""
        if coluna.startswith("x_"):
            eixo = " (eixo horizontal)"
        elif coluna.startswith("z_"):
            eixo = " (eixo vertical)"

        descricoes.append({"coluna": coluna, "o que é": texto + eixo})

    st.dataframe(pd.DataFrame(descricoes), width="stretch", height=400)

    st.markdown(
        """
        ### Duas observações importantes

        **1. Tem medida repetida.** Repare em `temperature_c` e `temperature_f`: é a mesma
        temperatura em duas unidades. O mesmo vale para as velocidades em `_in_s` (polegadas) e
        `_mm_s` (milímetros). Isso vai ser resolvido no Passo 4.

        **2. A coluna `fault` é bagunçada.** Ela foi preenchida à mão durante os ensaios, e tem
        erro de digitação, abreviação e variação de escrita. Isso vai ser resolvido no Passo 3.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de leituras", f"{len(bruto):,}".replace(",", "."))
        st.metric("Colunas", bruto.shape[1])
    with col2:
        periodo = bruto["created_at"].max() - bruto["created_at"].min()
        st.metric("Período coberto", f"{periodo.days} dias")
        st.metric("Textos diferentes em 'fault'", bruto["fault"].nunique())


# ---------------------------------------------------------------------------
# 2. Passo 1 — Ordenar
# ---------------------------------------------------------------------------

elif secao == SECOES[2]:
    st.title("Passo 1 — Colocar em ordem de tempo")

    estagios = carregar_estagios()
    bruto, ordenado = estagios["bruto"], estagios["ordenado"]

    st.markdown(
        """
        ### O que este passo faz

        Reorganiza as linhas para que a leitura mais antiga fique em cima e a mais recente
        embaixo.

        ### Por que isso importa

        Imagine as páginas de um diário fora de ordem. Você consegue ler cada página, mas não
        consegue entender a **história** — o que veio antes, o que veio depois, o que estava
        piorando.

        Os passos seguintes todos dependem da ordem:

        - o Passo 2 precisa saber quando houve uma pausa longa entre duas leituras;
        - o Passo 6 calcula se a vibração estava **subindo ou descendo** — e isso só faz sentido
          se as leituras estiverem na ordem certa.
        """
    )

    st.divider()
    st.markdown("### O dataset antes e depois")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ANTES** — ordem do arquivo")
        st.dataframe(
            bruto[["id", "created_at", "fault"]].head(8),
            width="stretch",
        )
        st.caption(
            "Já estava ordenado? "
            + ("Sim." if bruto["created_at"].is_monotonic_increasing else "Não.")
        )
    with col2:
        st.markdown("**DEPOIS** — ordem cronológica")
        st.dataframe(
            ordenado[["id", "created_at", "fault"]].head(8),
            width="stretch",
        )
        st.caption("Garantidamente em ordem crescente de tempo.")

    st.success(
        f"Nenhuma linha foi criada ou apagada neste passo. "
        f"Continuam {len(ordenado):,}".replace(",", ".") + " leituras — só mudaram de lugar."
    )


# ---------------------------------------------------------------------------
# 3. Passo 2 — Segmentos
# ---------------------------------------------------------------------------

elif secao == SECOES[3]:
    st.title("Passo 2 — Separar em blocos (segmentos)")

    estagios = carregar_estagios()
    ordenado, com_segmento, dt = (
        estagios["ordenado"],
        estagios["com_segmento"],
        estagios["intervalos"],
    )

    st.markdown(
        """
        ### O que é um segmento

        Um **segmento** é um ensaio inteiro: o período contínuo em que o motor rodou com um
        defeito específico instalado, do começo ao fim.

        ### A analogia

        Pense em um arquivo de música com várias faixas gravadas em sequência, sem separação.
        Você precisa marcar onde cada faixa começa e termina. Um segmento é uma faixa.

        ### Como o programa descobre onde uma faixa termina

        Ele abre um bloco novo quando acontece **uma** destas duas coisas:

        **1. O defeito anotado mudou.** Se a coluna `fault` era `rolamento_inner` e passou a ser
        `desbalanceado`, alguém trocou a peça. É outro ensaio.

        **2. O sensor ficou mais de 1 hora sem registrar nada.** Isso significa que o
        equipamento foi desligado. Mesmo que o defeito anotado seja o mesmo, o motor foi
        desmontado e remontado no meio — as condições mudaram.
        """
    )

    st.divider()
    st.markdown("### Os intervalos entre leituras")

    col1, col2, col3 = st.columns(3)
    col1.metric("Intervalo típico", f"{dt.median():.1f} segundos")
    col2.metric("Pausas maiores que 1 hora", int((dt > 3600).sum()))
    col3.metric("Maior pausa", f"{dt.max()/86400:.1f} dias")

    st.info(
        f"**Guarde este número: {dt.median():.0f} segundos entre uma leitura e outra.** "
        "Ele reaparece no Passo 5, quando for preciso converter 'tamanho da janela' em "
        "'quantos minutos de motor'."
    )

    pausas = ordenado.loc[dt.gt(3600), ["created_at", "fault"]].copy()
    pausas["pausa"] = dt[dt.gt(3600)].apply(
        lambda s: f"{s/3600:.1f} horas" if s < 86400 else f"{s/86400:.1f} dias"
    )
    st.markdown("**As pausas longas encontradas** — cada uma abre um segmento novo:")
    st.dataframe(pausas.head(15), width="stretch")

    st.divider()
    st.markdown("### O dataset antes e depois")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ANTES**")
        st.dataframe(ordenado[["created_at", "fault"]].head(10), width="stretch")
        st.caption(f"{ordenado.shape[1]} colunas")
    with col2:
        st.markdown("**DEPOIS** — com a coluna `segment_id`")
        st.dataframe(
            com_segmento[["created_at", "fault", "segment_id"]].head(10), width="stretch"
        )
        st.caption(f"{com_segmento.shape[1]} colunas — ganhou `segment_id`")

    st.success(
        f"Nenhuma linha foi apagada. Foi acrescentada **uma coluna nova**, `segment_id`, que "
        f"diz a qual ensaio cada leitura pertence. Foram identificados "
        f"**{com_segmento['segment_id'].nunique()} segmentos**."
    )

    st.markdown(
        """
        ### Por que isso vai importar muito lá na frente

        Duas leituras do mesmo segmento são **muito parecidas** — mesmo motor, mesma montagem,
        poucos minutos de diferença. Elas quase não são exemplos independentes.

        Isso vira o problema central do projeto no **Passo 8**. Vale lembrar deste ponto quando
        chegar lá.
        """
    )


# ---------------------------------------------------------------------------
# 4. Passo 3 — Limpar rótulos
# ---------------------------------------------------------------------------

elif secao == SECOES[4]:
    st.title("Passo 3 — Limpar os nomes das falhas")

    estagios = carregar_estagios()
    com_segmento, com_classe, limpo = (
        estagios["com_segmento"],
        estagios["com_classe"],
        estagios["limpo"],
    )

    st.markdown(
        """
        ### O problema

        A coluna `fault` diz qual era o defeito. Ela foi digitada à mão, ensaio após ensaio, por
        pessoas diferentes. E ficou assim:
        """
    )

    exemplos_bagunca = (
        com_segmento["fault"]
        .value_counts()
        .reset_index()
        .rename(columns={"fault": "texto digitado", "count": "vezes"})
    )
    st.dataframe(exemplos_bagunca.head(20), width="stretch")

    st.markdown(
        f"""
        Existem **{com_segmento['fault'].nunique()} textos diferentes** nessa coluna. Mas não
        existem tantos defeitos assim — a maioria é o mesmo defeito escrito de outro jeito.

        ### Por que isso quebra o aprendizado

        Para o computador, `desbalanceado` e `desabalanceado` são duas coisas **completamente
        diferentes**, do mesmo jeito que "cachorro" e "guarda-chuva" são. Ele não sabe que foi
        erro de digitação.

        Resultado: os exemplos de um mesmo defeito ficam espalhados em várias categorias, e
        nenhuma delas tem exemplos suficientes para o computador aprender o padrão.

        ### Os três tipos de arrumação
        """
    )

    tab1, tab2, tab3 = st.tabs(["Erros de digitação", "Sinônimos", "Descartes"])

    with tab1:
        st.markdown("Textos diferentes que são o mesmo defeito. São unificados:")
        st.dataframe(
            pd.DataFrame(
                [
                    {"digitado": "desabalanceado", "vira": "desbalanceado"},
                    {"digitado": "desbanlanceado", "vira": "desbalanceado"},
                    {"digitado": "ddesbalanceado", "vira": "desbalanceado"},
                    {"digitado": "dedesbalanceado", "vira": "desbalanceado"},
                    {"digitado": "desabanceado", "vira": "desbalanceado"},
                    {"digitado": "normla", "vira": "normal"},
                    {"digitado": "mortor_desligado", "vira": "motor_desligado"},
                    {"digitado": "cockecocked", "vira": "cocked"},
                ]
            ),
            width="stretch",
        )

    with tab2:
        st.markdown(
            "Palavras diferentes para a mesma condição. `baseline` é o termo técnico para "
            "'motor de referência, sem defeito' — ou seja, motor **normal**:"
        )
        st.dataframe(
            pd.DataFrame([{"digitado": "baseline", "vira": "normal"}]), width="stretch"
        )
        st.markdown(
            "Também são removidos sufixos de numeração: `rolamento_inner_2`, "
            "`rolamento_inner_ensaio3` e `rolamento_inner` são todos o mesmo defeito, "
            "então todos viram `rolamento_inner`."
        )

    with tab3:
        st.markdown(
            "Anotações que não descrevem defeito nenhum. Essas linhas são **apagadas** do "
            "conjunto, porque não há o que aprender com elas:"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"digitado": "teste", "motivo": "É um teste do equipamento, não um ensaio."},
                    {"digitado": "acelerando", "motivo": "Descreve uma manobra, não um defeito."},
                    {"digitado": "new_tes", "motivo": "Anotação incompleta."},
                ]
            ),
            width="stretch",
        )

    st.divider()
    st.markdown("### O dataset antes e depois")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ANTES** — coluna `fault` crua")
        st.dataframe(com_segmento[["created_at", "fault"]].head(10), width="stretch")
        st.metric("Textos diferentes", com_segmento["fault"].nunique())
        st.metric("Linhas", f"{len(com_segmento):,}".replace(",", "."))
    with col2:
        st.markdown("**DEPOIS** — nova coluna `classe`")
        st.dataframe(limpo[["created_at", "fault", "classe"]].head(10), width="stretch")
        st.metric("Classes finais", limpo["classe"].nunique())
        st.metric("Linhas", f"{len(limpo):,}".replace(",", "."))

    removidas = len(com_segmento) - len(limpo)
    st.success(
        f"De **{com_segmento['fault'].nunique()} textos** bagunçados para "
        f"**{limpo['classe'].nunique()} classes** organizadas. "
        f"Foram apagadas **{removidas} linhas** que não descreviam defeito."
    )

    st.markdown("### Quantos exemplos temos de cada defeito")
    contagem = (
        limpo["classe"].value_counts().reset_index()
        .rename(columns={"classe": "defeito", "count": "leituras"})
    )
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.dataframe(contagem, width="stretch")
    with col2:
        st.bar_chart(contagem.set_index("defeito")["leituras"])

    st.warning(
        "Repare que alguns defeitos têm muito menos exemplos que outros. Isso vai causar "
        "problema no Passo 5 — os defeitos raros podem sumir completamente."
    )


# ---------------------------------------------------------------------------
# 5. Passo 4 — Selecionar sinais
# ---------------------------------------------------------------------------

elif secao == SECOES[5]:
    st.title("Passo 4 — Escolher quais medidas usar")

    estagios = carregar_estagios()
    limpo, final = estagios["limpo"], estagios["final"]

    descartadas_previa = [c for c in limpo.columns if c not in final.columns]

    st.markdown(
        f"""
        ### A ideia deste passo

        Nem tudo que veio no arquivo serve para o computador aprender. Aqui separamos o que
        entra e o que fica de fora.

        Das {limpo.shape[1]} colunas disponíveis, **{len(descartadas_previa)} foram retiradas** e
        sobraram {final.shape[1]} — sendo {len(SINAIS)} delas as medidas do sensor.

        ### Os três motivos de uma coluna sair

        **Motivo 1 — É a mesma coisa, escrita de outro jeito.**
        O arquivo traz a temperatura duas vezes: em Celsius e em Fahrenheit. E traz a vibração
        duas vezes: em milímetros e em polegadas. São 30 °C e 86 °F — o mesmo calor.
        Guardar os dois é como anotar seu peso em quilos e em libras e achar que você tem dois
        dados sobre si. Você tem um só.

        **Motivo 2 — Não fala nada sobre o motor.**
        A coluna `id` é só a numeração das linhas. Não mede nada.

        **Motivo 3 — Já foi substituída por uma versão arrumada.**
        A coluna `fault` (o defeito digitado à mão) virou a coluna `classe` no Passo 3. A versão
        bagunçada não é mais necessária.

        Abaixo, a justificativa de **cada coluna, uma por uma**.
        """
    )

    st.divider()
    st.markdown("### Cada coluna que saiu, e o motivo dela ter saído")

    descartadas = [c for c in limpo.columns if c not in final.columns]

    # Justificativa escrita uma a uma, sem regra automática
    JUSTIFICATIVAS = {
        "id": {
            "o que era": "Um número de ordem: 1, 2, 3, 4...",
            "por que saiu": (
                "É só a numeração das linhas, como o número da página de um livro. "
                "Não diz nada sobre o motor."
            ),
            "risco de manter": (
                "Alto. Os ensaios foram gravados em sequência, então os defeitos do começo "
                "têm número baixo e os do fim têm número alto. O computador perceberia isso e "
                "passaria a 'adivinhar' pelo número da linha, em vez de olhar a vibração. "
                "Funcionaria neste arquivo e falharia em qualquer motor novo."
            ),
            "substituída por": "—",
        },
        "temperature_f": {
            "o que era": "A temperatura do motor, em graus Fahrenheit.",
            "por que saiu": (
                "É a mesma temperatura que já temos em Celsius, só que em outra escala. "
                "30 °C e 86 °F são o mesmo calor."
            ),
            "risco de manter": (
                "Baixo, mas atrapalha. O computador olharia duas colunas para descobrir uma "
                "coisa só, e a informação da temperatura acabaria pesando o dobro do que deveria."
            ),
            "substituída por": "temperature_c",
        },
        "z_rms_velocity_in_s": {
            "o que era": "O quanto o motor balança na vertical, medido em polegadas por segundo.",
            "por que saiu": "Mesma medida que já temos em milímetros. 1 polegada = 25,4 mm.",
            "risco de manter": "Baixo, mas é informação repetida ocupando espaço.",
            "substituída por": "z_rms_velocity_mm_s",
        },
        "x_rms_velocity_in_s": {
            "o que era": "O quanto o motor balança na horizontal, em polegadas por segundo.",
            "por que saiu": "Mesma medida que já temos em milímetros.",
            "risco de manter": "Baixo, mas é informação repetida ocupando espaço.",
            "substituída por": "x_rms_velocity_mm_s",
        },
        "z_peak_velocity_in_s": {
            "o que era": "O balanço mais forte na vertical, em polegadas por segundo.",
            "por que saiu": "Mesma medida que já temos em milímetros.",
            "risco de manter": "Baixo, mas é informação repetida ocupando espaço.",
            "substituída por": "z_peak_velocity_mm_s",
        },
        "x_peak_velocity_in_s": {
            "o que era": "O balanço mais forte na horizontal, em polegadas por segundo.",
            "por que saiu": "Mesma medida que já temos em milímetros.",
            "risco de manter": "Baixo, mas é informação repetida ocupando espaço.",
            "substituída por": "x_peak_velocity_mm_s",
        },
        "fault": {
            "o que era": "O defeito anotado à mão durante o ensaio.",
            "por que saiu": (
                "Não foi descartada de verdade — foi **arrumada** no Passo 3 e virou a coluna "
                "`classe`. O que sai é a versão bagunçada, com os erros de digitação."
            ),
            "risco de manter": (
                "Alto. Se as duas ficassem, o computador veria `desbalanceado` e "
                "`desabalanceado` como defeitos diferentes."
            ),
            "substituída por": "classe",
        },
    }

    tabela_descarte = pd.DataFrame(
        [
            {
                "coluna que saiu": coluna,
                "o que ela media": JUSTIFICATIVAS[coluna]["o que era"],
                "por que saiu": JUSTIFICATIVAS[coluna]["por que saiu"],
                "o que ficou no lugar": JUSTIFICATIVAS[coluna]["substituída por"],
            }
            for coluna in descartadas
            if coluna in JUSTIFICATIVAS
        ]
    )
    st.dataframe(tabela_descarte, width="stretch", height=290)

    nao_documentadas = [c for c in descartadas if c not in JUSTIFICATIVAS]
    if nao_documentadas:
        st.warning(
            "Estas colunas saíram e ainda não têm justificativa escrita: "
            + ", ".join(f"`{c}`" for c in nao_documentadas)
        )

    st.markdown("#### E se tivéssemos mantido? O risco de cada uma")
    for coluna in descartadas:
        if coluna not in JUSTIFICATIVAS:
            continue
        info = JUSTIFICATIVAS[coluna]
        with st.expander(f"`{coluna}` — {info['o que era']}"):
            st.markdown(f"**Por que saiu:** {info['por que saiu']}")
            st.markdown(f"**O que aconteceria se ficasse:** {info['risco de manter']}")
            if info["substituída por"] != "—":
                st.markdown(
                    f"**A informação não se perdeu** — ela continua na coluna "
                    f"`{info['substituída por']}`."
                )
            else:
                st.markdown(
                    "**Nada se perdeu** — essa coluna não trazia informação sobre o motor."
                )

    st.divider()
    st.markdown("### A prova de que as duplicatas são mesmo a mesma coisa")
    st.caption(
        "Não é preciso acreditar na palavra: dá para conferir nos próprios dados. "
        "Escolha um par e compare."
    )

    pares = {
        "Temperatura: Celsius × Fahrenheit": ("temperature_c", "temperature_f"),
        "Balanço vertical: milímetros × polegadas": (
            "z_rms_velocity_mm_s",
            "z_rms_velocity_in_s",
        ),
        "Balanço horizontal: milímetros × polegadas": (
            "x_rms_velocity_mm_s",
            "x_rms_velocity_in_s",
        ),
    }
    escolha = st.selectbox("Comparar:", list(pares))
    mantida, removida = pares[escolha]

    amostra = limpo[[mantida, removida]].head(8).copy()
    if mantida == "temperature_c":
        amostra["conta: C × 9/5 + 32"] = (amostra[mantida] * 9 / 5 + 32).round(2)
        explicacao = (
            "A coluna que saiu é exatamente o resultado da conta feita sobre a que ficou. "
            "É a mesma temperatura, escrita de outro jeito."
        )
    else:
        amostra["conta: polegadas × 25,4"] = (amostra[removida] * 25.4).round(4)
        explicacao = (
            "A coluna que saiu, multiplicada por 25,4 (o número de milímetros em uma "
            "polegada), dá exatamente a coluna que ficou. É a mesma vibração."
        )

    st.dataframe(amostra, width="stretch")
    st.info(explicacao)

    correlacao = limpo[mantida].corr(limpo[removida])
    col1, col2 = st.columns(2)
    col1.metric("Semelhança entre as duas colunas", f"{correlacao*100:.4f}%")
    col2.metric("Informação nova que a coluna removida trazia", "0%")
    st.caption(
        "Semelhança de 100% significa que, sabendo uma coluna, você sabe a outra sem "
        "precisar olhar. Guardar as duas é como anotar seu peso em quilos e em libras: "
        "são dois números, mas uma informação só."
    )

    st.divider()
    st.markdown(f"### As {len(SINAIS)} medidas que ficaram")
    st.dataframe(pd.DataFrame({"medida mantida": SINAIS}), width="stretch", height=300)

    st.divider()
    st.markdown("### O dataset antes e depois")

    col1, col2 = st.columns(2)
    with col1:
        caixa_dataset(limpo, "ANTES", "Todas as colunas do arquivo, mais as que criamos.", 6)
    with col2:
        caixa_dataset(final, "DEPOIS", "Só o que o modelo vai usar.", 6)

    st.success(
        f"**Este é o dataset limpo, pronto.** {len(final):,}".replace(",", ".")
        + f" linhas e {final.shape[1]} colunas: a data, as {len(SINAIS)} medidas, "
        "o defeito (`classe`) e o ensaio (`segment_id`). Sem nenhum valor faltando."
    )


# ---------------------------------------------------------------------------
# 6. Passo 5 — Fatiar em janelas
# ---------------------------------------------------------------------------

elif secao == SECOES[6]:
    st.title("Passo 5 — Fatiar em janelas")

    estagios = carregar_estagios()
    final = estagios["final"]

    st.markdown(
        f"""
        ### O problema

        Até aqui temos uma tabela gigante de leituras, uma por linha, com os ensaios já
        separados. Antes de resumir qualquer coisa, é preciso decidir **o que vai ser resumido**
        — ou seja, recortar os pedaços.

        Os ensaios têm durações muito diferentes. Alguns têm 50 leituras, outros têm 6.000. Se
        pegássemos cada ensaio inteiro como um pedaço, teríamos poucos exemplos, e um exemplo de
        6.000 leituras contaria o mesmo que um de 50.

        ### A solução: fatiar

        Cortamos cada ensaio em pedaços do mesmo tamanho, chamados **janelas**. A configuração
        atual usa janelas de **{JANELA_TAMANHO} leituras**.

        Como cada leitura vem a cada 2 segundos, isso equivale a cerca de
        **{JANELA_TAMANHO * 2 / 60:.0f} minutos de motor** por janela.

        ### As duas regras do fatiamento

        **1. As janelas se sobrepõem pela metade.** A janela seguinte começa na metade da
        anterior. Isso gera mais exemplos e evita que um defeito seja cortado bem no meio da
        fronteira entre duas janelas.

        **2. Uma janela nunca atravessa dois ensaios.** Se o ensaio acabou, a janela acaba junto.
        Misturar dois ensaios na mesma janela criaria um exemplo que não corresponde a defeito
        nenhum.
        """
    )

    st.divider()
    st.markdown("### O preço do fatiamento: ensaios curtos são jogados fora")

    st.markdown(
        f"""
        Se um ensaio tem **menos** leituras que o tamanho da janela, ele não cabe em nenhuma
        janela — e é **descartado inteiro**. Nenhum exemplo é aproveitado dele.

        Com janela de {JANELA_TAMANHO}, todo ensaio com menos de {JANELA_TAMANHO} leituras se
        perde. Veja o tamanho:
        """
    )

    tam_seg = final.groupby("segment_id").size()
    col1, col2, col3 = st.columns(3)
    col1.metric("Menor ensaio", f"{tam_seg.min()} leituras")
    col2.metric("Ensaio típico (mediana)", f"{int(tam_seg.median())} leituras")
    col3.metric("Maior ensaio", f"{tam_seg.max()} leituras")

    linhas_descarte = []
    for t in (30, 90, 180, 360):
        ok = tam_seg[tam_seg >= t]
        ids = ok.index
        classes_ok = final.loc[final["segment_id"].isin(ids), "classe"].nunique()
        linhas_descarte.append(
            {
                "janela": t,
                "ensaios aproveitados": len(ok),
                "ensaios jogados fora": len(tam_seg) - len(ok),
                "% jogado fora": f"{100*(len(tam_seg)-len(ok))/len(tam_seg):.0f}%",
                "defeitos que sobram": classes_ok,
            }
        )
    st.dataframe(pd.DataFrame(linhas_descarte), width="stretch")

    st.error(
        f"**O ensaio típico tem {int(tam_seg.median())} leituras — menos que a janela de "
        f"{JANELA_TAMANHO}.** Por isso, mais de dois terços dos ensaios são descartados na "
        "configuração atual. Ao subir a janela de 90 para 180, o descarte pulou de 27% para 68%."
    )

    st.divider()
    st.markdown("### O dataset antes e depois")

    amostras = montar_amostras("janela")
    exibicao = amostras[["segment_id", "classe"]].copy()
    exibicao.insert(0, "janela nº", range(1, len(exibicao) + 1))
    exibicao["conteúdo"] = [
        f"{JANELA_TAMANHO} leituras × {len(SINAIS)} medidas" for _ in range(len(exibicao))
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ANTES** — uma tabela corrida de leituras")
        st.dataframe(final[["created_at", "classe", "segment_id"]].head(8), width="stretch")
        st.metric("Linhas", f"{len(final):,}".replace(",", "."))
        st.metric("Defeitos representados", final["classe"].nunique())
    with col2:
        st.markdown("**DEPOIS** — a mesma tabela recortada em pedaços")
        st.dataframe(exibicao.head(8), width="stretch")
        st.metric("Janelas (pedaços)", f"{len(amostras):,}".replace(",", "."))
        st.metric("Defeitos representados", amostras["classe"].nunique())

    st.info(
        f"**Nada foi resumido ainda.** Cada janela continua sendo um bloco de "
        f"{JANELA_TAMANHO} leituras × {len(SINAIS)} medidas — números de verdade, não um "
        "resumo. Transformar cada bloco desses em uma linha só é o que o **Passo 6** faz."
    )

    with st.expander(f"Ver uma janela por dentro — as {JANELA_TAMANHO} leituras de um pedaço"):
        janela_exemplo = amostras.iloc[0]["janela_bruta"]
        st.dataframe(janela_exemplo.head(15), width="stretch")
        st.caption(
            f"Janela nº 1, do ensaio {int(amostras.iloc[0]['segment_id'])} "
            f"(defeito: {amostras.iloc[0]['classe']}). "
            f"Formato: {janela_exemplo.shape[0]} linhas × {janela_exemplo.shape[1]} colunas. "
            "Mostrando as 15 primeiras linhas."
        )

    perdidas = set(final["classe"].unique()) - set(amostras["classe"].unique())
    if perdidas:
        st.error(
            f"**Defeitos que desapareceram: {', '.join(sorted(perdidas))}.** "
            "Nenhum ensaio desse tipo é longo o bastante para formar uma janela de "
            f"{JANELA_TAMANHO} leituras. O sistema simplesmente não consegue mais detectar "
            "esse defeito — ele nunca viu um exemplo."
        )

    st.markdown("### Quantas janelas de cada defeito")
    contagem = (
        amostras["classe"].value_counts().reset_index()
        .rename(columns={"classe": "defeito", "count": "janelas"})
    )
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.dataframe(contagem, width="stretch")
    with col2:
        st.bar_chart(contagem.set_index("defeito")["janelas"])

    st.divider()
    st.markdown("### Explorar os sinais de um ensaio")
    st.caption("Inspeção visual dos dados brutos. Não é o que entra no modelo — é só para ver.")

    defeito = st.selectbox("Defeito:", sorted(final["classe"].dropna().unique()))
    segmentos_defeito = sorted(final.loc[final["classe"] == defeito, "segment_id"].unique())
    segmento = st.selectbox(f"Ensaio ({len(segmentos_defeito)} disponíveis):", segmentos_defeito)
    medidas = st.multiselect("Medidas:", SINAIS, default=SINAIS[:3])

    trecho = final.loc[final["segment_id"] == segmento]
    st.caption(f"{len(trecho)} leituras neste ensaio.")
    if medidas and not trecho.empty:
        dados_plot = trecho[medidas].copy()
        dados_plot.index = trecho["created_at"]
        st.line_chart(dados_plot)


# ---------------------------------------------------------------------------
# 7. Passo 6 — Resumir em números
# ---------------------------------------------------------------------------

elif secao == SECOES[7]:
    st.title("Passo 6 — Resumir cada janela em números")

    estagios = carregar_estagios()
    final = estagios["final"]

    st.markdown(
        f"""
        ### O problema

        O Passo 5 recortou os dados em janelas. Mas cada janela ainda é um **bloco**:
        {JANELA_TAMANHO} linhas por {len(SINAIS)} colunas.

        O modelo não aceita blocos. Ele precisa que cada exemplo seja **uma linha só**. E não
        sabe ler gráfico: não entende "essa linha está subindo" ou "esse trecho está mais
        agitado".

        Então cada bloco de {JANELA_TAMANHO} × {len(SINAIS)} precisa virar **uma única linha**.

        ### A analogia

        Imagine descrever um mês de temperatura para alguém por telefone. Você não vai ler os 30
        valores. Você vai dizer: *"a média foi 22 graus, variou bastante, e foi esquentando ao
        longo do mês"*. Três números que resumem trinta.

        É exatamente isso que fazemos — só que com 5 números, para cada uma das 18 medidas.

        ### Os 5 resumos
        """
    )

    st.markdown(
        """
        | Resumo | O que responde | Por que ajuda |
        |---|---|---|
        | **Mediana** | "Qual o valor típico?" | O nível normal da vibração naquele trecho. Ignora picos isolados. |
        | **Desvio padrão** | "Quanto oscila?" | Rolamento gasto faz a vibração ficar instável. |
        | **Inclinação** | "Está subindo ou descendo?" | Distingue um defeito que está piorando de um estável. |
        | **Amplitude** | "Qual a diferença entre o maior e o menor?" | Captura picos extremos. |
        | **IQR** | "Quanto oscila, ignorando os extremos?" | Como o desvio padrão, mas sem se deixar enganar por um pico solitário. |
        """
    )

    st.info(
        f"**{len(SINAIS)} medidas × 5 resumos = {len(NOMES_FEATURES)} números.** "
        "Essa lista de números é o que o modelo recebe. No jargão, cada um desses números se "
        "chama *feature* (característica)."
    )

    st.divider()
    st.markdown("### Vendo a transformação acontecer")

    amostras = montar_amostras("janela")
    janela_exemplo = amostras.iloc[0]["janela_bruta"]
    vetor = amostras.iloc[0]["features"]

    st.markdown(f"**ENTRA:** uma janela — um bloco de {JANELA_TAMANHO} linhas")
    st.dataframe(janela_exemplo.head(10), width="stretch")
    st.caption(
        f"Formato: {janela_exemplo.shape[0]} linhas × {janela_exemplo.shape[1]} colunas. "
        "Mostrando as 10 primeiras linhas."
    )

    st.markdown(f"**SAI:** uma linha só — com {len(vetor)} colunas")
    linha_unica = pd.DataFrame([vetor], columns=NOMES_FEATURES, index=["janela nº 1"])
    st.dataframe(linha_unica, width="stretch")
    st.caption(
        f"Formato: 1 linha × {len(vetor)} colunas. "
        "Arraste a barra horizontal para ver todas — a tabela é bem mais larga que a tela."
    )

    st.success(
        f"**É isso que o passo faz:** um bloco de {janela_exemplo.shape[0]} linhas entrou, "
        f"e saiu **uma linha só**, com {len(vetor)} colunas. "
        f"Cada uma das {len(amostras):,}".replace(",", ".")
        + " janelas vira uma linha dessas na tabela final."
    )

    st.markdown("#### A mesma informação, virada de lado")
    st.caption(
        "Uma tabela de 90 colunas não cabe na tela. Para conseguir ler os nomes com calma, "
        "a lista abaixo mostra os mesmos 90 valores empilhados na vertical. "
        "**É a mesma linha de cima, girada** — na tabela de verdade, cada um desses nomes "
        "é uma coluna, e os valores ficam todos na mesma linha."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "nome da coluna": NOMES_FEATURES,
                "valor nesta janela": vetor,
                "medida de origem": [s for s in SINAIS for _ in ESTATISTICAS],
                "resumo": ESTATISTICAS * len(SINAIS),
            }
        ),
        width="stretch",
        height=400,
    )

    st.markdown("### Vendo no gráfico o que os 5 resumos capturam")
    st.caption(
        "Escolha uma das 18 medidas. O gráfico mostra como ela se comportou ao longo desta "
        "janela, e abaixo estão os 5 números que substituem toda essa curva."
    )
    sinal_escolhido = st.selectbox("Escolha uma medida para visualizar:", SINAIS)
    serie = janela_exemplo[sinal_escolhido].reset_index(drop=True)
    st.line_chart(serie)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mediana", f"{np.median(serie):.3f}")
    col2.metric("Desvio padrão", f"{np.std(serie):.3f}")
    t = np.arange(len(serie)) - (len(serie) - 1) / 2
    col3.metric("Inclinação", f"{np.dot(t, serie)/np.dot(t, t):.4f}")
    col4.metric("Amplitude", f"{serie.max()-serie.min():.3f}")
    col5.metric("IQR", f"{np.percentile(serie,90)-np.percentile(serie,10):.3f}")

    st.caption(
        "Esses 5 números são tudo o que o modelo vai saber sobre essa curva. "
        "Toda a forma do gráfico é jogada fora e substituída por eles."
    )

    st.warning(
        "**Atenção a uma coincidência que confunde.** O resultado tem 90 colunas porque "
        "18 × 5 = 90. Esse 90 **não tem nada a ver** com o tamanho da janela do Passo 5 "
        "(que já foi 90 em um dos testes). Mudar o tamanho da janela não muda a quantidade "
        "de colunas — continuam sendo 90."
    )

    st.divider()
    st.markdown("### A tabela final, completa")
    st.caption(
        "Empilhando todas as janelas, uma por linha, chega-se à tabela que vai para o modelo."
    )

    matriz = pd.DataFrame(np.vstack(amostras["features"].to_list()), columns=NOMES_FEATURES)
    matriz.insert(0, "classe", amostras["classe"].to_numpy())
    matriz.insert(1, "segment_id", amostras["segment_id"].to_numpy())

    st.dataframe(matriz.head(10), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Linhas (uma por janela)", f"{len(matriz):,}".replace(",", "."))
    col2.metric("Colunas de números", len(NOMES_FEATURES))
    col3.metric("Colunas de apoio", "2 (classe e ensaio)")

    st.download_button(
        "Baixar esta tabela em CSV",
        data=matriz.to_csv(index=False),
        file_name=f"dataset_janela_{JANELA_TAMANHO}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 8. Passo 7 — Modelo
# ---------------------------------------------------------------------------

elif secao == SECOES[8]:
    st.title("Passo 7 — O modelo que aprende")

    st.markdown(
        """
        ### O que é uma "árvore de decisão"

        Uma árvore de decisão é uma sequência de perguntas de sim/não, como um jogo de adivinhação:

        > A vibração vertical passa de 2,5 mm/s?
        > → **Sim**: A temperatura passa de 45 °C?
        > →→ **Sim**: é `rolamento_inner`.
        > →→ **Não**: é `desbalanceado`.

        O computador monta essas perguntas sozinho, olhando os exemplos e descobrindo quais
        perguntas melhor separam um defeito do outro.

        ### Por que usamos uma *floresta* de árvores

        Uma árvore sozinha é **teimosa**: ela decora os exemplos que viu e erra feio em exemplos
        novos.

        A solução é montar **400 árvores diferentes**, cada uma vendo um pedaço sorteado dos dados
        e um subconjunto sorteado das perguntas possíveis. Na hora de decidir, todas votam, e a
        resposta mais votada vence.

        É a mesma lógica de pedir opinião a 400 pessoas em vez de uma. Cada uma erra de um jeito,
        e os erros tendem a se cancelar. Por isso o nome: **Random Forest** — floresta aleatória.
        """
    )

    st.divider()
    st.markdown("### A configuração usada")
    st.dataframe(
        pd.DataFrame(
            [
                {"ajuste": "n_estimators = 400", "significa": "A floresta tem 400 árvores."},
                {"ajuste": "random_state = 0", "significa": "Fixa o sorteio, para o resultado ser sempre igual quando você repetir."},
                {"ajuste": "n_jobs = -1", "significa": "Usa todos os núcleos do processador, para treinar mais rápido."},
            ]
        ),
        width="stretch",
    )

    st.markdown(
        """
        ### Por que este modelo, e não outro

        - **Não se incomoda com escalas diferentes.** Nossas medidas estão em unidades bem
          diferentes (graus, hertz, milímetros por segundo). Muitos modelos exigem colocar tudo na
          mesma escala antes; este não.
        - **Aguenta defeitos com poucos exemplos.** Alguns defeitos têm bem menos janelas que
          outros, e a floresta lida razoavelmente com esse desequilíbrio.
        - **Consegue explicar no que se baseou.** Dá para perguntar quais medidas mais pesaram
          na decisão — o que não é possível em modelos mais fechados.
        """
    )

    st.divider()
    st.markdown("### O formato exato do que entra no modelo")

    amostras = montar_amostras("janela")
    X = np.vstack(amostras["features"].to_list())

    col1, col2, col3 = st.columns(3)
    col1.metric("Exemplos (janelas)", f"{len(X):,}".replace(",", "."))
    col2.metric("Números por exemplo", X.shape[1])
    col3.metric("Defeitos a distinguir", amostras["classe"].nunique())

    st.markdown("**A tabela final, do jeito que o modelo vê:**")
    previa = pd.DataFrame(X[:6], columns=NOMES_FEATURES).iloc[:, :6]
    previa.insert(0, "→ RESPOSTA CERTA", amostras["classe"].head(6).to_numpy())
    st.dataframe(previa, width="stretch")
    st.caption(
        f"Mostrando 6 exemplos e as 6 primeiras colunas de {X.shape[1]}. "
        "O modelo aprende a ligação entre os números e a coluna de resposta."
    )


# ---------------------------------------------------------------------------
# 9. Passo 8 — Validação
# ---------------------------------------------------------------------------

elif secao == SECOES[9]:
    st.title("Passo 8 — Como sabemos se funciona")

    st.markdown(
        """
        ### A regra de ouro

        Nunca se testa o modelo com os mesmos exemplos usados para treiná-lo.

        Seria como dar a um aluno a prova com o gabarito junto. Ele tira 10, e você não descobriu
        nada sobre o quanto ele aprendeu.

        ### Como o teste é feito

        Dividimos os exemplos em 5 grupos. Treinamos com 4 e testamos no que sobrou. Repetimos 5
        vezes, cada vez deixando um grupo diferente de fora. No fim, tiramos a média.

        Cada uma dessas 5 rodadas se chama **fold**.

        ### E aqui está a parte importante deste projeto

        **Existem duas maneiras de fazer essa divisão, e elas dão respostas radicalmente
        diferentes.**
        """
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Jeito 1: dividir no sorteio")
        st.markdown(
            """
            Embaralha todas as janelas e reparte em 5 grupos.

            **O que dá errado:** lembra que janelas do mesmo ensaio são quase idênticas —
            mesmo motor, mesma montagem, minutos de diferença, e ainda por cima elas se
            sobrepõem pela metade?

            Ao embaralhar, pedaços do **mesmo ensaio** vão parar no treino **e** no teste.

            É como estudar para a prova com as respostas de metade das questões que vão cair.
            O modelo reconhece o ensaio que já viu, não o defeito.
            """
        )
        st.metric("Acurácia por este método", "92,1%")
        st.error("Este número **não vale**. Ele está inflado pela cola.")

    with col2:
        st.markdown("### Jeito 2: dividir por ensaio")
        st.markdown(
            """
            Mantém **todas as janelas de um ensaio juntas**, do mesmo lado da divisão. Se o
            ensaio nº 12 está no treino, nenhum pedaço dele aparece no teste.

            **O que isso testa:** o modelo é obrigado a opinar sobre um ensaio que nunca viu.

            É exatamente a pergunta que interessa na prática: *quando eu instalar isso num motor
            novo, na fábrica, vai funcionar?*
            """
        )
        st.metric("Acurácia por este método", "43,8%")
        st.success("**Este é o número honesto.** É o que se deve reportar.")

    st.divider()
    st.markdown(
        """
        ### O que a distância entre 92% e 44% significa

        Quase 50 pontos de diferença. Essa distância é a medida exata do quanto o modelo está
        **colando**.

        Ela revela o problema central do projeto: o modelo aprendeu a reconhecer **o ensaio**, e
        não **o defeito**. Ele decorou coisas específicas daquela montagem — a temperatura
        ambiente do dia, a rotação exata, a maneira como o sensor foi parafusado — em vez de
        aprender como um rolamento gasto se comporta em geral.

        Num motor novo, nada disso se repete, e o desempenho despenca.

        ### Colocando 44% em perspectiva

        Não é um resultado bom, mas também não é nada. São 13 defeitos possíveis: chutar acertaria
        cerca de 8%. Acertar 44% significa que existe sinal real sendo capturado — só que muito
        menos do que os 92% sugeriam.
        """
    )

    st.divider()
    st.markdown("### Rodar a validação agora")
    st.warning(
        "Este cálculo treina 10 florestas de 400 árvores. **Leva alguns minutos.** "
        "O resultado fica guardado depois da primeira vez."
    )

    modo = st.radio(
        "Formato dos exemplos:",
        ["janela", "segmento"],
        format_func=lambda v: (
            f"Janelas de {JANELA_TAMANHO} leituras" if v == "janela" else "Ensaios inteiros"
        ),
        horizontal=True,
    )

    if st.button("Executar a validação", type="primary"):
        resultados = rodar_avaliacao(modo, None)

        col1, col2 = st.columns(2)
        for coluna, resultado in zip((col1, col2), resultados):
            with coluna:
                honesto = resultado["nome"] == "por_segmento"
                titulo = "Dividindo por ensaio" if honesto else "Dividindo no sorteio"
                st.markdown(f"#### {titulo}")
                st.metric("Acurácia média", f"{resultado['media']*100:.1f}%")
                if honesto:
                    st.success("Número honesto.")
                else:
                    st.error("Inflado pela cola.")

                detalhes = pd.DataFrame(
                    [
                        {
                            "rodada": f["fold"],
                            "treino": f["train_size"],
                            "teste": f["test_size"],
                            "acertos": f"{f['accuracy']*100:.1f}%",
                        }
                        for f in resultado["folds"]
                    ]
                )
                st.dataframe(detalhes, width="stretch")

        honesto = [r for r in resultados if r["nome"] == "por_segmento"][0]
        espalhamento = np.std([f["accuracy"] for f in honesto["folds"]])
        st.info(
            f"**Sobre a variação entre as rodadas.** As 5 rodadas honestas variaram com desvio "
            f"de {espalhamento*100:.1f} pontos. Isso é bastante: com janela de {JANELA_TAMANHO} "
            "sobram poucos ensaios, então cada rodada testa em pouca gente e o resultado balança "
            "muito. Diferenças pequenas entre configurações não são confiáveis — ponto que volta "
            "na seção do Optuna."
        )


# ---------------------------------------------------------------------------
# 10. Experimento janelas
# ---------------------------------------------------------------------------

elif secao == SECOES[10]:
    st.title("Experimento — qual o melhor tamanho de janela?")

    st.markdown(
        """
        ### A pergunta

        No Passo 5, escolhemos fatiar em janelas de 180 leituras. Mas por que 180? Testamos
        quatro tamanhos para descobrir.

        ### O dilema

        - **Janela pequena:** muitos exemplos, mas cada um vê pouco tempo de motor. Pode ser
          curto demais para o defeito aparecer.
        - **Janela grande:** cada exemplo vê bastante tempo, mas sobram poucos exemplos e muitos
          ensaios curtos são jogados fora.
        """
    )

    st.divider()
    st.markdown("### Os resultados medidos")

    tabela = pd.DataFrame(
        [
            {"janela": 30, "tempo de motor": "~1 min", "exemplos": 10825, "defeitos": 14,
             "ensaios usados": "229 de 234", "acurácia honesta": "43,3%", "acurácia inflada": "92,2%"},
            {"janela": 90, "tempo de motor": "~3 min", "exemplos": 3401, "defeitos": 14,
             "ensaios usados": "172 de 234", "acurácia honesta": "45,0%", "acurácia inflada": "91,4%"},
            {"janela": 180, "tempo de motor": "~6 min", "exemplos": 1559, "defeitos": 13,
             "ensaios usados": "75 de 234", "acurácia honesta": "43,8%", "acurácia inflada": "92,1%"},
            {"janela": 360, "tempo de motor": "~12 min", "exemplos": 721, "defeitos": 12,
             "ensaios usados": "63 de 234", "acurácia honesta": "41,8%", "acurácia inflada": "82,7%"},
        ]
    )
    st.dataframe(tabela, width="stretch")

    st.markdown("### O que os números dizem")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**A acurácia honesta quase não muda**")
        st.bar_chart(
            pd.DataFrame(
                {"acurácia honesta (%)": [43.3, 45.0, 43.8, 41.8]},
                index=["30", "90", "180", "360"],
            )
        )
        st.caption(
            "Todas ficam entre 42% e 45%. Uma variação de 3 pontos, dentro do próprio ruído "
            "da medição. **O tamanho da janela não é o que decide o resultado.**"
        )
    with col2:
        st.markdown("**Mas o descarte de ensaios dispara**")
        st.bar_chart(
            pd.DataFrame(
                {"% de ensaios jogados fora": [2.1, 26.5, 67.9, 73.1]},
                index=["30", "90", "180", "360"],
            )
        )
        st.caption(
            "Aqui a diferença é enorme. Entre 90 e 180 o descarte pula de 27% para 68% — porque "
            "o ensaio típico tem 150 leituras, e a janela de 180 não cabe nele."
        )

    st.divider()
    st.markdown(
        """
        ### A conclusão do experimento

        **90 seria a escolha melhor pelos números:** tem a maior acurácia honesta (45,0%), mantém
        os 14 defeitos, e aproveita 172 ensaios em vez de 75.

        Com 180, dois defeitos ficam prejudicados — `falta_fase` desaparece completamente e
        `motor_desligado` fica com duas janelas apenas.

        Com 360, até a acurácia inflada desaba (de 92% para 83%): sobram exemplos de menos para
        o modelo aprender qualquer coisa.

        **A configuração atual é 180**, por decisão de projeto. Este painel registra o que a
        medição mostrou, para que a escolha seja informada.
        """
    )

    st.info(
        "A lição maior do experimento: quatro tamanhos diferentes, e a acurácia honesta variou "
        "só 3 pontos. Mexer no tamanho da janela não resolve o problema deste sistema. O que "
        "trava o resultado é outra coisa — o modelo aprender ensaio em vez de defeito."
    )


# ---------------------------------------------------------------------------
# 11. Optuna
# ---------------------------------------------------------------------------

elif secao == SECOES[11]:
    st.title("Optuna — o que aquela mensagem queria dizer")

    st.markdown(
        """
        Se você marcou a opção de usar os parâmetros do Optuna e apareceu um aviso dizendo que
        eles são **piores**, esta seção explica o que aconteceu. É um resultado que confunde à
        primeira vista, e vale a pena entender.
        """
    )

    st.divider()
    st.markdown(
        """
        ### O que o Optuna faz

        O modelo tem "botões de ajuste": quantas árvores usar, quão profundas elas podem ser,
        quantos exemplos são necessários para criar uma nova pergunta, e assim por diante. Esses
        botões se chamam **hiperparâmetros**.

        Escolher bons valores na mão é chute. O Optuna é uma ferramenta que testa muitas
        combinações automaticamente e vai aprendendo quais regiões são promissoras — em vez de
        sortear às cegas.

        Testamos **80 combinações diferentes**.
        """
    )

    st.markdown("### O que ele encontrou")

    if PARAMS_JSON.exists():
        melhores = carregar_melhores_params()
        traducao = {
            "n_estimators": "Quantidade de árvores na floresta",
            "criterion": "Fórmula usada para escolher cada pergunta",
            "max_depth": "Profundidade máxima de cada árvore",
            "min_samples_split": "Mínimo de exemplos para criar uma pergunta nova",
            "min_samples_leaf": "Mínimo de exemplos em cada resposta final",
            "max_features": "Quantas medidas cada árvore pode considerar por vez",
            "class_weight": "Se dá peso extra aos defeitos com poucos exemplos",
            "bootstrap": "Se cada árvore vê uma amostra sorteada ou os dados todos",
        }
        st.dataframe(
            pd.DataFrame(
                [
                    {"botão": traducao.get(k, k), "valor escolhido": str(v)}
                    for k, v in melhores.items()
                    if k in traducao
                ]
            ),
            width="stretch",
        )
    else:
        st.info("Rode `python otimizacao.py` para gerar os parâmetros.")

    col1, col2 = st.columns(2)
    col1.metric("Ajustes padrão (sem Optuna)", "43,8%")
    col2.metric("Melhor combinação do Optuna", "45,3%", delta="+1,5 ponto")

    st.markdown("Parece uma melhora. **Mas não é.**")

    st.divider()
    st.markdown(
        """
        ### Onde está a pegadinha

        #### A analogia da loteria

        Imagine que você quer descobrir se alguém tem talento para prever a loteria.

        Você chama **80 pessoas** e pede o palpite de cada uma para os sorteios da semana passada.
        Uma delas acerta bem mais que as outras. Você anuncia: *"encontramos a vidente!"*

        Mas espere. Com 80 pessoas chutando, é **esperado** que alguma se saia melhor — por puro
        acaso. Você não descobriu talento. Você descobriu **a pessoa mais sortuda daquela
        semana específica**.

        Na semana seguinte, ela volta a ser gente comum.

        #### Foi exatamente isso que aconteceu

        O Optuna testou 80 combinações **nas mesmas 5 rodadas de teste**, e escolheu a que se saiu
        melhor **naquelas 5 rodadas**. Como vimos no Passo 8, essas rodadas variam muito entre si
        — desvio de uns 7 pontos.

        Com uma variação dessa e 80 tentativas, alguma combinação acerta o padrão de sorte
        daquelas rodadas. Os 45,3% mediram sorte, não qualidade.
        """
    )

    st.divider()
    st.markdown(
        """
        ### Como descobrimos que era sorte

        Existe um teste mais rigoroso, chamado **validação aninhada**. A ideia: fazer a busca do
        Optuna **só dentro do material de treino**, e depois testar em ensaios que não
        participaram nem do treino nem da busca.

        Voltando à analogia: em vez de premiar quem acertou os sorteios da semana passada, você
        pega a pessoa escolhida e pede o palpite dela para os sorteios da **semana que vem** —
        que ninguém viu ainda.

        **O resultado:**
        """
    )

    comparacao = pd.DataFrame(
        {
            "rodada": ["1", "2", "3", "4", "5", "MÉDIA"],
            "com Optuna": ["52,3%", "37,3%", "44,6%", "35,6%", "34,5%", "40,8%"],
            "ajustes padrão": ["52,3%", "45,8%", "51,2%", "34,6%", "35,1%", "43,8%"],
            "quem ganhou": ["empate", "padrão", "padrão", "Optuna", "padrão", "PADRÃO"],
        }
    )
    st.dataframe(comparacao, width="stretch")

    col1, col2 = st.columns(2)
    col1.metric("Com Optuna (teste rigoroso)", "40,8%")
    col2.metric("Ajustes padrão", "43,8%", delta="+3,0 pontos", delta_color="normal")

    st.error(
        "**Os ajustes encontrados pelo Optuna são 3 pontos PIORES que não ter ajustado nada.** "
        "O ganho de +1,5 ponto era ilusão. Por isso o aviso na tela: eles estão disponíveis para "
        "inspeção, mas não são usados pelo sistema."
    )

    st.divider()
    st.markdown(
        """
        ### Por que o Optuna falhou aqui — e isso não é defeito dele

        O Optuna funciona bem quando existe sinal claro para otimizar. Aqui não existia.

        Com janela de 180, sobram **75 ensaios**. Divididos em 5 rodadas, cada teste tem uns 15
        ensaios. Com tão pouca gente, o resultado de cada rodada balança uns 7 pontos só por
        causa de **quais** ensaios caíram no teste.

        A diferença real entre uma boa e uma má configuração de botões é **menor que esses 7
        pontos de balanço**. O sinal está afogado no ruído.

        O Optuna fez o trabalho dele: encontrou o que maximizava o número que você pediu. O
        problema é que aquele número era, em boa parte, ruído — e ele otimizou o ruído.

        ### A lição

        Isso não significa que ajustar hiperparâmetros seja inútil em geral. Significa que **neste
        conjunto de dados, com esta quantidade de ensaios, não há o que ajustar.**

        O que trava o sistema é o mesmo de sempre: poucos ensaios, e o modelo aprendendo a
        reconhecer o ensaio em vez do defeito. Enquanto isso não mudar, nem tamanho de janela nem
        ajuste de botões vai mover o resultado.
        """
    )

    st.divider()
    st.markdown("### Uma decisão de projeto que vale explicar")
    st.info(
        "A busca foi configurada para otimizar a **acurácia honesta** (divisão por ensaio), não "
        "a inflada. Se tivesse sido apontada para a inflada, o Optuna teria encontrado os ajustes "
        "que melhor **colam** — exatamente o defeito que já limita o sistema. O número apareceria "
        "perto de 93% e não significaria nada."
    )


# ---------------------------------------------------------------------------
# 12. Resumo
# ---------------------------------------------------------------------------

elif secao == SECOES[12]:
    st.title("Resumo final e limitações")

    st.markdown("### O caminho completo, em uma tabela")

    estagios = carregar_estagios()
    final = estagios["final"]
    amostras = montar_amostras("janela")

    resumo = pd.DataFrame(
        [
            {"etapa": "Arquivo original", "o que é uma linha": "uma leitura do sensor",
             "linhas": f"{len(estagios['bruto']):,}".replace(",", "."),
             "colunas": str(estagios["bruto"].shape[1])},
            {"etapa": "1 — Ordenado no tempo", "o que é uma linha": "uma leitura do sensor",
             "linhas": f"{len(estagios['ordenado']):,}".replace(",", "."),
             "colunas": str(estagios["ordenado"].shape[1])},
            {"etapa": "2 — Separado em ensaios", "o que é uma linha": "uma leitura, com o ensaio marcado",
             "linhas": f"{len(estagios['com_segmento']):,}".replace(",", "."),
             "colunas": str(estagios["com_segmento"].shape[1])},
            {"etapa": "3 — Nomes limpos", "o que é uma linha": "uma leitura, com o defeito padronizado",
             "linhas": f"{len(estagios['limpo']):,}".replace(",", "."),
             "colunas": str(estagios["limpo"].shape[1])},
            {"etapa": "4 — Medidas escolhidas", "o que é uma linha": "uma leitura, só com o necessário",
             "linhas": f"{len(final):,}".replace(",", "."), "colunas": str(final.shape[1])},
            {"etapa": "5 — Recortado em janelas", "o que é uma linha": "ainda uma leitura, mas agrupada em blocos",
             "linhas": f"{len(amostras):,} blocos".replace(",", "."),
             "colunas": f"{JANELA_TAMANHO} leituras × {len(SINAIS)} medidas cada"},
            {"etapa": "6 — Cada janela resumida", "o que é uma linha": "uma janela inteira virada em 90 números",
             "linhas": f"{len(amostras):,}".replace(",", "."), "colunas": f"{len(NOMES_FEATURES)} + resposta"},
        ]
    )
    st.dataframe(resumo, width="stretch")

    st.caption(
        "Repare como o significado de 'linha' muda no meio do caminho: no começo é uma leitura "
        "do sensor; no fim é uma janela inteira resumida."
    )

    st.divider()
    st.markdown("### O resultado")

    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia honesta", "43,8%")
    col2.metric("Chutar acertaria", "~8%")
    col3.metric("Acurácia inflada (não usar)", "92,1%")

    st.markdown(
        """
        O sistema aprendeu algo real — acerta cinco vezes mais que o chute. Mas erra mais da
        metade das vezes, e não está pronto para decidir manutenção sozinho.
        """
    )

    st.divider()
    st.markdown("### As limitações, sem maquiagem")

    st.markdown(
        """
        **1. O modelo aprende o ensaio, não o defeito.**
        É a limitação principal. A distância entre 92% e 44% mostra que boa parte do acerto vem de
        reconhecer condições específicas de cada montagem, e não o comportamento do defeito. Num
        motor novo, isso não se transfere.

        **2. Há poucos ensaios.** São 234 no total, e só 75 sobrevivem à janela de 180. Para 13
        defeitos diferentes, é pouco. Toda medição fica com margem de erro grande.

        **3. Dois defeitos não são detectáveis na configuração atual.** `falta_fase` sumiu do
        conjunto e `motor_desligado` tem duas janelas. Se detectar falta de fase for um requisito,
        esta configuração não atende.

        **4. Ajustar não adiantou.** Testamos quatro tamanhos de janela (variação de 3 pontos) e
        80 combinações de hiperparâmetros (piorou 3 pontos). O gargalo não é configuração.
        """
    )

    st.divider()
    st.markdown("### O que teria chance de melhorar de verdade")

    st.markdown(
        """
        Todos os caminhos abaixo atacam a limitação nº 1 — fazer o modelo enxergar o defeito e
        não o ensaio:

        **Comparar cada leitura com o próprio ensaio, em vez de usar o valor absoluto.**
        Hoje o modelo vê "vibração de 3,2 mm/s". Se visse "vibração 40% acima do normal deste
        motor", o número deixaria de carregar a assinatura da montagem.

        **Normalizar cada ensaio pela própria linha de base**, antes de resumir em números.
        Mesma ideia, aplicada de forma sistemática.

        **Revisar como os ensaios são separados.** Vale checar se a regra atual (mudou o rótulo,
        ou parou por 1 hora) está agrupando coisas que deveriam ficar separadas.

        **Coletar mais ensaios**, especialmente dos defeitos raros. É o caminho mais lento e o
        mais garantido.
        """
    )

    st.divider()
    st.markdown("### Onde cada coisa está no código")
    st.dataframe(
        pd.DataFrame(
            [
                {"arquivo": "prep.py", "responsabilidade": "Passos 1 a 6 — toda a preparação dos dados"},
                {"arquivo": "avaliacao.py", "responsabilidade": "Passo 8 — as duas validações"},
                {"arquivo": "otimizacao.py", "responsabilidade": "A busca do Optuna e o teste rigoroso"},
                {"arquivo": "sistema.py", "responsabilidade": "Consultar o modelo já treinado"},
                {"arquivo": "app.py", "responsabilidade": "Esta interface"},
                {"arquivo": "pipeline_manutencao_preditiva.ipynb", "responsabilidade": "A mesma explicação em formato de notebook"},
            ]
        ),
        width="stretch",
    )
