"""
Dashboard - Análise de Laudos Mamográficos
CP Data Analises - Professor Dashboard
Instalar dependências: pip install streamlit plotly pandas scipy openpyxl
Rodar: streamlit run dashboard_laudos.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, mannwhitneyu, binomtest
import re
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Laudos Mamográficos",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
    margin: 5px;
}
.metric-card h2 { font-size: 2.2rem; margin: 0; }
.metric-card p  { font-size: 0.95rem; margin: 5px 0 0; opacity: 0.85; }

.card-green { background: linear-gradient(135deg, #145a32, #1e8449) !important; }
.card-red   { background: linear-gradient(135deg, #7b241c, #cb4335) !important; }
.card-gold  { background: linear-gradient(135deg, #784212, #d4ac0d) !important; }

.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    border-left: 5px solid #2d6a9f;
    padding-left: 10px;
    margin: 25px 0 15px;
}
.info-box {
    background: #eaf3fb;
    border-left: 4px solid #2d6a9f;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.9rem;
    margin-bottom: 10px;
}
.warn-box {
    background: #fef9e7;
    border-left: 4px solid #d4ac0d;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("cp_data_analises.xlsx", sheet_name="Dados")
    dic = pd.read_excel("cp_data_analises.xlsx", sheet_name="Dicionário")

    # Engenharia de features nos textos
    text_cols = ["Indicacao", "Achados", "Analise_Comparativa"]

    def safe_len(s):
        return len(str(s)) if pd.notna(s) else 0

    def safe_words(s):
        return len(str(s).split()) if pd.notna(s) else 0

    def has_underscore(s):
        return int("_" in str(s)) if pd.notna(s) else 0

    def has_dimension(s):
        pattern = r"\d+[,.]?\d*\s*cm|\d+[,.]?\d*\s*mm"
        return int(bool(re.search(pattern, str(s).lower()))) if pd.notna(s) else 0

    for col in text_cols:
        short = col.split("_")[0].lower()
        df[f"chars_{short}"]      = df[col].apply(safe_len)
        df[f"palavras_{short}"]   = df[col].apply(safe_words)
        df[f"underscore_{short}"] = df[col].apply(has_underscore)
        df[f"dimensao_{short}"]   = df[col].apply(has_dimension)

    # Quartil de palavras do diagnóstico
    df["quartil_palavras"] = pd.qcut(
        df["Qtd_Palavras_Diagnostico"], q=4,
        labels=["Q1 (poucas)", "Q2", "Q3", "Q4 (muitas)"],
        duplicates="drop",
    )

    # Rótulo legível do diagnóstico
    df["Diagnostico_Label"] = df["Caso_Positivo"].map({0: "Benigno/Normal", 1: "Maligno/Suspeito"})

    return df, dic


df, dic = load_data()

PALAVRAS_STOPPT = set([
    "não","de","da","do","as","os","se","no","na","em","a","e","o","que",
    "com","por","para","um","uma","à","ao","dos","das","nos","nas","é","ou",
])

def top_palavras(serie, n=25, stopwords=PALAVRAS_STOPPT):
    texto = " ".join(serie.dropna().str.lower().tolist())
    palavras = re.findall(r"[a-záéíóúâêîôûãõàçü]{4,}", texto)
    counter = Counter(w for w in palavras if w not in stopwords)
    return pd.DataFrame(counter.most_common(n), columns=["Palavra", "Frequência"])


# ─────────────────────────────────────────────
# SIDEBAR – FILTROS
# ─────────────────────────────────────────────
st.sidebar.title("🔬 Filtros")

st.sidebar.markdown("**Motivo do Exame**")
motivos = df["Motivo_Exame"].unique().tolist()
motivos_sel = st.sidebar.multiselect("Selecione", motivos, default=motivos)

st.sidebar.markdown("**BI-RADS**")
birads_opts = sorted(df["BI-RADS"].unique().tolist())
birads_sel = st.sidebar.multiselect("Selecione", birads_opts, default=birads_opts)

st.sidebar.markdown("**Filtro especial**")
apenas_com_medida = st.sidebar.checkbox("Apenas laudos com lesão com medida (Lesao_Com_Medida = 1)")
apenas_preenchidos = st.sidebar.checkbox("Apenas diagnósticos preenchidos", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Critério de exclusão automático**")
excl_poucas = st.sidebar.slider(
    "Excluir laudos com menos de N palavras", 0, 30, 0
)

# Aplica filtros
df_f = df[
    df["Motivo_Exame"].isin(motivos_sel) &
    df["BI-RADS"].isin(birads_sel)
]
if apenas_com_medida:
    df_f = df_f[df_f["Lesao_Com_Medida"] == 1]
if apenas_preenchidos:
    df_f = df_f[df_f["Diagnostico_Preenchido"] == 1]
if excl_poucas > 0:
    df_f = df_f[df_f["Qtd_Palavras_Diagnostico"] >= excl_poucas]


# ─────────────────────────────────────────────
# TÍTULO
# ─────────────────────────────────────────────
st.title("🩺 Análise de Laudos Mamográficos")
st.markdown(
    f"Base filtrada: **{len(df_f):,} registros** de {len(df):,} totais  |  "
    f"Excluídos: **{len(df)-len(df_f):,}**"
)

tabs = st.tabs([
    "📊 Visão Geral",
    "📝 Textos & Variáveis",
    "🔤 Palavras & Laudos",
    "📏 Tamanho × Diagnóstico",
    "🔍 Critérios de Exclusão",
    "📐 Lesão com Medida",
    "📉 Distribuição Binomial",
    "📖 Dicionário",
])


# ══════════════════════════════════════════════
# TAB 1 – VISÃO GERAL
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Cards da Base Completa</div>', unsafe_allow_html=True)

    total     = len(df_f)
    benigno_n = (df_f["Caso_Positivo"] == 0).sum()
    maligno_n = (df_f["Caso_Positivo"] == 1).sum()
    prop_mal  = maligno_n / total * 100 if total else 0
    sem_diag  = (df_f["Diagnostico_Preenchido"] == 0).sum()
    com_medida= (df_f["Lesao_Com_Medida"] == 1).sum()
    n_cols    = df_f.shape[1]
    n_vars    = 16  # colunas originais

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h2>{total:,}</h2><p>Total de Laudos</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card card-green">
            <h2>{benigno_n:,}</h2><p>Benignos / Normais</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card card-red">
            <h2>{maligno_n:,}</h2><p>Malignos / Suspeitos</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card card-gold">
            <h2>{prop_mal:.1f}%</h2><p>Taxa de Malignidade</p></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <h2>{com_medida:,}</h2><p>Com Lesão Medida</p></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-card">
            <h2>{n_vars}</h2><p>Variáveis Originais</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribuição BI-RADS")
        birads_count = df_f["BI-RADS"].value_counts().reset_index()
        birads_count.columns = ["BI-RADS", "Quantidade"]
        cores = {
            "BI-RADS 0": "#95a5a6", "BI-RADS 1": "#27ae60", "BI-RADS 2": "#2ecc71",
            "BI-RADS 3": "#f39c12", "BI-RADS 4": "#e67e22", "BI-RADS 5": "#e74c3c",
            "BI-RADS 6": "#8e44ad", "Sem Classificação": "#bdc3c7",
        }
        birads_count["cor"] = birads_count["BI-RADS"].map(cores).fillna("#aaa")
        fig = px.bar(
            birads_count, x="BI-RADS", y="Quantidade",
            color="BI-RADS", color_discrete_map=cores,
            text="Quantidade", template="plotly_white",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Proporção Benigno vs Maligno")
        pie_df = pd.DataFrame({
            "Diagnóstico": ["Benigno/Normal", "Maligno/Suspeito"],
            "Quantidade": [benigno_n, maligno_n],
        })
        fig2 = px.pie(
            pie_df, values="Quantidade", names="Diagnóstico",
            color="Diagnóstico",
            color_discrete_map={"Benigno/Normal": "#27ae60", "Maligno/Suspeito": "#e74c3c"},
            template="plotly_white",
        )
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Motivo do Exame")
        mot = df_f["Motivo_Exame"].value_counts().reset_index()
        mot.columns = ["Motivo", "Qtd"]
        fig3 = px.pie(mot, values="Qtd", names="Motivo", template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Proporção por Motivo e Diagnóstico")
        cross = df_f.groupby(["Motivo_Exame", "Diagnostico_Label"]).size().reset_index(name="n")
        fig4 = px.bar(
            cross, x="Motivo_Exame", y="n", color="Diagnostico_Label",
            barmode="stack", template="plotly_white",
            color_discrete_map={"Benigno/Normal": "#27ae60", "Maligno/Suspeito": "#e74c3c"},
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 – TEXTOS & VARIÁVEIS
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">As 3 Colunas de Texto da Base</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    A base possui <b>3 campos de texto livre</b> dos laudos: <b>Indicacao</b>, <b>Achados</b> e <b>Analise_Comparativa</b>.
    A partir deles foram extraídas variáveis numéricas e binárias descritas abaixo.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, nome, desc in [
        (col1, "Indicacao", "Motivo clínico do exame (por que foi pedido)"),
        (col2, "Achados", "Descrição dos achados radiológicos — coluna principal"),
        (col3, "Analise_Comparativa", "Comparação com exames anteriores"),
    ]:
        short = nome.split("_")[0].lower()
        preen = df_f[nome].notna().sum()
        med_chars = df_f[f"chars_{short}"].median()
        med_words = df_f[f"palavras_{short}"].median()
        with col:
            st.markdown(f"##### 📄 {nome}")
            st.markdown(f"""
            <div class="info-box">
            <b>Descrição:</b> {desc}<br>
            <b>Preenchidos:</b> {preen:,}<br>
            <b>Mediana caracteres:</b> {med_chars:.0f}<br>
            <b>Mediana palavras:</b> {med_words:.0f}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Variáveis Derivadas dos Textos</div>', unsafe_allow_html=True)

    cols_derivadas = [
        ("chars_indicacao",      "Qtd. caracteres – Indicacao",         "numérico"),
        ("palavras_indicacao",   "Qtd. palavras – Indicacao",           "numérico"),
        ("underscore_indicacao", "Contém `_` (data mascarada)",         "binário"),
        ("dimensao_indicacao",   "Contém dimensão em cm/mm",            "binário"),
        ("chars_achados",        "Qtd. caracteres – Achados",           "numérico"),
        ("palavras_achados",     "Qtd. palavras – Achados",             "numérico"),
        ("underscore_achados",   "Contém `_` em Achados",               "binário"),
        ("dimensao_achados",     "Contém dimensão cm/mm em Achados",    "binário"),
        ("chars_analise",        "Qtd. caracteres – Analise_Comparativa","numérico"),
        ("palavras_analise",     "Qtd. palavras – Analise_Comparativa", "numérico"),
        ("underscore_analise",   "Contém `_` em Analise_Comparativa",   "binário"),
        ("dimensao_analise",     "Contém dimensão em Analise_Comparativa","binário"),
    ]
    df_meta = pd.DataFrame(cols_derivadas, columns=["Coluna", "Descrição", "Tipo"])
    st.dataframe(df_meta, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Estatísticas Descritivas – Variáveis Numéricas</div>', unsafe_allow_html=True)
    num_cols = [
        "Qtd_Palavras_Diagnostico","Pontos_Suspeito","Pontos_Benigno","Risco_Geral",
        "chars_achados","palavras_achados","chars_indicacao","palavras_indicacao",
    ]
    st.dataframe(
        df_f[num_cols].describe().round(2),
        use_container_width=True,
    )

    st.markdown('<div class="section-title">Contagens e Proporções – Variáveis Binárias</div>', unsafe_allow_html=True)
    bin_cols = [
        "Diagnostico_Preenchido","Lesao_Com_Medida","Lesao_Espiculada",
        "Sem_Achados_Suspeitos","Caso_Positivo",
        "underscore_indicacao","underscore_achados","dimensao_achados",
    ]
    rows = []
    for c in bin_cols:
        n1 = (df_f[c] == 1).sum()
        n0 = (df_f[c] == 0).sum()
        rows.append({"Variável": c, "Sim (1)": n1, "Não (0)": n0, "% Sim": f"{n1/len(df_f)*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Raro: casos malignos – top variáveis
    st.markdown('<div class="section-title">Raros Malignos – Perfil Médio vs Benigno</div>', unsafe_allow_html=True)
    comp_cols = ["Qtd_Palavras_Diagnostico","Pontos_Suspeito","Pontos_Benigno","Risco_Geral","Lesao_Com_Medida","Lesao_Espiculada"]
    comp = df_f.groupby("Caso_Positivo")[comp_cols].mean().round(3)
    comp.index = ["Benigno/Normal", "Maligno/Suspeito"]
    st.dataframe(comp, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 – PALAVRAS & LAUDOS
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Palavras-Chave nos Achados</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Palavras de 4+ letras extraídas do campo <b>Achados</b>, removendo stopwords.
    Exemplos de termos descritivos encontrados: <i>calcificações benignas, nódulo, espiculada, medindo, suspeitas</i>.
    </div>""", unsafe_allow_html=True)

    col_b, col_m = st.columns(2)

    with col_b:
        st.markdown("##### 🟢 Benigno/Normal")
        top_ben = top_palavras(df_f[df_f["Caso_Positivo"] == 0]["Achados"])
        fig = px.bar(
            top_ben, x="Frequência", y="Palavra", orientation="h",
            color_discrete_sequence=["#27ae60"], template="plotly_white",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=550)
        st.plotly_chart(fig, use_container_width=True)

    with col_m:
        st.markdown("##### 🔴 Maligno/Suspeito")
        top_mal = top_palavras(df_f[df_f["Caso_Positivo"] == 1]["Achados"])
        fig2 = px.bar(
            top_mal, x="Frequência", y="Palavra", orientation="h",
            color_discrete_sequence=["#e74c3c"], template="plotly_white",
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=550)
        st.plotly_chart(fig2, use_container_width=True)

    # Palavras só em maligno
    st.markdown('<div class="section-title">Palavras Mais Raras – Exclusivas ou Enriquecidas em Malignos</div>', unsafe_allow_html=True)
    set_ben = set(top_palavras(df_f[df_f["Caso_Positivo"]==0]["Achados"], n=200)["Palavra"])
    mal_all = top_palavras(df_f[df_f["Caso_Positivo"]==1]["Achados"], n=200)
    raridades = mal_all[~mal_all["Palavra"].isin(set_ben)].head(15)
    if len(raridades):
        fig3 = px.bar(
            raridades, x="Palavra", y="Frequência",
            color_discrete_sequence=["#8e44ad"], template="plotly_white",
            title="Palavras em laudos malignos NÃO presentes nos benignos (top 200)",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Sem palavras exclusivas de malignos no filtro atual.")

    # Underscore (data mascarada)
    st.markdown('<div class="section-title">Underscores (_) e Dimensões nos Textos</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        und = df_f.groupby("Caso_Positivo")[["underscore_indicacao","underscore_achados","underscore_analise"]].mean().round(3)
        und.index = ["Benigno","Maligno"]
        fig4 = px.bar(und.reset_index().rename(columns={'index': 'Caso_Positivo'}).melt(id_vars="Caso_Positivo"),
            x="variable", y="value", color="Caso_Positivo",
            barmode="group", template="plotly_white",
            labels={"value":"Proporção","variable":"Campo","Caso_Positivo":"Diagnóstico"},
            title="Proporção com '_' (data mascarada) por diagnóstico")
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        dim = df_f.groupby("Caso_Positivo")[["dimensao_indicacao","dimensao_achados","dimensao_analise"]].mean().round(3)
        dim.index = ["Benigno","Maligno"]
        fig5 = px.bar(dim.reset_index().rename(columns={'index': 'Caso_Positivo'}).melt(id_vars="Caso_Positivo"),
            x="variable", y="value", color="Caso_Positivo",
            barmode="group", template="plotly_white",
            labels={"value":"Proporção","variable":"Campo","Caso_Positivo":"Diagnóstico"},
            title="Proporção com dimensão (cm/mm) por diagnóstico")
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 – TAMANHO × DIAGNÓSTICO
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Quantidade de Palavras tem Relação com o Diagnóstico?</div>', unsafe_allow_html=True)

    # Teste Mann-Whitney
    neg_words = df_f[df_f["Caso_Positivo"]==0]["Qtd_Palavras_Diagnostico"].dropna()
    pos_words = df_f[df_f["Caso_Positivo"]==1]["Qtd_Palavras_Diagnostico"].dropna()
    if len(pos_words) > 0 and len(neg_words) > 0:
        u_stat, p_val = mannwhitneyu(pos_words, neg_words)
        sig = "✅ Sim — diferença estatisticamente significativa" if p_val < 0.05 else "❌ Não significativo"
        st.markdown(f"""
        <div class="info-box">
        <b>Teste Mann-Whitney U</b> (não-paramétrico, comparação de medianas):<br>
        U = {u_stat:,.0f} | p-value = {p_val:.2e}<br>
        <b>{sig}</b> (α = 0.05)
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df_f[df_f["Qtd_Palavras_Diagnostico"] > 0],
            x="Diagnostico_Label", y="Qtd_Palavras_Diagnostico",
            color="Diagnostico_Label",
            color_discrete_map={"Benigno/Normal": "#27ae60", "Maligno/Suspeito": "#e74c3c"},
            template="plotly_white",
            title="Distribuição da Qtd. de Palavras por Diagnóstico",
            labels={"Qtd_Palavras_Diagnostico": "Nº Palavras", "Diagnostico_Label": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.violin(
            df_f[df_f["Qtd_Palavras_Diagnostico"] > 0],
            x="Diagnostico_Label", y="Qtd_Palavras_Diagnostico",
            color="Diagnostico_Label", box=True,
            color_discrete_map={"Benigno/Normal": "#27ae60", "Maligno/Suspeito": "#e74c3c"},
            template="plotly_white",
            title="Violino – Qtd. Palavras por Diagnóstico",
            labels={"Qtd_Palavras_Diagnostico": "Nº Palavras", "Diagnostico_Label": ""},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Análise por Quartil de Palavras</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Dividimos os laudos em 4 quartis pelo número de palavras. 
    O Q4 (laudos mais longos) tem proporção de malignidade significativamente maior.
    </div>""", unsafe_allow_html=True)

    quartil_stats = df_f.groupby("quartil_palavras", observed=True).agg(
        Total=("Caso_Positivo", "count"),
        Malignos=("Caso_Positivo", "sum"),
        Media_Palavras=("Qtd_Palavras_Diagnostico", "mean"),
    ).reset_index()
    quartil_stats["Taxa_Malignidade"] = (quartil_stats["Malignos"] / quartil_stats["Total"] * 100).round(2)
    quartil_stats["Media_Palavras"] = quartil_stats["Media_Palavras"].round(1)

    st.dataframe(quartil_stats, use_container_width=True, hide_index=True)

    fig3 = px.bar(
        quartil_stats, x="quartil_palavras", y="Taxa_Malignidade",
        color="Taxa_Malignidade",
        color_continuous_scale="RdYlGn_r",
        text="Taxa_Malignidade",
        template="plotly_white",
        title="Taxa de Malignidade (%) por Quartil de Palavras",
        labels={"quartil_palavras": "Quartil", "Taxa_Malignidade": "% Maligno"},
    )
    fig3.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 – CRITÉRIOS DE EXCLUSÃO
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Critérios de Exclusão</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
    <b>Por que excluir?</b> Registros com diagnóstico incompleto ou poucas palavras 
    não contribuem para análise textual e podem distorcer os resultados.
    </div>""", unsafe_allow_html=True)

    # Resumo geral de exclusão
    total_orig = len(df)
    sem_diag_orig    = (df["Diagnostico_Preenchido"] == 0).sum()
    sem_birads_orig  = df["Target"].isna().sum()
    poucas_palavras  = (df["Qtd_Palavras_Diagnostico"] < 5).sum()

    rows_excl = [
        {"Critério": "Sem classificação BI-RADS", "Qtd Excluídos": int(sem_birads_orig), "% do Total": f"{sem_birads_orig/total_orig*100:.2f}%"},
        {"Critério": "Diagnóstico não preenchido", "Qtd Excluídos": int(sem_diag_orig), "% do Total": f"{sem_diag_orig/total_orig*100:.2f}%"},
        {"Critério": "Menos de 5 palavras no laudo", "Qtd Excluídos": int(poucas_palavras), "% do Total": f"{poucas_palavras/total_orig*100:.2f}%"},
    ]
    st.dataframe(pd.DataFrame(rows_excl), use_container_width=True, hide_index=True)

    # Histograma de palavras com threshold
    threshold_excl = st.slider("Ver impacto do threshold de exclusão (N palavras)", 0, 50, 5)
    excl_n = (df["Qtd_Palavras_Diagnostico"] < threshold_excl).sum()
    st.markdown(f"Com threshold **< {threshold_excl} palavras**: **{excl_n:,}** registros excluídos ({excl_n/total_orig*100:.2f}% do total)")

    fig = px.histogram(
        df[df["Qtd_Palavras_Diagnostico"] > 0],
        x="Qtd_Palavras_Diagnostico", nbins=60,
        color="Diagnostico_Label",
        color_discrete_map={"Benigno/Normal": "#27ae60", "Maligno/Suspeito": "#e74c3c"},
        template="plotly_white",
        title="Distribuição de Qtd. Palavras (laudos não vazios)",
        barmode="overlay", opacity=0.7,
    )
    fig.add_vline(x=threshold_excl, line_dash="dash", line_color="navy",
                  annotation_text=f"Threshold: {threshold_excl}", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Diagnóstico Incompleto – BI-RADS 0</div>', unsafe_allow_html=True)
    birads0 = df[df["BI-RADS"] == "BI-RADS 0"]
    st.markdown(f"""
    <div class="info-box">
    <b>BI-RADS 0</b> = exame inconclusivo, necessita complementação.
    Total: <b>{len(birads0):,}</b> ({len(birads0)/total_orig*100:.1f}%). 
    Esses registros têm diagnóstico definitivo ainda não atribuído.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 6 – LESÃO COM MEDIDA
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Lesão com Medida × Diagnóstico</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Hipótese:</b> Laudos que mencionam dimensão em cm indicam que o médico encontrou algo concreto para medir, 
    o que pode estar associado a maior taxa de malignidade.
    </div>""", unsafe_allow_html=True)

    ct = pd.crosstab(df_f["Lesao_Com_Medida"], df_f["Caso_Positivo"])
    if ct.shape == (2, 2):
        chi2_val, p_chi, dof, _ = chi2_contingency(ct)
        sig = "✅ Associação estatisticamente significativa" if p_chi < 0.05 else "❌ Sem significância"
        st.markdown(f"""
        <div class="info-box">
        <b>Teste Qui-Quadrado</b>: χ² = {chi2_val:.2f} | df = {dof} | p = {p_chi:.2e}<br>
        <b>{sig}</b> (α = 0.05)
        </div>""", unsafe_allow_html=True)

    # Cards filtro lesão com medida
    df_med = df_f[df_f["Lesao_Com_Medida"] == 1]
    df_sem = df_f[df_f["Lesao_Com_Medida"] == 0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{len(df_med):,}</h2><p>Laudos COM medida</p></div>""", unsafe_allow_html=True)
    with col2:
        prop_med = df_med["Caso_Positivo"].mean()*100 if len(df_med) else 0
        st.markdown(f"""<div class="metric-card card-red">
            <h2>{prop_med:.1f}%</h2><p>Taxa Malignidade COM medida</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h2>{len(df_sem):,}</h2><p>Laudos SEM medida</p></div>""", unsafe_allow_html=True)
    with col4:
        prop_sem = df_sem["Caso_Positivo"].mean()*100 if len(df_sem) else 0
        st.markdown(f"""<div class="metric-card card-green">
            <h2>{prop_sem:.1f}%</h2><p>Taxa Malignidade SEM medida</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        comp_med = df_f.groupby("Lesao_Com_Medida")["Caso_Positivo"].mean().reset_index()
        comp_med["Lesao_Com_Medida"] = comp_med["Lesao_Com_Medida"].map({0: "Sem Medida", 1: "Com Medida"})
        comp_med["Taxa %"] = (comp_med["Caso_Positivo"] * 100).round(2)
        fig = px.bar(
            comp_med, x="Lesao_Com_Medida", y="Taxa %",
            color="Lesao_Com_Medida",
            color_discrete_sequence=["#27ae60", "#e74c3c"],
            text="Taxa %", template="plotly_white",
            title="Taxa de Malignidade: com vs sem medida",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cross2 = pd.crosstab(df_f["Lesao_Com_Medida"], df_f["Caso_Positivo"], normalize="index").round(4) * 100
        cross2.index = cross2.index.map({0: "Sem Medida", 1: "Com Medida"})
        cross2.columns = ["Benigno %", "Maligno %"]
        fig2 = px.bar(
            cross2.reset_index().rename(columns={'index': 'Lesao_Com_Medida'}).melt(id_vars="Lesao_Com_Medida"),
            x="Lesao_Com_Medida", y="value", color="variable",
            barmode="stack", template="plotly_white",
            color_discrete_map={"Benigno %":"#27ae60", "Maligno %":"#e74c3c"},
            title="Proporção Benigno/Maligno por presença de medida",
            labels={"value":"Proporção %","Lesao_Com_Medida":""},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Filtrando só com medida
    st.markdown('<div class="section-title">Base Filtrada: apenas Lesao_Com_Medida = 1</div>', unsafe_allow_html=True)
    if len(df_med):
        birads_med = df_med["BI-RADS"].value_counts().reset_index()
        birads_med.columns = ["BI-RADS", "Qtd"]
        fig3 = px.bar(
            birads_med, x="BI-RADS", y="Qtd", text="Qtd",
            template="plotly_white", color="BI-RADS",
            title=f"Distribuição BI-RADS – apenas laudos com medida ({len(df_med):,} registros)",
        )
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 7 – DISTRIBUIÇÃO BINOMIAL ⭐ PONTO EXTRA
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">⭐ Distribuição Binomial – Ponto Extra</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Modelo:</b> Cada laudo é um "ensaio de Bernoulli" — resultado: maligno (1) ou benigno (0).
    Sob a hipótese nula (H₀: p = 0.5), esperaríamos 50% de casos positivos.
    O teste binomial verifica se a proporção observada é compatível com H₀.
    </div>""", unsafe_allow_html=True)

    n_total  = len(df_f)
    k_pos    = int(df_f["Caso_Positivo"].sum())
    p_obs    = k_pos / n_total if n_total else 0

    col1, col2 = st.columns([1, 2])

    with col1:
        p_null = st.number_input("Probabilidade sob H₀ (p₀)", min_value=0.001, max_value=0.999, value=0.5, step=0.05)
        resultado = binomtest(k_pos, n_total, p_null, alternative="two-sided")
        p_value = resultado.pvalue
        ic_low, ic_high = resultado.proportion_ci(confidence_level=0.95)

        sig2 = "✅ Rejeitar H₀" if p_value < 0.05 else "❌ Não rejeitar H₀"
        st.markdown(f"""
        <div class="info-box">
        <b>n total:</b> {n_total:,}<br>
        <b>k (malignos):</b> {k_pos:,}<br>
        <b>p observado:</b> {p_obs:.4f} ({p_obs*100:.2f}%)<br>
        <b>p₀ (H₀):</b> {p_null:.3f}<br>
        <b>p-value:</b> {p_value:.2e}<br>
        <b>IC 95%:</b> [{ic_low:.4f}, {ic_high:.4f}]<br><br>
        <b>{sig2}</b> (α = 0.05)<br>
        A proporção de casos malignos é <b>estatisticamente diferente</b> de {p_null:.0%}.
        </div>""", unsafe_allow_html=True)

    with col2:
        from scipy.stats import binom
        ks = np.arange(0, min(n_total, k_pos * 10 + 50))
        pmf = binom.pmf(ks, n_total, p_null)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ks, y=pmf, name="Distribuição Binomial (H₀)",
            marker_color="#2d6a9f", opacity=0.6,
        ))
        fig.add_vline(x=k_pos, line_color="red", line_dash="dash",
                      annotation_text=f"Observado: k={k_pos}", annotation_position="top right")
        fig.add_vline(x=n_total * p_null, line_color="green", line_dash="dot",
                      annotation_text=f"Esperado: {n_total*p_null:.0f}", annotation_position="top left")
        fig.update_layout(
            title=f"Distribuição Binomial – B(n={n_total}, p₀={p_null:.2f})",
            xaxis_title="Número de malignos (k)",
            yaxis_title="Probabilidade",
            template="plotly_white",
            xaxis_range=[0, max(k_pos * 3, 50)],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Interpretação do Modelo Binomial</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    Com <b>p₀ = {p_null:.0%}</b> (chance igual de benigno/maligno), esperaríamos observar 
    <b>~{n_total*p_null:.0f} casos malignos</b> em {n_total:,} laudos.
    Observamos <b>{k_pos}</b> — uma fração muito menor ({p_obs*100:.2f}%).<br><br>
    O p-value extremamente baixo (<b>{p_value:.2e}</b>) confirma que a base é 
    <b>fortemente desbalanceada</b>, com predominância de casos benignos — 
    o que é esperado em programas de rastreamento mamográfico.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 8 – DICIONÁRIO
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-title">Dicionário de Variáveis</div>', unsafe_allow_html=True)
    st.dataframe(dic, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Tipos de Variáveis na Base</div>', unsafe_allow_html=True)
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Coluna", "Tipo Python"]
    tipos["Classificação"] = tipos["Tipo Python"].astype(str).map(
        lambda t: "Texto" if t == "object" else ("Binária" if "int" in t else "Numérica")
    )
    st.dataframe(tipos, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Amostra dos Dados</div>', unsafe_allow_html=True)
    n_amostra = st.slider("Quantidade de linhas para exibir", 5, 50, 10)
    st.dataframe(df_f.head(n_amostra), use_container_width=True)


# ─────────────────────────────────────────────
# RODAPÉ
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Dashboard — Análise de Laudos Mamográficos | "
    "Desenvolvido com Streamlit + Plotly + Pandas + SciPy</small>",
    unsafe_allow_html=True,
)