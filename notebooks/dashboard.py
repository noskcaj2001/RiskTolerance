#%%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pickle import load
import cvxopt as opt
from cvxopt import blas, solvers

# Configuração da página
st.set_page_config(layout="wide", page_title="Robô Assistente para Alocação de Investimentos e Perfomance de Carteira")

# Carregar dados
@st.cache_data
def load_data():
    investidores = pd.read_csv('InputData.csv', index_col=0)
    
    ativos = pd.read_csv('SP500Data.csv', index_col=0)
    fracoes_faltantes = ativos.isnull().mean().sort_values(ascending=False)
    lista_remover = sorted(list(fracoes_faltantes[fracoes_faltantes > 0.3].index))
    ativos.drop(labels=lista_remover, axis=1, inplace=True)
    ativos = ativos.fillna(method='ffill')
    
    return investidores, ativos

investidores, ativos = load_data()

# Preparar opções do dropdown
opcoes = []
for ticker in ativos.columns:
    opcoes.append(ticker)

# Funções
def prever_tolerancia_risco(X_input):
    filename = 'finalized_model.sav'
    modelo_carregado = load(open(filename, 'rb'))
    predicoes = modelo_carregado.predict(X_input)
    return predicoes

def obter_alocacao_ativos(tolerancia_risco, tickers_selecionados):
    ativos_selecionados = ativos.loc[:, tickers_selecionados]
    vetor_retorno = np.array(ativos_selecionados.pct_change().dropna(axis=0)).T
    n = len(vetor_retorno)
    retornos = np.asmatrix(vetor_retorno)
    mus = 1 - tolerancia_risco
    
    # Converter para matrizes cvxopt
    S = opt.matrix(np.cov(vetor_retorno))
    pbar = opt.matrix(np.mean(vetor_retorno, axis=1))
    # Criar matrizes de restrição
    G = -opt.matrix(np.eye(n))   # matriz identidade negativa n x n
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calcular pesos da fronteira eficiente usando programação quadrática
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    Alocacao = pd.DataFrame(data=np.array(portfolios['x']), index=ativos_selecionados.columns)

    retornos_finais = (np.array(ativos_selecionados) * np.array(w))
    soma_retornos = np.sum(retornos_finais, axis=1)
    retorno_soma_pd = pd.DataFrame(soma_retornos, index=ativos.index)
    retorno_soma_pd = retorno_soma_pd - retorno_soma_pd.iloc[0,:] + 100   
    return Alocacao, retorno_soma_pd

# Layout da UI
st.title('Robô Assistente de Ações | + 3 no Cartola')

col1, col2 = st.columns([3, 7])

with col1:
    st.header("Passo 1: Informe as características do investidor")
    
    idade = st.slider('Idade:', 
                   min_value=int(investidores['AGE07'].min()), 
                   max_value=70, 
                   value=25,
                   step=5,
                   format="%d")
    
    patrimonio = st.slider('Patrimônio Líquido:', 
                     min_value=-1000000, 
                     max_value=3000000, 
                     value=10000,
                     step=50000,
                     format="R$%d")
    
    renda = st.slider('Renda:', 
                     min_value=-1000000, 
                     max_value=3000000, 
                     value=100000,
                     step=50000,
                     format="R$%d")
    
    educacao = st.slider('Nível de Educação (escala de 4):', 
                   min_value=int(investidores['EDCL07'].min()), 
                   max_value=int(investidores['EDCL07'].max()), 
                   value=2,
                   step=1)
    
    estado_civil = st.slider('Casado:', 
                       min_value=int(investidores['MARRIED07'].min()), 
                       max_value=int(investidores['MARRIED07'].max()), 
                       value=1,
                       step=1)
    
    filhos = st.slider('Filhos:', 
                    min_value=int(investidores['KIDS07'].min()), 
                    max_value=int(investidores['KIDS07'].max()), 
                    value=3,
                    step=1)
    
    ocupacao = st.slider('Ocupação:', 
                   min_value=int(investidores['OCCAT107'].min()), 
                   max_value=int(investidores['OCCAT107'].max()), 
                   value=3,
                   step=1)
    
    risco = st.slider('Disposição para assumir riscos:', 
                    min_value=int(investidores['RISK07'].min()), 
                    max_value=int(investidores['RISK07'].max()), 
                    value=3,
                    step=1)
    
    if st.button('Calcular Tolerância ao Risco'):
        X_input = [[idade, educacao, estado_civil, filhos, ocupacao, renda, risco, patrimonio]]
        tolerancia_risco = prever_tolerancia_risco(X_input)
        st.session_state.tolerancia_risco = round(float(tolerancia_risco*100), 2)

with col2:
    st.header("Passo 2: Alocação de ativos e desempenho da carteira")
    
    if 'tolerancia_risco' in st.session_state:
        st.text_input('Tolerância ao Risco (escala de 100):', value=st.session_state.tolerancia_risco, key='exibicao_tolerancia_risco')
    
    ativos_selecionados = st.multiselect('Selecione os ativos para a carteira:', 
                                    options=opcoes, 
                                    default=['GOOGL', 'FB', 'GS', 'MS', 'GE', 'MSFT'])
    
    if st.button('Enviar') and 'tolerancia_risco' in st.session_state and ativos_selecionados:
        tolerancia_risco = float(st.session_state.tolerancia_risco)/100
        Alocacao, RetornoInvestimento = obter_alocacao_ativos(tolerancia_risco, ativos_selecionados)
        
        # Criar gráficos
        fig1 = go.Figure(go.Bar(
            x=Alocacao.index,
            y=Alocacao.iloc[:,0],
            marker=dict(color='red')
        ))
        fig1.update_layout(title="Alocação de ativos - Alocação Média-Variância")
        
        fig2 = go.Figure(go.Scatter(
            x=RetornoInvestimento.index,
            y=RetornoInvestimento.iloc[:,0],
            name='Retorno (%)',
            marker=dict(color='red')
        ))
        fig2.update_layout(title="Valor da carteira para um investimento de R$100")
        
        # Exibir gráficos
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
# %%
