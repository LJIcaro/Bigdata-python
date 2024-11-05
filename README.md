# Bigdata-python
Repositorio contruido como hospedagem do trabalho avaliativo semestral da disciplina Tópicos de BigData em Python, ministrada pelo Prof. Roney Camargo, Malaguti, no Centro Universitário Ruy Barbosa

# Alunos envolvidos no projeto
- Ícaro Lima: 202308425951
- Ruan Müller: 202302372546
- Wesley Reis: 202202212075
- Alan Goes: 202403797291
- Luan Masao: 202308337696
- Cleiton Sousa: 202201017058

# Instruções para Criar um Ambiente Virtual e Instalar Dependências

## Passo a Passo

1. **Abra o terminal (Prompt de Comando ou PowerShell)**:
   - Se estiver usando o Windows, abra o Prompt de Comando ou PowerShell. Se preferir usar o WSL, abra o terminal do WSL.

2. **Navegue até a pasta do seu projeto**:
   - Use o comando `cd` para entrar no diretório onde está o seu projeto. Por exemplo:
     ```bash
     cd C:\Users\icaro.jesus\Desktop\python bigdata
     ```
     (Ajuste o caminho conforme necessário.)

3. **Crie o ambiente virtual**:
   - Execute o seguinte comando para criar um ambiente virtual chamado `venv`:
     ```bash
     python -m venv venv
     ```
     Isso criará uma pasta chamada `venv` dentro do seu diretório do projeto.

4. **Ative o ambiente virtual**:
   - **No Windows**:
     - Para ativar o ambiente virtual no Windows, use:
       ```bash
       .\venv\Scripts\activate
       ```
   - **No WSL**:
     - Para ativar o ambiente virtual no WSL, use:
       ```bash
       source venv/bin/activate
       ```
     Após a ativação, você verá o nome do ambiente virtual (por exemplo, `(venv)`) na frente da linha de comando.

5. **Instale as dependências a partir do `requirements.txt`**:
   - Uma vez que o ambiente virtual esteja ativo, você pode instalar as dependências listadas no arquivo `requirements.txt` com o seguinte comando:
     ```bash
     pip install -r requirements.txt
     ```




# Detalhameneto do Código-Fonte

## Importação das bibliotecas
```
import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from prophet import Prophet
```

* Importa as bibliotecas necessárias para manipulação de dados (pandas, numpy), visualização (plotly), aprendizado de máquina (sklearn), balanceamento de dados (imblearn) e modelagem preditiva (prophet).


## Função carregdar_dados
```
def carregar_dados(filepath):
    df = pd.read_csv(filepath)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y, %H:%M:%S')
    df['deathDate'] = pd.to_datetime(df['deathDate'], format='%d/%m/%Y, %H:%M:%S', errors='coerce')
    df.rename(columns={'main_reason': 'motivos'}, inplace=True)
    return df
```

* Carrega os dados de um arquivo CSV.
* Converte as colunas de data (data e deathDate) para o formato datetime.
* Renomeia a coluna main_reason para motivos.
* Retorna o DataFrame processado.

## Carregamento e Seleção de Dados
```
df = carregar_dados('fogo_cruzado_2022_2024.csv')
top_bairros = df['neighborhood'].value_counts().nlargest(35)
df = df[df['neighborhood'].isin(top_bairros.index)]
```

* Carrega os dados do arquivo CSV.
* Seleciona os 35 bairros com o maior número de incidentes.
* Filtra o DataFrame para incluir apenas esses bairros.

## Gráfico: Número de Incidentes Por Bairros
```
top_bairros = df['neighborhood'].value_counts().nlargest(35)
df_top_bairros = pd.DataFrame({'Bairro': top_bairros.index, 'Número de Incidentes': top_bairros.values})

fig1 = px.bar(
    df_top_bairros,
    x='Número de Incidentes', 
    y='Bairro',
    orientation='h',
    title="Número de Incidentes por Bairro",
    labels={'Número de Incidentes': 'Número de Incidentes', 'Bairro': 'Bairro'},
    hover_data={'Número de Incidentes': True}
)
fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=1000)
fig1.show()
```

* Cria um gráfico de barras horizontal mostrando o número de incidentes por bairro.
* Ordena os bairros de forma decrescente pelo número de incidentes.
* Ajusta a altura do gráfico para melhor visualização.

## Gráfico: Tipos de Ocorrências Por Bairros
```
df_grouped = df.groupby(['neighborhood', 'motivos']).size().unstack(fill_value=0)
df_grouped = df_grouped.loc[top_bairros.index]

fig2 = go.Figure()
for motivo in df_grouped.columns:
    fig2.add_trace(go.Bar(
        y=df_grouped.index,
        x=df_grouped[motivo],
        name=motivo,
        orientation='h'
    ))

fig2.update_layout(
    barmode='stack', 
    title='Tipos de Ocorrências por Bairro',
    yaxis={'categoryorder': 'total ascending'},
    xaxis_title='Número de Ocorrências',
    yaxis_title='Bairro',
    height=1000
)
fig2.show()
```

* Agrupa os dados por bairro e tipo de motivo, criando uma tabela de contingência.
* Cria um gráfico de barras empilhadas horizontal mostrando os tipos de ocorrências por bairro.
* Ajusta a altura do gráfico para melhor visualização.

## Função engenharia_de_recursos
```
def engenharia_de_recursos(df):
    df['hour'] = df['data'].dt.hour
    df['day_of_week'] = df['data'].dt.dayofweek
    df['month'] = df['data'].dt.month
    return df
```
* Adiciona colunas ao DataFrame para representar a hora, o dia da semana e o mês do incidente.
* Retorna o DataFrame atualizado.

## Preparação dos Dados para Classificação
```
df = engenharia_de_recursos(df)

def preparar_dados(df):
    features = ['latitude', 'longitude', 'hour', 'day_of_week', 'month', 'age']
    X = df[features]
    y = df['motivos']

    le = LabelEncoder()
    y = le.fit_transform(y)

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le

X, y, le = preparar_dados(df)
```

* Aplica engenharia de recursos ao DataFrame.
* Seleciona as colunas de características (features) e a coluna alvo (motivos).
* Codifica a coluna alvo (motivos) usando LabelEncoder.
* Imputa valores faltantes nas características usando a média.
* Normaliza as características usando StandardScaler.
* Retorna as características (X), a coluna alvo (y) e o LabelEncoder.

## Tratamento de Desequilíbrio de Classes 
```
def tratar_desequilibrio(X, y):
    class_counts = np.bincount(y)
    valid_classes = np.where(class_counts >= 2)[0]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    smote = SMOTE(random_state=42, k_neighbors=1)
    return smote.fit_resample(X, y)

X_res, y_res = tratar_desequilibrio(X, y)
```
* Remove classes com menos de 2 amostras.
* Aplica a técnica SMOTE para balancear as classes.
* Retorna as características (X_res) e a coluna alvo (y_res) balanceadas.

## Treinamento do Modelo de Classificação
```
def treinar_modelo(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
model = treinar_modelo(X_train, y_train)
```
* Define uma grade de parâmetros para o RandomForestClassifier.
* Realiza uma busca em grade (GridSearchCV) para encontrar os melhores parâmetros.
* Divide os dados em conjuntos de treino e teste.
* Treina o modelo usando os melhores parâmetros encontrados.
* Retorna o melhor estimador.

## Avaliação do Modelo de Classificação
```
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Acurácia:", accuracy_score(y_test, y_pred))
```
* Faz previsões no conjunto de teste.
* Imprime o relatório de classificação (classification_report) e a acurácia (accuracy_score).

## Agrupamento dos Incidentes por Similaridade
```
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

df['Cluster'] = clusters
```
* Aplica o algoritmo K-Means para agrupar os incidentes em 5 clusters.
* Adiciona a coluna Cluster ao DataFrame original.

## Gráfico: Ocorrências por Dia da Semana e Horário
```
def grafico_ocorrencias_dia_hora(df):
    df['dia_semana'] = df['data'].dt.day_name().map({
        'Monday': 'Segunda-feira',
        'Tuesday': 'Terça-feira',
        'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira',
        'Friday': 'Sexta-feira',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    })
    df['hora'] = df['data'].dt.hour
    ocorrencias_dia_hora = df.groupby(['dia_semana', 'hora']).size().reset_index(name='ocorrencias')
    fig = px.density_heatmap(ocorrencias_dia_hora, x='hora', y='dia_semana', z='ocorrencias', 
                             title='Ocorrências por Dia da Semana e Horário', 
                             labels={'hora': 'Hora do Dia', 'dia_semana': 'Dia da Semana', 'ocorrencias': 'Número de Ocorrências'},
                             color_continuous_scale='Viridis')
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(
            categoryorder='array',
            categoryarray=['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        ),
        coloraxis_colorbar=dict(
            title='Número de Ocorrências'
        )
    )
    fig.update_traces(
        hovertemplate='Hora: %{x}:00-%{x}:59<br>Dia: %{y}<br>Número de Ocorrências: %{z}<extra></extra>'
    )
    return fig

fig4 = grafico_ocorrencias_dia_hora(df)
fig4.show()
```
* Traduz os dias da semana para português.
* Cria um gráfico de mapa de calor mostrando as ocorrências por dia da semana e horário.

## Gráfico: Ocorrências ao Longo do Tempo por Bairro
```
def grafico_ocorrencias_tempo_bairro(df):
    df['ano_mes'] = df['data'].dt.to_period('M').astype(str)
    ocorrencias_tempo_bairro = df.groupby(['ano_mes', 'neighborhood']).size().reset_index(name='ocorrencias')
    fig = px.line(ocorrencias_tempo_bairro, x='ano_mes', y='ocorrencias', color='neighborhood',
                  title='Ocorrências ao Longo do Tempo por Bairro',
                  labels={'ano_mes': 'Ano e Mês', 'ocorrencias': 'Número de Ocorrências', 'neighborhood': 'Bairro'})
    return fig

fig5 = grafico_ocorrencias_tempo_bairro(df)
fig5.show()
```
* Cria uma coluna ano_mes representando o ano e o mês do incidente.
* Agrupa os dados por ano_mes e neighborhood.
* Cria um gráfico de linha mostrando as ocorrências ao longo do tempo por bairro.

## Gráfico: Ocorrências por Tipo de Crime e Bairro
```
def grafico_ocorrencias_tipo_bairro(df):
    ocorrencias_tipo_bairro = df.groupby(['motivos', 'neighborhood']).size().reset_index(name='ocorrencias')
    fig = px.bar(ocorrencias_tipo_bairro, x='neighborhood', y='ocorrencias', color='motivos',
                 title='Ocorrências por Tipo de Crime e Bairro',
                 labels={'neighborhood': 'Bairro', 'ocorrencias': 'Número de Ocorrências', 'motivos': 'Tipo de Crime'})
    return fig

fig6 = grafico_ocorrencias_tipo_bairro(df)
fig6.show()
```
* Agrupa os dados por motivos e bairros.
* Cria um gráfico de barras mostrando as ocorrências por tipo de crime e bairro.


## Gráfico: Idade das Vítimas por Tipo de Crime
```
def grafico_idade_tipo_crime(df):
    fig = px.scatter(df, x='age', y='motivos', color='motivos',
                     title='Idade das Vítimas por Tipo de Crime',
                     labels={'age': 'Idade', 'motivos': 'Tipo de Crime'})
    return fig

fig8 = grafico_idade_tipo_crime(df)
fig8.show()
```

* Cria um gráfico de dispersão mostrando a relação entre a idade das vítimas e os tipos de crimes.

# Gráfico 9: Ocorrências por Mês e Tipo de Crime
```
def grafico_ocorrencias_mes_tipo(df):
    ocorrencias_mes_tipo = df.groupby(['month', 'motivos']).size().reset_index(name='ocorrencias')
    fig = px.bar(ocorrencias_mes_tipo, x='month', y='ocorrencias', color='motivos',
                 title='Ocorrências por Mês e Tipo de Crime',
                 labels={'month': 'Mês', 'ocorrencias': 'Número de Ocorrências', 'motivos': 'Tipo de Crime'})
    return fig

fig7 = grafico_ocorrencias_mes_tipo(df)
fig7.show()
```

* Agrupa os dados por month e motivos.
* Cria um gráfico de barras mostrando as ocorrências por mês e tipo de crime.

# Função previsao_tendencia_bairros
```
def previsao_tendencia_bairros(df):
    df['ano_mes'] = df['data'].dt.to_period('M').astype(str)
    ocorrencias_tempo_bairro = df.groupby(['ano_mes', 'neighborhood']).size().reset_index(name='ocorrencias')
    
    previsoes = []
    for bairro in ocorrencias_tempo_bairro['neighborhood'].unique():
        df_bairro = ocorrencias_tempo_bairro[ocorrencias_tempo_bairro['neighborhood'] == bairro]
        df_bairro = df_bairro.rename(columns={'ano_mes': 'ds', 'ocorrencias': 'y'})
        df_bairro['ds'] = pd.to_datetime(df_bairro['ds'])
        
        modelo = Prophet()
        modelo.fit(df_bairro)
        
        futuro = modelo.make_future_dataframe(periods=12, freq='M')
        previsao = modelo.predict(futuro)
        
        previsao['neighborhood'] = bairro
        previsoes.append(previsao[['ds', 'yhat', 'neighborhood']])
    
    previsoes_df = pd.concat(previsoes)
    
    fig = px.line(previsoes_df, x='ds', y='yhat', color='neighborhood',
                  title='Previsão de Tendência de Ocorrências por Bairro',
                  labels={'ds': 'Ano e Mês', 'yhat': 'Previsão de Ocorrências', 'neighborhood': 'Bairro'})
    return fig

fig_previsao = previsao_tendencia_bairros(df)
fig_previsao.show()
```
* Prepara os dados para o Prophet, criando uma coluna ano_mes.
* Agrupa os dados por ano_mes e neighborhood.
* Para cada bairro, treina um modelo Prophet e faz previsões para os próximos 12 meses.
* Concatena todas as previsões em um DataFrame.
* Cria um gráfico de linha mostrando a previsão de tendência de ocorrências por bairro.
