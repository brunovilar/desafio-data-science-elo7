# Desenvolvimento do Teste de Data Science para o Elo7

Este repositório contém o desenvolvimento de um projeto a ser avaliado no processo seletivo para o Elo7. A descrição do do contexto, definição do problema e apresentação dos critérios podem ser encontradas na [Descrição da Avaliação](DescricaoAvaliacao.md). 

Esta versão contém a implementação de uma primeira abordagem para lidar com todos os problemas estabelecidos e estabelecer uma entrega completa. O desenvolvimento foi realizado de modo a permitir entender os dados, seus potenciais e limitações, criar as estruturas de experimentação para cada problema e desenvolver os elementos básicos que simulam o processo de produtização de cada solução. Como é uma visão inicial, cada etapa, documentada em um *notebook*, possui a descrição do raciocínio utilizado para o desenvolvimento e os pontos de melhoria que foram identificados para versões futuras.

## Configuração do Ambiente de Desenvolvimento


Clonar o projeto:

```
git clone https://github.com/brunovilar/teste_HT 
```

Criar *virtual environment* do Python para o projeto.

```
mkdir virtual_envs
python3 -m venv virtual_envs/teste_HT
```

Ativar *virtual environment* e instalar módulos necessários:

```
source virtual_envs/teste_HT/bin/activate
pip install --upgrade setuptools
pip install -r teste_HT/requirements.txt
```

Baixar e extrair o modelo pré-treinado do [Word2Vec](https://fasttext.cc/docs/en/crawl-vectors.html) em Português:

```
mkdir teste_HT/models/
wget -O teste_HT/models/cc.pt.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz
gunzip teste_HT/models/cc.pt.300.bin.gz
rm teste_HT/models/cc.pt.300.bin.gz
```

## Execução do Projeto

Ativar o *virtual environment* do projeto:

```
source virtual_envs/teste_HT/bin/activate
```

Iniciar o Jupyter Notebook:

```
cd teste_HT
jupyter lab
```

## Visualização Histórico de Experimentos na Interface do MLflow

Ativar o *virtual environment* do projeto:

```
source virtual_envs/teste_HT/bin/activate
```

Iniciar o MLflow:

```
cd teste_HT/notebooks
mlflow ui
```

Acessar o endereço no navegador: http://localhost:5000.

## Criação dos Recursos para Execução dos Sistemas

Os principais resultados do projeto podem ser verificados em funcionamento utilizando a aplicação via CLI ([cli/teste_ds.py](cli/teste_ds.py)) ou executando o *notebook* que automatiza a sua utilização ([Consumo da Produtização](notebooks/06_Consumo_Produtizacao.ipynb)). Para ser capaz de executar a apliação, é preciso executar alguns dos *notebooks* para garantir que os conjuntos de dados estejam disponíveis e que os modelos estejam criados. Para isso, a sequência de passos a ser realizada é:

 - Executar os notebooks:
     - [01_Estruturacao.ipynb](notebooks/01_Estruturacao.ipynb) para baixar o conjunto de dados e separar o conteúdo entre treinamento e teste.
     - [03.0_Classificacao_de_Categorias.ipynb](notebooks/03.0_Classificacao_de_Categorias.ipynb) para criar o modelo de Classificação de Produtos em Categorias.
     - [04.1_Calibracao_Intencoes_de_Busca.ipynb](notebooks/04.1_Calibracao_Intencoes_de_Busca.ipynb) para criar o modelo não supervisionado de agrupamento de produtos que permite determinar as intenções de busca a partir do histórico de resultados com interações.
     - [04.3_Classificacao_de_Intencoes.ipynb](notebooks/04.3_Classificacao_de_Intencoes.ipynb) para criar um modelo supervisionado que define a intenção de busca sem depender do histórico de interações com resultados de busca.
 - Configurar a chave dos modelos criados a partir do MLflow:
     - Usar a interface do [MlFlow](README.md#Visualizar-Histórico-de-Experimentos-na-Interface-do-MLFlow) para obter o *run_id* de cada experimento;
     - Atualizar, no arquivo [settings.py](src/settings.py), o *run_id* das constantes:
         > CATEGORY_CLASSIFICATION_RUN_ID
         > 
         > USUPERVISED_INTENT_CLASSIFICATION_RUN_ID
         > 
         > SUPERVISED_INTENT_CLASSIFICATION_RUN_ID
     - Executar o notebook [06_Consumo_Produtizacao.ipynb](notebooks/06_Consumo_Produtizacao.ipynb) para visualizar o resultado dos 3 modelos criados.
