# Organização dos Notebooks

Este diretório contém os *notebooks* utilizados para entender os dados, criar os modelos e resolver o [problema proposto](../DescricaoAvaliacao.md). A organização do conteúdo tende a seguir a sequência cronológica dos passos realizados, com algumas exceções que fizeram com que passos adicionais fossem criados em numeração anterior.

O conteúdo dos *notebooks* é descrito a seguir.

 - [01 Estruturação](01_Estruturacao.ipynb): faz o primeiro acesso aos dados, identifica os elementos presentes e estabelece o critério de corte para separar os dados que serão usados para análises e treinamento de modelos dos dados que serão usados na avaliação final, de teste.
 - [02 Análise Exploratória](02.0_Analise_Exploratoria.ipynb): utiliza os dados de treinamento para se construir o entendimento dos dados e do domínio. As análises são mais focadas em limitações e potenciais de uso dos dados do que na capacidade preditiva deles.
     - [02.1 Análise Textual](02.1_Analise_Textual.ipynb): este notebook explora técnicas de representação de conteúdo textual para permitir o desenvolvimento do projeto. A análise é focada no uso de um modelo de *embeddings* e em seu  potencial de representação e capacidade preditiva.
 - [03_Classificação de Categorias](03.0_Classificacao_de_Categorias.ipynb): possui recursos para realizar experimentações com o objetivo de classificar produtos em categorias. 
     - [03.1 Histórico de Experimentos de Classificação de Categorias](03.1_Historico_de_Experimentos.ipynb): recupera o histórico de experimentações feitas e estabelece o mecanismo de reprodução e reuso dos experimentos, necessários para a produtização do modelo.
 - [04 Análise de Intenções de Busca](04.0_Analise_de_Intencoes_de_Busca.ipynb): realiza o estudo de como determinar intenções de busca a partir de um modelo não supervisionado.
     - [04.1 Calibração de Intenções de Busca](04.1_Calibracao_Intencoes_de_Busca.ipynb): a partir da metodologia definida no *notebook* de análise de intenções, estrutura a base de experimentação e construção dos modelos de intenções de busca, para calibrar seus parâmetros.
     - [04.2 Análise para a Classificação de Intenções](04.2_Analise_para_Classificacao_de_Intencoes.ipynb): como o modelo não supervisionado de definição de intenção de busca requer a análise da busca e de seus resultados, é preciso criar uma forma mais eficiente de determinação da intenção. Para isso, este notebook faz a análise de como extrair da sentença de busca as características que ajudam a determinar a intenção da pessoa que a criou.
     - [04.3_Classificação de Intenções](04.3_Classificacao_de_Intencoes.ipynb): cria a base para se fazer experimentações e criar modelos supervisionados para determinar a intenção de uma busca.
     - [04.4 Histórico e Análise de Experimentos de Classificação de Intenções de Busca](04.4_Historico_e_Analise_de_Experimentos.ipynb): recupera o histórico de experimentações e permite analisar os resultados do modelo criado em mais profundidade.
 - [05 Análise de Interação de Serviços](05.0_Analise_de_Integracao_de_Servicos.ipynb): realiza o estudo de como combinar os modelos criados nos passos anteriores para criar um recomendador de produtos a partir de buscas.
 - [06 Consumo de Produtização](06_Consumo_Produtizacao.ipynb): mantém um histórico de como utilizar a Command Line Interface para utilizar os serviços criados, com uma linha de atuação que se assemelha a testes automatizados.

Cabe ressaltar que todos os *notebooks* utilizam classes e funções implementadas e disponíveis no [diretório de códigos](../src).