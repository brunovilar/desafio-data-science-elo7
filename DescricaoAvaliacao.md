# Teste Data Science Elo7

Esse teste faz parte da etapa téorica do processo seletivo para a vaga no time Data Science do Elo7. O objetivo é avaliar como você desenvolve uma solução completa para um problema muito similar ao que poderia encontrar trabalhando conosco.

## Descrição do Problema

---

Durante o dia a dia de trabalho no Elo7 nos deparamos com uma quantidade de produtos únicos da ordem de seis milhões. As informações encontradas na página de um produto são inteiramente inseridas pelo seu vendedor. Sendo assim, temos dados textuais pouco estruturados e dados numéricos que podem ser incoerentes com a prática do mercado. Isso faz com que cada produto seja único, tornando mais difícil a busca por padrões. 

Considerando a interação dos compradores com o site, estes todos os dias utilizam nosso sistema de busca com dois objetivos principais: encontrar um produto desejado; ou descobrir novos itens que os surpreendam.

O termo de busca (ou query) é a informação que o usuário nos dá sobre a sua intenção e podemos obter informações extras após o usuário visualizar ou comprar algum produto. Será possível identificar algum tipo de intenção do usuário apenas pelo termo de busca?

Dito isso, queremos que você utilize nossos dados para atingir três objetivos:

- Criar um sistema supervisionado que classifique os produtos em categorias usando dados rotulados (campo `category`);
- Criar um sistema não supervisionado que utilize os termos de busca para produzir pelo menos duas classes de intenção do usuário;
- Criar um sistema híbrido (utilizando os dois primeiros) que ao receber um termo de busca:
    - Retorne a possível intenção do usuário;
    - Retorne a que categoria este termo pertenceria caso fosse um produto;
    - Recomende os dez produtos mais relevantes dados o termo de busca e a intenção do usuário.

Mais adiante você encontrará detalhes de como avaliaremos o seu desempenho.

## Dataset

O dataset escolhido para esse teste é composto de uma amostra de dados do Elo7. Você pode obter o dataset através desse [link](https://elo7-datasets.s3.amazonaws.com/data_scientist_position/elo7_recruitment_dataset.csv) (se tiver problemas para obter os dados, por favor, nos avise que ajudaremos você).

Em resumo, o dataset contém 38.507 registros distribuídos em 5 categorias (`Bebê`, `Bijuterias e Jóias`, `Decoração`, `Lembrancinhas`, `Papel e Cia` e `Outros`). Cada registro corresponde a um clique em um produto a partir de um termo de busca no site. 

Nesse dataset você encontrará as seguintes colunas:

- `product_id` - identificação de produto
- `seller_id` - identificação do vendedor 
- `query` - termo de busca inserido pelo usuário
- `search_page` - número da página que o produto apareceu nos resultados de busca (mín 1 e máx 5)
- `position` - número da posição que o produto apareceu dentro da página de busca (mín 0 e máx 38)
- `title` - título do produto  
- `concatenated_tags` - tags do produto inseridas pelo vendedor (as tags estão concatenadas por espaço)
- `creation_date` - data de criação do produto na plataforma do Elo7
- `price` - preço do produto em reais  
- `weight` - peso em gramas da unidade do produto reportado pelo vendedor
- `express_delivery` - indica se o produto é pronta entrega (1) ou não (0)
- `minimum_quantity` - quantidade de unidades mínima necessária para compra
- `view_counts` - número de cliques no produto nos últimos três meses  
- `order_counts` - número de vezes que o produto foi comprado nos últimos três meses
- `category` - categoria do produto   

## Construção da solução

A sua solução deve ser estruturada da seguinte maneira:

1. **Análise exploratória**
2. **Sistema de classificação de produtos**
3. **Sistema de termos de busca**
4. **Avaliação dos sistemas de classificação**
5. **Colaboração entre os sistemas**
6. **Entrega dos sistemas**

Abaixo vamos explicar melhor cada etapa evidenciando como iremos avaliar cada uma delas.

## Parte 1 - Análise exploratória

Essa etapa é uma das mais importantes de qualquer trabalho de um Cientista de Dados. Antes de executar qualquer algoritmo, nós precisamos primeiro entender o nosso problema. Por isso, queremos que você realize uma análise exploratória e indique as principais características presentes nos dados. Você tem total liberdade para escolher qualquer ferramenta ou algoritmo nessa etapa.

Para facilitar um pouco o fluxo de informações, recomendamos nessa etapa explicitar as perguntas que você deseja responder com as análises. Dessa forma, a análise ficará mais estruturada.

Use bastante criatividade ao longo do seu desenvolvimento! Muito provavelmente, essa análise fornecerá ideias para o sistema de classificação de produtos que você precisará desenvolver na etapa seguinte.

## Parte 2 - Sistema de Classificação de Produtos

Implemente um classificador de categorias (a nível de prova de conceito) que categorize os produtos. Escolha pelo menos uma característica textual e uma numérica para servir de entrada do seu modelo.

Não vamos indicar nenhum algoritmo ou técnica para executar essa tarefa, pois isso enviesaria a sua solução. Entretanto, é necessário que você documente todas as decisões tomadas (inclusive aquelas que deram errado). Não se preocupe em encontrar o melhor algoritmo para resolver o problema. Preferimos que você se preocupe em criar uma boa "história" com os dados, alternando código e o seu raciocínio (lembre de explicar suas escolhas).

## Parte 3 - Sistema de termos de busca

Implemente um sistema classificador que categorize os **termos de busca** em classes de intenção do usuário. Note que não há no dataset essa "intenção". Você deve determinar quais serão essas classes fazendo uma modelagem guiada pela investigação dos dados. Para te auxiliar seguem alguns exemplos de possíveis classes:

- Duas classes: usuário em fase de descoberta de produtos novos e usuário que já sabe que produto quer;
- Duas ou mais classes: usuário que compra produtos de determinadas faixas de preço. Ex: até R\\$ 25.00, de R\\$ 25.00 até R\\$ 100.00 e mais de R\\$ 100.00;
- Duas ou mais classes: Usuário que busca produtos de categorias específicas ou usuários com gostos mais gerais.

Este sistema é não supervisionado a princípio pela inexistência de rótulos da intenção do usuário nos dados que disponibilizamos, mas caso queira é permitido criar um método para rotulação e seguir com um sistema supervisionado (apenas lembre-se de fornecer as justificativas necessárias).

Por fim, seu sistema deve receber o termo de busca e retornar a classe de intenção do usuário que você criou. Os três pricipais pontos de avaliação nesta parte serão: o método de escolha das classes, o método de rotulação escolhido e a implementação do classificador.

## Parte 4 - Avaliação do Sistema de Classificação

Nessa etapa você deve definir as métrica(s) necessárias para avaliar os sistemas criados nas partes 2 e 3.

Essa etapa é muito importante para avaliar e comparar diferentes algoritmos. Podemos utilizar diversas métricas para testar diferentes hipóteses, mas dificilmente teremos uma métrica única para todos os problemas. Que técnica(s) você utilizará para avaliar o seu algoritmo?

Lembre-se de justificar sua escolha da(s) métrica(s) de avaliação, evite gastar tempo otimizando seus métodos para encontrar os melhores valores possíveis das métricas. Queremos principalmente avaliar como você testa um sistema.

*Obs*: Novamente, seja **criativo**!

## Parte 5 - Colaboração entre os sistemas

Agora que seus sistemas estão funcionando você deve fazer algo muito importante na vida de um Cientista de Dados, fazer com que esses sistemas trabalhem em conjunto. Uma integração interessante seria se seu sistema de recomendação fosse compatível de alguma forma com a classe de intenção do usuário criada artificialmente por você. Outra aplicação útil seria usar este sistema para recomendar produtos de maneira que dado o termo de busca como entrada o sistema recomendaria dez produtos para o usuário.

Você tem liberdade total para unir seus sistemas em um único, obedecendo apenas os critérios de funcionamento para o *sistema final*. A colaboração deve ser dada da seguinte maneira:
- Deve receber como entrada um termo de busca qualquer, não necessariamente apenas aqueles presentes no dataset;
- Deve ter três saídas:
  1. A categoria do termo de busca, como se esse fosse o texto de um produto. Note que aqui você não tem features numéricas para fazer a classificação, explore como contornar isso e documente bem a sua ideia;
  2. O nome da classe de intenção do usuário;
  3. Uma lista com o `id` e o `título` dos dez produtos recomendados.

Seguem algumas ideias que podem te auxiliar a desenvolver essa solução dos sistemas em colaboração. A primeira delas é determinar que tipo de recomendação você deseja criar. Seguem alguns exemplos:
- Recomendação de produtos similares ao que o usuário procura;
- Recomendação de produtos que podem ser complementares ao que usuário procura;
- Recomendação de produtos populares, sejam os mais procurados ou mais comprados.

A segunda é questionar o que obteríamos se usássemos o sistema de classificação de produtos para classificar os termos de busca em categorias? Por exemplo, isso pode ser usado para identificar qual a demanda atual dos usuários. Podemos descobrir que embora tenhamos vários produtos de decoração disponíveis os maiores volumes de busca no site são de produtos de lembrancinhas. Um sistema dessa espécie pode ser usado para alertar os vendedores do comportamento da demanda e assim orientar a sua adaptação.

Nessa etapa os principais pontos de avaliação serão: a criatividade, a robustez da integração dos sistemas e as técnicas usadas para criar as recomendações.

## Parte 6 - Entrega dos sistemas

Por fim, você deve entregar toda a sua investigação, experimentação, explicação e execução da solução documentada em um `jupyter notebook` (pode usar mais de um se julgar necessário), bem como um script em python que possa ser usado para executar todos os sistemas como nos exemplos a seguir:

1. Classificação de produtos

```shell
$ teste_ds.py --category "{'feature_1':<input_da_feature_1>,'feature_2':<input_da_feature_2>,...}"
>>> "<nome_da_categoria>"
```

2. Classificação de termo de busca

```shell
$ teste_ds.py --intent "<termo_de_busca_do_usuário>"
>>> "<nome_da_intenção_do_usuário>"
```

3. Sistema híbrido

```shell
$ teste_ds.py --recommendation "<termo_de_busca_do_usuário>"
>>> "<nome_da_categoria>"
>>> "<nome_da_intenção_do_usuário>"
>>> "<id1>,<título_do_produto_recomendado_1>"
>>> "<id2>,<título_do_produto_recomendado_2>"
>>> ...
>>> "<id10>,<título_do_produto_recomendado_10>"
```

### Observações gerais

Você deve submeter um ou mais [jupyter notebooks](http://jupyter.org/) com o código desenvolvido por você com a solução desse desafio. Lembre-se de documentar seu código e utilizar células _Markdown_ para explicar **detalhadamente** sua solução. Lembre-se de explicar seu raciocínio e justificar os métodos utilizados por você. Explicite os algoritmos utilizados e as etapas de pré-processamento que você recomenda fazer, justificando o porquê de cada uma das decisões tomadas.

[Aqui](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels) você pode ver uma lista de linguagens compatíveis com o *jupyter* e [aqui](https://ipython.readthedocs.io/en/latest/install/kernel_install.html) algumas instruções que podem auxiliar na instalação da mesma.

Suba os arquivos finais `.ipynb`, um `README.md` e um `requirements.txt` (gerado pelo comando `pip freeze > requirements.txt`) em seu github pessoal. Deixe o repositório público e nos mande o link por email. Caso tenha utilizado uma linguagem diferente de *Python*, nos **explique** como rodar o seu projeto localmente. Se preferir a utilização do Python, por favor submeta o teste utilizando o Python 3+.

Sinta-se à vontade para fazer o uso de bibliotecas (como o [scikit-learn](http://scikit-learn.org/) e [scipy](https://www.scipy.org/)), mas, novamente, você deve saber explicar o porquê de você aplicar determinado algoritmo para determinada situação.

Não queremos a solução ideal para o problema! Queremos entender sua forma de pensar. =)

**IMPORTANTE**: Seja criativo na resolução do problema! O trabalho de um Cientista de Dados envolve conhecimento técnico, metodologia científica e muita criatividade para abordar problemas complexos. Procure formular uma hipótese, crie o algoritmo de categorização de produtos e encontre uma métrica que teste a sua hipótese.

Boa sorte e qualquer dúvida, pode nos mandar um e-mail!
