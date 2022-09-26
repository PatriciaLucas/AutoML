# AUTOML PARA PREVISÃO DE SÉRIES TEMPORAIS

#### Projeto de tese de doutorado desenvolvido no [Laboratório MINDS](https://minds.eng.ufmg.br/) ([PPGEE - UFMG](https://www.ppgee.ufmg.br/)).

*Autora: Patrícia de Oliveira e Lucas* <a itemprop="sameAs" content="https://orcid.org/0000-0002-7334-8863" href="https://orcid.org/0000-0002-7334-8863" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

*Orientador: Frederico Gadelha Guimarães* <a itemprop="sameAs" content="https://orcid.org/0000-0001-9238-8839" href="https://orcid.org/0000-0001-9238-8839" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

*Coorientador: _Eduardo Mazzoni_ Andrade Marçal Mendes* <a itemprop="sameAs" content="https://orcid.org/0000-0002-3267-3862" href="https://orcid.org/0000-0002-3267-3862" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

*Data de início: 10/02/2022*

<img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figuras/principal_completa3_ufmg.jpg" width="150"/>   <img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figuras/salinas_horizontal_jpg.jpg" width="150"/>   <img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figuras/images.jpg" width="70"/>

<details><summary>Motivação:</summary><p>
	
A previsão de séries temporais é um problema de grande importância prática, fundamental nas ações de planejamento e controle ao prever padrões e detectar situações anormais futuras. Em áreas como meteorologia \citep{Coban2021PrecipitationTurkey}, finanças \citep{Sun2018}, agricultura \citep{deOliveiraeLucas2020ReferenceNetworks}, consumo de energia \citep{Savi2021Short-TermApproach} e transmissão de doenças \citep{Chimmula2020TimeNetworks}, essa tarefa possibilita reagir apropriadamente às manifestações de sistemas complexos, que muitas vezes dependem da capacidade de prever observações com base na história passada. Portanto, buscar formas de melhorar a precisão de modelos de previsão é de suma importância \citep{Oreshkin2021Meta-learningForecasting}. 

Além das questões que levaram ao surgimento de métodos AutoML já apresentadas, a previsão de séries temporais ainda requer habilidades técnicas específicas para lidar com as características peculiares desses dados como: tendência, sazonalidade, outliers, desvios e mudanças abruptas \citep{Paldino2021DoesForecasting}. O desafio de trabalhar com dados temporais fica evidente em competições como a M4\footnote{Última de uma série influente de competições de previsão organizadas por Spyros Makridakis desde 1982.} \citep{Makridakis2018TheForward}. Ao contrário de áreas como a visão computacional, a competição chega a conclusão de que ainda existem evidências de que algoritmos de ML e \textit{deep learning} (DL) lutam para superar as abordagens estatísticas clássicas de previsão de séries temporais.
	
Muitas ferramentas que automatizam tarefas de previsão de séries temporais já foram propostas. Porém, essas soluções estão longe de serem universais, já que é inviável incorporar todos os modelos disponíveis em apenas uma aplicação. Além disso, essas soluções não implementam todas as formas de previsão (pontual, intervalar e probabilística) e algumas trabalham apenas com séries temporais univariadas. Em \cite{Hyndman2008AutomaticR}, por exemplo, o foco são métodos clássicos, como ARIMA e suavização exponencial. A DeepAR da Amazon \citep{Salinas2020DeepAR:Networks} apresenta uma solução com redes recorrentes autoregressivas profundas para previsão probabilística. Em \cite{Oreshkin2019N-BEATS:Forecasting} a proposta é um modelo híbrido de modelos DL e autoregressivos para o problema de previsão pontual de séries temporais univariadas. 
	
Outro ponto importante é que muitas ferramentas não foram feitas especificamente para resolver problemas de previsão, não apresentando um componente de pré-processamento que lide com dados temporais, como em \citep{OlsonEvaluationScience,Thornton2012Auto-WEKA:Algorithms}. Em \cite{Nikitin2022AutomatedPipelines} e \cite{Shah2021AutoAI-TS:Forecasting} os autores desenvolveram soluções com foco em problemas de previsão que incluem pré-processamento para séries temporais com o mínimo de entradas pelo usuário. 
	
Apesar da variedade de soluções de AutoML, \cite{Paldino2021DoesForecasting} aponta que essas abordagens ainda não estão maduras o suficiente para lidar com tarefas de previsão. Resultados encontrados pelo autor concluíram que frameworks AutoML (AutoGluon, H2O, TPOT e Auto-sklearn) não superaram significativamente estratégias de previsão simples e convencionais (suavização ingênua e exponencial) em uma série de desafios de previsão. \cite{Nikitin2022AutomatedPipelines} também concorda que ainda existe espaço para melhorias, como inserção de suporte para processamento distribuído, explicabilidade e algoritmos de otimização mais robustos.
	
Durante o mestrado pude trabalhar com técnicas diferentes de previsão de séries temporais como Fuzzy Times Series, Temporal Convolutional Network (TCN), Long Short-Term Memory Networks (LSTM) e modelos clássicos. Também tive a oportunidade de estudar como melhorar a precisão desses modelos com uso de Ensemble Learning e otimização de hiperparâmetros. Portanto, propõe-se desenvolver uma abordagem de AutoML específica para previsão de séries temporais que contribua com soluções que diminuam as lacunas na área.
	
</p></details>

## Mapa da pesquisa

<img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figuras/mapa.png" width="600"/> 

### [AutoML](https://github.com/PatriciaLucas/AutoML/blob/main/automl.md) :triangular_flag_on_post:
- Componentes
 	- [Preparação dos dados]()
 	- [Engenharia de recursos](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/selecao_modelos.md)
	- [Seleção de modelos](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/engenharia_recursos.md)
	- [Otimização de hiperparâmetros](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/otimizacao_hiperparametros.md)
- [Frameworks](https://github.com/PatriciaLucas/AutoML/tree/main/frameworks)
- Séries Temporais
	- [Datasets](https://github.com/PatriciaLucas/AutoML/blob/main/series_temporais/datasets)
	- [Frameworks](https://github.com/PatriciaLucas/AutoML/blob/main/series_temporais/frameworks)
	- [PyFTS](https://github.com/PatriciaLucas/AutoML/blob/main/series_temporais/pyfts)


