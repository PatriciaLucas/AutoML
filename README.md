# AUTOML FOR TIME SERIES FORECASTING

#### Doctoral Thesis Project ([PPGEE - UFMG](https://www.ppgee.ufmg.br/)).

<img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figures/principal_completa3_ufmg.jpg" width="150"/>   <img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figures/salinas_horizontal_jpg.jpg" width="150"/>   <img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figures/images.jpg" width="70"/>

*Author: Patrícia de Oliveira e Lucas* <a itemprop="sameAs" content="https://orcid.org/0000-0002-7334-8863" href="https://orcid.org/0000-0002-7334-8863" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

*Advisor: Frederico Gadelha Guimarães* <a itemprop="sameAs" content="https://orcid.org/0000-0001-9238-8839" href="https://orcid.org/0000-0001-9238-8839" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

*Co-advisor: _Eduardo Mazzoni_ Andrade Marçal Mendes* <a itemprop="sameAs" content="https://orcid.org/0000-0002-3267-3862" href="https://orcid.org/0000-0002-3267-3862" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>

*Start date: 10/02/2022*



<details><summary>Motivation:</summary><p>
	
Time series forecasting is a problem of significant practical importance, fundamental for planning and control actions as it predicts patterns and detects future abnormal situations. In fields such as meteorology \citep{Coban2021PrecipitationTurkey}, finance \citep{Sun2018}, agriculture \citep{deOliveiraeLucas2020ReferenceNetworks}, energy consumption \citep{Savi2021Short-TermApproach}, and disease transmission \citep{Chimmula2020TimeNetworks}, this task enables appropriate responses to complex system manifestations, which often depend on the ability to predict observations based on past history. Thus, finding ways to improve the accuracy of forecasting models is of utmost importance \citep{Oreshkin2021Meta-learningForecasting}.

In addition to the challenges that prompted the development of existing AutoML methods, time series forecasting requires specific technical skills to address the peculiar characteristics of such data, including trends, seasonality, outliers, deviations, and abrupt changes \citep{Paldino2021DoesForecasting}. The complexity of working with temporal data is evident in competitions like the M4\footnote{The latest in an influential series of forecasting competitions organized by Spyros Makridakis since 1982.} \citep{Makridakis2018TheForward}. Unlike fields such as computer vision, the competition concludes that machine learning (ML) and deep learning (DL) algorithms still struggle to outperform classical statistical approaches for time series forecasting.

Many tools have been proposed to automate time series forecasting tasks. However, these solutions are far from universal, as incorporating all available models into a single application is unfeasible. Moreover, these tools do not implement all forms of forecasting (point, interval, and probabilistic), and some work exclusively with univariate time series. For instance, \cite{Hyndman2008AutomaticR} focuses on classical methods such as ARIMA and exponential smoothing. Amazon's DeepAR \citep{Salinas2020DeepAR:Networks} offers a solution using deep autoregressive recurrent networks for probabilistic forecasting. \cite{Oreshkin2019N-BEATS:Forecasting} proposes a hybrid model combining DL and autoregressive models for the point forecasting of univariate time series.

Another critical point is that many tools are not explicitly designed to solve forecasting problems and lack preprocessing components tailored to temporal data, as seen in \citep{OlsonEvaluationScience,Thornton2012Auto-WEKA:Algorithms}. In \cite{Nikitin2022AutomatedPipelines} and \cite{Shah2021AutoAI-TS:Forecasting}, the authors developed solutions focused on forecasting problems that include preprocessing for time series with minimal user input.

Despite the variety of AutoML solutions, \cite{Paldino2021DoesForecasting} highlights that these approaches are not yet mature enough to address forecasting tasks effectively. The author's results concluded that AutoML frameworks (AutoGluon, H2O, TPOT, and Auto-sklearn) did not significantly outperform simple and conventional forecasting strategies (naive and exponential smoothing) across several forecasting challenges. \cite{Nikitin2022AutomatedPipelines} also agrees that there is still room for improvement, including support for distributed processing, explainability, and more robust optimization algorithms.

During my master's degree, I worked with various time series forecasting techniques such as Fuzzy Time Series, Temporal Convolutional Networks (TCN), Long Short-Term Memory Networks (LSTM), and classical models. I also explored ways to enhance the accuracy of these models through Ensemble Learning and hyperparameter optimization. Therefore, this project proposes developing a specific AutoML approach for time series forecasting that contributes to solutions addressing the current gaps in the field.
</p></details>

## Research map

<img src="https://github.com/PatriciaLucas/AutoML/blob/main/Figures/diagram-20230421.png?raw=true" width="800"/> 

### [AutoML](https://github.com/PatriciaLucas/AutoML/blob/main/automl.md) :triangular_flag_on_post:
- Componentes
 	- [Preparação dos dados](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/preparacao_dados.md)
 	- [Engenharia de recursos](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/selecao_modelos.md)
	- [CASH](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/CASH.md)
	- [Avaliação de modelos](https://github.com/PatriciaLucas/AutoML/blob/main/componentes/avaliacao_modelos.md)
- [Frameworks](https://github.com/PatriciaLucas/AutoML/tree/main/frameworks)
- [Séries Temporais](https://github.com/PatriciaLucas/AutoML/blob/main/series_temporais/series_temporais.md)
	- [Datasets](https://github.com/PatriciaLucas/AutoML/blob/main/series_temporais/datasets)
	- [Frameworks](https://github.com/PatriciaLucas/AutoML/blob/main/series_temporais/frameworks)
		


