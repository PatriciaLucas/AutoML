import setuptools
from setuptools import find_packages

setuptools.setup(
    name='MARTS',
    install_requires=[
          'numpy',
          'pandas',
          'ray',
          'tigramite',
          'stationarizer',
          'emd',
          'scikit-learn',
          'lightgbm',
          'xgboost'
          ],
    packages=find_packages(),
    version='1.0',
    description='Auto-TSF',
    long_description='Automator Machine Learning Based on Decomposition, Causality and Evolutionary Multitask Optimization for Time Series Forecasting',
    long_description_content_type="text/markdown",
    author='Patrícia de Oliveira e Lucas',
    author_email='patelucas@gmail.com',
    url='',
    download_url='',
    keywords=['Time Series Forecasting', 'causal', 'Decomposition', 'Machine Learning', 'AuotML', 'Evolutionary Multitask Optimization', 'Causality'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
    
    ]
)
