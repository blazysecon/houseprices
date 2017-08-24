HousePrices
==============================

Kaggle competition - predict sales prices of houses sold in Ames, Iowa

This project contains the sources, notebooks and makefiles I used to generate submissions for the HousePrices Kaggle competition (https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

Source code documentation under : https://fmassen.github.io/houseprices/

<p>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</p>

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make features` or `make predict`, …
    │
    ├── README.md          <- The top-level README.    
    │    
    ├── docs               <- Sphinx documentation.
    │
    ├── models             <- Predictions made using the model.
    │
    ├── notebooks          <- Jupyter notebooks for data and model exploration.
    │
    ├── references         <- Classification of input data.
    │
    ├── reports            <- /
    │   └── figures        <- Generated graphics and figures.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └─── src               <- Source code for use in this project.
       ├── __init__.py     <- Makes src a Python module.
       │    │   │
       ├── features        <- Scripts to turn raw data into features for modeling.
       │   └── build_features.py
       │
       ├── models          <- Scripts to train models and then use trained models to make 
       │   │                 predictions.
       │   ├── StackingTransformer.py 
       │   ├── train_model.py 	
       │   ├── evaluate_model.py
       │   └── predict_model.py
       │
       └── visualization   <- Scripts to create exploratory and results oriented visualizations.
           └── visualize.py
    


--------


