Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Preparing the data
^^^^^^^^^^^^^^^^^^
* `make features` will clean and transform the indicated file ($IN) and output the cleaned data in CSV format to the indicated ($OUT) path. There are different modes ($MODE):
	- train : data will be cleaned, transformed and outliers will be removed
	- eval : data will be cleaned and transformed
	- test : data will be cleaned and transformed

Evaluating the model
^^^^^^^^^^^^^^^^^^^^
* `make evaluate` will take the input data ($IN), split it into a training and a test set. The model is fitted to the training set and used to predict the target of the test set. Since the data from a single input source is split into a training and a test set, this data should not have been treated with outlier removal.
Additionally one can write input data with the predictions and the score to a csv file ($OUT) (for visualizations for example). The path to the original untreated data has also to be provided ($ORIGINAL) to allow writing the original non-transformed features to the file. 


Using the model to make predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `make predict` will take the training data ($TRAIN) and the testing data ($TEST), fit the model to the training set and use it to predict the target of the test set. The predictions are written to a csv file ($OUT).


Visualizing the data
^^^^^^^^^^^^^^^^^^^^
* `make visualize` will take the input data ($IN) and plot the features against the target (the SalePrice) and print the plots to a PDF file ($OUT). There are two different modes:
	- features : all features are plotted in described fashion
	- score : all features are plotted in described fashion and the points are coloured in function of the additional data column ‘Score’
