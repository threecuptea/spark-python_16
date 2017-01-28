### spark-python_16 collects all python files that I worked on using apache spark python 1.6.  The majority is my re-work of labs from edx.org BerkeleyX course "CS120x: Distrubuted Machine Learning with Apache Spark". I tried spark python 2 and encountered some serious issues which I documented in lab2_linear_reg.py .
#### The topics include:
    1. Train a Spark-ML linear regression model to predict the release year of a song given a set of audio features.
       a) Performs a grid search to find the best model by evaluating their RMSEs.
       b) Enhance by adding two-way interaction between features.
       c) Automate the process using Pipeline with PolynomialExpansion as the first stage. 
 
    2. Train a Spark-ML logistic regression model for creating click-through rate (CTR) pipeline.
       a) Featurize categorical data using OHE (one hot encoding)
       b) Enhance by reducing feature dimension via feature hashing
       c) Logistic regression models the probability of a click-through event
       d) Evaluate model performance using log loss which is calculated using actual label and the prediction probability
    
    I use Spark python 1.6 DataFrame for the above 2 labs.

    3. Principal component analysis (PCA) using Eigendecomposition
       a) Generate data representaion (called top scores) of reduced dimension which accounts for significant portion
          of variance of orginal data
       b) Explain percentage of variance covered by top components

    I use Spark Python 1.6 without DataFrame(purely RDD) and also apply numpy.linalg.eigh for Eigendecomposition.

      
