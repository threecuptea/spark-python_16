### spark-python_16 collect all python files I works on using apache spark python 1.6.  The majority is my re-work of labs from edx.org BerkeleyX course "CS120x: Distrubuted Machine Learning with Apache Spark".
#### The topics include:
    1. Train a linear regression model to predict the release year of a song given a set of audio features.
       a) Performs a grid search to find the best model by evaluating their RMSE.  
       b) Enhance by adding two-way interaction betwen features. 
       c) Automate the process using Pipeline with PolynomialExpansion as the first stage. 
 
    2. Train a logistic regression model for creating click-through rate (CTR) pipeline. Logistic regression model
       a) Featurize categorical data using OHE (one hot encoding)
       b) Enhance by reducing feature dimension via feature hashing
       c) Logistic regression models the probability of a click-through event
       d) Evaluate model performance using log loss which is calculated using actual label and the prediction probability 
    
    I use Spark python 1.6 DataFrame for the above 2 labs.

    3. Principal component analysis (PCA) using Eigendecomposition
       a) Generate data representaion (called top scores) of reduced dimension which account for significant portion 
          of variance of orginal data
       b) Expaln percentage of variance covered by top components 

    I use Spark Python 1.6 without DataFrame(purely RDD) and also numpy.linalg.eigh for Eigendecomposition.    

      
