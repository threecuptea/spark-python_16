import itertools
import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector

import pyspark.sql.functions as func

import numpy as np
import math

from test_helper import Test

#############################################
# This is my re-write based upon
# The lab2 of the curriculum  of edx.org BerkeleyX -
#      CS120x: Distrubuted Machine Learning with Apache Spark

# This lab covers a common supervised learning pipeline, using a subset of the data from
# https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD UCI Machine Learning Mini-song
# The goal is to train a linear regression model to predict the release year of a song given a set of audio features.
#
# Originally the lab was done in python notebook hosted in community.cloud.databricks.com. See the link for original assignment
# https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs120_lab2_linear_regression_df.dbc.
# You can get a free account in community.databrick.com then import using the above link to see the assignment
#
# I wrote the lab using Spark 1.6 DataFrame on Jan. 2017, I also try rewrite it in Spark 2 and encounter some serious issue.
# All were documented in codes
#############################################

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >> sys.stderr,  "Usage: lab2_linear_reg [million-song-file]"
        sys.exit(-1)

    sc = SparkContext(appName='lab2_learn_reg')
    sqlContext = SQLContext(sc)
    sqlContext.setConf('spark.sql.shuffle.partitions', '6')

    #############################################
    #  PART 1: Initial load and processing
    #############################################
    #############################################
    #         (1a) load and verify data
    #############################################
    raw_df = sqlContext.read.text(sys.argv[1])

    num_points = raw_df.count()
    print num_points

    sample_points = raw_df.take(5)
    print sample_points

    Test.assertEquals(num_points, 6724, 'incorrect value for num_points')
    Test.assertEquals(len(sample_points), 5)

    #############################################
    #         (1b) Using LabeledPoint
    #############################################
    ## In Spark 2, DataFrame has no 'map' method, Need to convert to rdd first then toDF
    parsed_points_df = raw_df.select(func.split(raw_df.value, ',').alias('values')).rdd.map(
        lambda row: LabeledPoint(float(row.values[0]), row.values[1:])).toDF()

    first_point_features = parsed_points_df.first().features
    first_point_label = parsed_points_df.first().label
    print first_point_features, first_point_label

    d = len(first_point_features)
    print d

    Test.assertTrue(isinstance(first_point_label, float), 'Label must be float')
    Test.assertEquals(first_point_label, float(2001.0), 'Label has incorrect value')
    Test.assertEquals(d, 12, 'Features has incorrect length')

    #############################################
    #         (1c) Find the range
    #############################################
    # Although func does not have min any more, add group functions and groupBy().min('label')
    min_max = parsed_points_df.groupBy().agg(func.min(parsed_points_df.label), func.max(parsed_points_df.label)).first()
    min_year, max_year = min_max[0], min_max[1]
    print min_year, max_year
    year_range = max_year - min_year

    Test.assertEquals(min_year, float(1922.0), 'Incorrect minimum year')
    Test.assertTrue(year_range == 89, 'Incorrect year range')

    #############################################
    #         (1d) Shift the label
    #############################################
    parsed_data_df = parsed_points_df.select('features', (parsed_points_df.label - min_year).alias('label'))
    print '\n{0}'.format(parsed_data_df.first())

    old_sample_features = parsed_points_df.first().features
    new_sample_features = parsed_data_df.first().features
    Test.assertTrue(np.allclose(old_sample_features, new_sample_features),
                    'new features do not match old features')
    sum_feat_two = parsed_data_df.rdd.map(lambda lp: lp.features[2]).sum()
    Test.assertTrue(np.allclose(sum_feat_two, 3158.96224351), 'parsed_data_df has unexpected values')
    min_year_new = parsed_data_df.groupBy().min('label').first()[0]
    max_year_new = parsed_data_df.groupBy().max('label').first()[0]
    Test.assertTrue(min_year_new == 0, 'incorrect min year in shifted data')
    Test.assertTrue(max_year_new == 89, 'incorrect max year in shifted data')

    #############################################
    #         (1e) Split data into traing, validation and test sets
    #############################################
    # spark 1.6 and spark 2.0 randomSplit works very differently, I will get different results if I use Spark 2
    weights = [.8, .1, .1]
    seed = 42
    parsed_train_data_df, parsed_val_data_df, parsed_test_data_df = parsed_data_df.randomSplit(weights, seed)
    parsed_train_data_df.cache()
    parsed_val_data_df.cache()
    parsed_test_data_df.cache()
    n_train = parsed_train_data_df.count()
    n_val = parsed_val_data_df.count()
    n_test = parsed_test_data_df.count()

    print n_train, n_val, n_test, n_train + n_val + n_test
    print parsed_data_df.count()

    Test.assertTrue(n_train + n_val + n_test == parsed_data_df.count(),
                    'Sum of splits does not match original data count')
    Test.assertEquals(len(parsed_train_data_df.first().features), 12,
                      'parsed_train_data_df has wrong number of features')
    Test.assertTrue(n_train + n_val + n_test == 6724, 'Sumup of splits should equals to the whole')
    # I cannot do the following tests in 2.0
    Test.assertEquals(n_train, 5382, 'unexpected value for nTrain')
    Test.assertEquals(n_val, 672, 'unexpected value for nVal')
    Test.assertEquals(n_test, 670, 'unexpected value for nTest')


    #############################################
    #  PART 2: Create Baseline
    #############################################
    #############################################
    #         (2a) Average label
    #############################################
    average_train_year = (parsed_train_data_df.groupBy().mean('label').first()[0])

    # average_train_year = parsed_train_data_df.groupBy().mean('label').first()[0]
    print average_train_year

    Test.assertTrue(np.allclose(average_train_year, 54.0403195838),
                    'incorrect value for average_train_year')

    preds_and_labels_train_df =  parsed_train_data_df.withColumn('prediction', func.lit(average_train_year))
    preds_and_labels_val_df = parsed_val_data_df.withColumn('prediction', func.lit(average_train_year))
    preds_and_labels_test_df = parsed_test_data_df.withColumn('prediction', func.lit(average_train_year))

    evaluator = RegressionEvaluator() #default predictionCol="prediction", labelCol="label", metricName="rmse"
    rmse_train_base = evaluator.evaluate(preds_and_labels_train_df)
    rmse_val_base = evaluator.evaluate(preds_and_labels_val_df)
    rmse_test_base = evaluator.evaluate(preds_and_labels_test_df)

    print 'Baseline Train RMSE = {0:.3f}'.format(rmse_train_base)
    print 'Baseline Validation RMSE = {0:.3f}'.format(rmse_val_base)
    print 'Baseline Test RMSE = {0:.3f}'.format(rmse_test_base)

    Test.assertTrue(np.allclose([rmse_train_base, rmse_val_base, rmse_test_base],
                                [21.4303303309, 20.9179691056, 21.828603786]), 'incorrect RMSE values')


    #############################################
    #  PART 3: Gradient-descent: evaluation linear regression model using gradient-descent
    #############################################
    #############################################
    #         (3a) Gradient Summand
    #############################################
    # You can view gradient-descent formula image herebhttps://fuyangliudk.files.wordpress.com/2015/12/gradient-descent-2.jpg?w=768

    def gradient_summand(weights, lp):
        """Calculates the gradient summand for a given weight and `LabeledPoint`.
        Note:
            `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
            within this function.  For example, they both implement the `dot` method.
        Args:
            weights (DenseVector): An array of model weights (betas).
            lp (LabeledPoint): The `LabeledPoint` for a single observation.
        Returns:
            DenseVector: An array of values the same length as `weights`.  The gradient summand.
        """
        return (weights.dot(lp.features) - lp.label) * lp.features


    # In Spark 2, If I use DenseVector, I will always get TypeError: Cannot treat type <class 'pyspark.mllib.linalg.DenseVector'> as a vector
    example_w = DenseVector([1, 1, 1])
    example_lp = LabeledPoint(2.0, [3, 1, 4])
    # gradient_summand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
    summand_one = gradient_summand(example_w, example_lp)
    print summand_one

    Test.assertTrue(np.allclose(summand_one, [18., 6., 24.]), 'incorrect value for summand_one')


    #############################################
    #         (3b) Use weights to make prediction
    #############################################
    def get_labeled_prediction(weights, observation):
        """Calculates predictions and returns a (prediction, label) tuple.
        Args:
            weights (np.ndarray): An array with one weight for each features in `trainData`.
            observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
                features for the data point.
        Returns:
            tuple: A (prediction, label) tuple. Convert the return type of the label and prediction to a float.
        """
        return (float(weights.dot(observation.features)), float(observation.label))


    # In Spark 2, If I use DenseVector, I will always get TypeError: Cannot treat type <class 'pyspark.mllib.linalg.DenseVector'> as a vector
    weights = DenseVector([1.0, 1.5])
    prediction_example = sc.parallelize([LabeledPoint(2., [1.0, .5]), LabeledPoint(1.5, [.5, .5])])
    preds_and_labels_example = prediction_example.map(lambda lp: get_labeled_prediction(weights, lp))
    print preds_and_labels_example.collect()

    Test.assertTrue(isinstance(preds_and_labels_example.first()[0], float), "prediction must be float type")
    Test.assertEquals(preds_and_labels_example.collect(), [(1.75, 2.0), (1.25, 1.5)],
                      'incorrect definition for getLabeledPredictions')


    #############################################
    #         (3c) Gradient descent
    #############################################

    def linreg_gradient_descent(train_data, num_iters):
        """Calculates the weights and error for a linear regression model trained with gradient descent.
        Args:
            train_data (RDD of LabeledPoint): The labeled data for use in training the model.
            num_iters (int): The number of iterations of gradient descent to perform.

        Returns:
            (np.ndarray, np.ndarray): A tuple of (weights, training errors).  Weights will be the
                final weights (one weight per feature) for the model, and training errors will contain
                an error (RMSE) for each iteration of the algorithm.
        """
        n = train_data.count()
        d = len(train_data.first().features)
        w = np.zeros(d)
        alpha = 1.0
        error_train = np.zeros(num_iters)

        for i in range(num_iters):
            pred_label_train = train_data.map(lambda lp: get_labeled_prediction(w, lp))
            pred_label_train_df = sqlContext.createDataFrame(pred_label_train, ['prediction', 'label'])
            error_train[i] = evaluator.evaluate(pred_label_train_df)

            gradient = train_data.map(lambda lp: gradient_summand(w, lp)).sum()
            alpha_i = alpha / (n * math.sqrt(i + 1))
            w -= alpha_i * gradient

        return w, error_train


    example_n = 10
    example_d = 3
    # take is an action which return list(Row), need to convert it to rdd before It can be mapped, parallelize convert a list into rdd
    example_data = sc.parallelize(parsed_train_data_df.take(example_n)).map(
        lambda lp: LabeledPoint(lp.label, lp.features[:example_d]))

    print example_data.take(2)
    example_num_iters = 5
    example_weights, example_error_train = linreg_gradient_descent(example_data, example_num_iters)

    print example_weights

    #############################################
    #         (3d) Train the model
    #############################################
    num_iters = 50
    weights_LR0, error_train_LR0 = linreg_gradient_descent(parsed_train_data_df.rdd, num_iters)
    preds_and_labels = parsed_val_data_df.rdd.map(lambda lp: get_labeled_prediction(weights_LR0, lp))
    preds_and_labels_df = sqlContext.createDataFrame(preds_and_labels, ['prediction', 'label'])
    rmse_val_LR0 = evaluator.evaluate(preds_and_labels_df)
    print 'Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}'.format(rmse_val_base,
                                                                           rmse_val_LR0)

    ############################################
    #  PART 4: Train using SparkML and perform grid search
    #############################################
    #############################################
    #         (4a) Linear Regression
    #############################################
    num_iters = 500  # iterations
    reg = 1e-1  # regParam
    alpha = .2  # elasticNetParam
    use_intercept = True  # intercept

    # LinearRegression.fit will get the following error in Spark 2, cannot get around yet.
    # pyspark.sql.utils.IllegalArgumentException: u'requirement failed: Column features must be of type org.apache.
    # spark.ml.linalg.VectorUDT@3bfc3ba7 but was actually org.apache.spark.mllib.linalg.VectorUDT@f71b0bce.'
    lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
    first_model = lin_reg.fit(parsed_train_data_df)

    # coeffsLR1 stores the model coefficients; interceptLR1 stores the model intercept
    coeffs_LR1 = first_model.coefficients
    intercept_LR1 = first_model.intercept
    print coeffs_LR1, intercept_LR1

    #############################################
    #         (4c) Evaluate RMSE
    #############################################
    val_pred_df = first_model.transform(parsed_val_data_df)
    rmse_val_LR1 = evaluator.evaluate(val_pred_df)

    print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}' +
           '\n\tLR1 = {2:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1)

    #############################################
    #         (4d) Grid Search
    #############################################
    best_RMSE = rmse_val_LR1
    best_reg_param = reg
    best_model = first_model

    num_iters = 500  # iterations
    alpha = .2  # elasticNetParam
    use_intercept = True  # intercept

    for reg in [1e-10, 1e-5, 1e-2, 1.0]:
        lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
        model = lin_reg.fit(parsed_train_data_df)
        val_pred_df = model.transform(parsed_val_data_df)
        rmse_val_grid = evaluator.evaluate(val_pred_df)
        print rmse_val_grid

        if rmse_val_grid < best_RMSE:
            best_RMSE = rmse_val_grid
            best_reg_param = reg
            best_model = model

    rmse_val_LR_grid = best_RMSE
    print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n' +
           '\tLRGrid = {3:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1, rmse_val_LR_grid)

    Test.assertTrue(np.allclose(15.3052663831, rmse_val_LR_grid), 'incorrect value for rmseValLRGrid')

    ############################################
    #  PART 4: Add interactions between features
    #############################################
    #############################################
    #         (5a) Add two-way interaction
    #############################################

    def two_way_interactions(lp):
        """Creates a new `LabeledPoint` that includes two-way interactions.
        Note:
            For features [x, y] the two-way interactions would be [x^2, x*y, y*x, y^2] and these
            would be appended to the original [x, y] feature list.
        Args:
            lp (LabeledPoint): The label and features for this observation.
        Returns:
            LabeledPoint: The new `LabeledPoint` should have the same label as `lp`.  Its features
                should include the features from `lp` followed by the two-way interaction features.
        """
        two_way =  [ x[0]*x[1] for x in itertools.product(lp.features, repeat=2) ]
        return LabeledPoint(lp.label, np.hstack((lp.features, two_way)))

    train_data_interact_df = parsed_train_data_df.rdd.map(lambda lp: two_way_interactions(lp)).toDF()
    val_data_interact_df = parsed_val_data_df.rdd.map(lambda lp: two_way_interactions(lp)).toDF()
    test_data_interact_df = parsed_test_data_df.rdd.map(lambda lp: two_way_interactions(lp)).toDF()

    two_points_example = two_way_interactions(LabeledPoint(0.0, [2.,3.]))
    Test.assertTrue(np.allclose(sorted(two_points_example.features), sorted([2.0,3.0,4.0,6.0,6.0,9.0])),
                    'incorrect features generatedBy two_way_interactions')
    three_points_example = two_way_interactions(LabeledPoint(1.0, [1.,2.,3.]))
    Test.assertTrue(np.allclose(sorted(three_points_example.features), sorted([1.0,2.0,3.0,1.0,2.0,3.0,2.0,4.0,6.0,3.0,6.0,9.0])),
                    'incorrect features generatedBy two_way_interactions')
    Test.assertTrue(np.allclose(sum(train_data_interact_df.first().features), 28.623429648737346),
                    'incorrect features in train_data_interact_df')
    Test.assertTrue(np.allclose(sum(val_data_interact_df.first().features), 23.582959172640948),
                    'incorrect features in val_data_interact_df')
    Test.assertTrue(np.allclose(sum(test_data_interact_df.first().features), 26.045820467171758),
                    'incorrect features in test_data_interact_df')

    #############################################
    #         (5b) Build interaction model
    #############################################
    num_iters = 500
    reg = 1e-10
    alpha = .2
    use_intercept = True

    lin_reg = LinearRegression(maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
    model_interact = lin_reg.fit(train_data_interact_df)
    preds_and_labels_interact_df = model_interact.transform(val_data_interact_df)
    rmse_val_interact = evaluator.evaluate(preds_and_labels_interact_df)

    print ('Validation RMSE:\n\tBaseline = {0:.3f}\n\tLR0 = {1:.3f}\n\tLR1 = {2:.3f}\n\tLRGrid = ' +
           '{3:.3f}\n\tLRInteract = {4:.3f}').format(rmse_val_base, rmse_val_LR0, rmse_val_LR1,
                                                     rmse_val_LR_grid, rmse_val_interact)

    Test.assertTrue(np.allclose(rmse_val_interact, 14.3495530997), 'incorrect value for rmse_val_interact')

    #############################################
    #         (5c) Evaluate interfaction model on test data
    #############################################

    preds_and_labels_test_df = model_interact.transform(test_data_interact_df)
    rmse_test_interact = evaluator.evaluate(preds_and_labels_test_df)

    print ('Test RMSE:\n\tBaseline = {0:.3f}\n\tLRInteract = {1:.3f}'
           .format(rmse_test_base, rmse_test_interact))

    Test.assertTrue(np.allclose(rmse_test_interact, 14.9990015721),
                    'incorrect value for rmse_test_interact')

    #############################################
    #         (5d) Use the pipeline to create the interaction model
    #############################################
    num_iters = 500
    reg = 1e-10
    alpha = .2
    use_intercept = True

    polynomial_expansion = PolynomialExpansion(degree=2, inputCol="features", outputCol="polyFeatures")
    linear_regression = LinearRegression(featuresCol="polyFeatures", maxIter=num_iters, regParam=reg, elasticNetParam=alpha, fitIntercept=use_intercept)
    pipeline = Pipeline(stages=[polynomial_expansion, linear_regression])
    pipeline_model = pipeline.fit(parsed_train_data_df)

    predictions_df = pipeline_model.transform(parsed_test_data_df)
    rmse_test_pipeline = evaluator.evaluate(predictions_df)
    print('RMSE for test data set using pipelines: {0:.3f}'.format(rmse_test_pipeline))

    Test.assertTrue(np.allclose(rmse_test_pipeline, 14.99415450247963),
                'incorrect value for rmse_test_pipeline')















