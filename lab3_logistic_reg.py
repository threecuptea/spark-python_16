import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import SparseVector, VectorUDT, Vectors
from pyspark.ml.classification import LogisticRegression

from pyspark.sql.functions import explode, split, udf, when, log
import pyspark.sql.functions as sqlfunc
import pyspark.sql.types as sqltype
import numpy as np
from math import exp
from collections import defaultdict
import hashlib

from test_helper import Test

#############################################
# This is my re-write based upon
# This lab3 of the curriculum  of edx.org BerkeleyX -
#      CS120x: Distrubuted Machine Learning with Apache Spark
#
# This lab covers the steps for creating a click-through rate (CTR) prediction pipeline.
# The data comes from http://labs.criteo.com/, dataset which was used for 2014 Kraggle competition.
# https://www.kaggle.com/c/criteo-display-ad-challenge
#
# Originally the lab was done in python notebook hosted in community.cloud.databricks.com. See the link for original assignment
# https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs120_lab3_ctr_df.dbc.
# You can get a free account in community.databrick.com then import using the above link to see the assignment
#
# I wrote the lab using Spark 1.6 DataFrame on Jan. 2017
#############################################

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >> sys.stderr,  "Usage: lab3_logistic_reg [dac_example_file]"
        sys.exit(-1)

    sc = SparkContext(appName='lab3_logistic_reg')
    sqlContext = SQLContext(sc)
    sqlContext.setConf('spark.sql.shuffle.partitions', '6')
    #############################################
    #  PART 1: Featurize categorical data using one hot encoding
    #############################################
    #############################################
    #         (1a) One-hot encoding
    #############################################
    # animal, color, food
    sample_one = [(0, 'mouse'), (1, 'black')]
    sample_two = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
    sample_three = [(0, 'bear'), (1, 'black'), (2, 'salmon')]

    sample_data_df = sqlContext.createDataFrame([(sample_one,), (sample_two,), (sample_three,)], ['features'])

    sample_data_df.show(truncate=False)

    # The dict is in the order of featureID plus category (alpha ascending)
    sample_ohe_dict_manual = {}
    sample_ohe_dict_manual[(0, 'bear')] = 0
    sample_ohe_dict_manual[(0, 'cat')] = 1
    sample_ohe_dict_manual[(0, 'mouse')] = 2
    sample_ohe_dict_manual[(1, 'black')] = 3
    sample_ohe_dict_manual[(1, 'tabby')] = 4
    sample_ohe_dict_manual[(2, 'mouse')] = 5
    sample_ohe_dict_manual[(2, 'salmon')] = 6

    #############################################
    #         (1c) OHE features in SparseVector
    #############################################

    #SparseVector has two forms,SparseVector(7, [2,3], [1,1.]) or SparseVector(7, [(2,1.),(3,1.)])
    #7 is numbers of element, [2,3] is the indexes with values, [1.,1.] is what values for those indices,
    # the second is the form of collection of (index, value) pair.  The second form does not requires indices sorted
    print "(1c) OHE features in SparseVector"
    sample_one_ohe_feat_manual = SparseVector(7, [2, 3], [1.,1.])
    sample_two_ohe_feat_manual = SparseVector(7, [1, 4, 5], [1., 1.,1.])
    sample_three_ohe_feat_manual = SparseVector(7, [0, 3, 6], [1., 1., 1.])

    print sample_one_ohe_feat_manual.indices, sample_one_ohe_feat_manual.values

    Test.assertTrue(isinstance(sample_one_ohe_feat_manual, SparseVector), 'sample_one_feat_manual should be SparseVector')
    Test.assertTrue(isinstance(sample_two_ohe_feat_manual, SparseVector), 'sample_two_feat_manual should be SparseVector')
    Test.assertTrue(isinstance(sample_three_ohe_feat_manual, SparseVector),
                    'sample_three_feat_manual should be SparseVector')
    indices = sample_one_ohe_feat_manual.indices
    Test.assertTrue(
        len(indices) == 2 and len(set([2, 3]).intersection(indices)) == 2, 'incorrect sample_one_feat_manual indices')
    indices = sample_two_ohe_feat_manual.indices
    Test.assertTrue(
        len(indices) == 3 and len(set([1, 4, 5]).intersection(indices)) == 3, 'incorrect sample_two_feat_manual indices')
    indices = sample_three_ohe_feat_manual.indices
    Test.assertTrue(
        len(indices) == 3 and len(set([0, 3, 6]).intersection(indices)) == 3,
        'incorrect sample_three_feat_manual indices')

    #############################################
    #         (1d) Define OHE functions
    #############################################
    def one_hot_encoding(raw_feats, ohe_dict_broadcast):
        """Produce a one-hot-encoding from a list of features and an OHE dictionary.
        Note:
            You should ensure that the indices used to create a SparseVector are sorted.
            Enhanced and exclude those feat not in the ohe_dict_broadcast
        Args:
            raw_feats (list of (int, str)): The features corresponding to a single observation.  Each
                feature consists of a tuple of featureID and the feature's value. (e.g. sample_one)
            ohe_dict_broadcast (Broadcast of dict): Broadcast variable containing a dict that maps
                (featureID, category) to unique integer.
        Returns:
            SparseVector: A SparseVector of length num_ohe_feats with indices equal to the unique
                identifiers for the (featureID, value) combinations that occur in the observation and
                with values equal to 1.0.
        """
        ohe_dict = ohe_dict_broadcast.value
        turples = [ (ohe_dict[feat], 1.0) for feat in raw_feats if feat in ohe_dict ]
        return SparseVector(len(ohe_dict), turples)

    print ""
    print "(1d) Define OHE functions"
    sample_ohe_dict_manual_broadcast = sc.broadcast(sample_ohe_dict_manual)
    sample_one_ohe_feat = one_hot_encoding(sample_one, sample_ohe_dict_manual_broadcast)
    sample_two_ohe_feat = one_hot_encoding(sample_two, sample_ohe_dict_manual_broadcast)
    sample_three_ohe_feat = one_hot_encoding(sample_three, sample_ohe_dict_manual_broadcast)
    print sample_one_ohe_feat, sample_two_ohe_feat, sample_three_ohe_feat

    Test.assertEquals(sample_one_ohe_feat, sample_one_ohe_feat_manual,
                      "sample_one_ohe_feat should be equals to sample_one_ohe_feat_manual")
    Test.assertEquals(sample_two_ohe_feat, sample_two_ohe_feat_manual,
                      "sample_two_ohe_feat should be equals to sample_two_ohe_feat_manual")
    Test.assertEquals(sample_three_ohe_feat, sample_three_ohe_feat_manual,
                      "sample_three_ohe_feat should be equals to sample_three_ohe_feat_manual")


    def ohe_udf_generator(ohe_dict_broadcast):
        """Generate a UDF that is setup to one-hot-encode rows with the given dictionary.
        Note:
            We'll reuse this function to generate a UDF that can one-hot-encode rows based on a
            one-hot-encoding dictionary built from the training data.
        Args:
            ohe_dict_broadcast (Broadcast of dict): Broadcast variable containing a dict that maps
                (featureID, value) to unique integer.

        Returns:
            UserDefinedFunction: A UDF can be used in `DataFrame` `select` statement to call a
                function on each row in a given column.  This UDF should call the one_hot_encoding
                function with the appropriate parameters.
        """
        return udf(lambda raw_feats: one_hot_encoding(raw_feats, ohe_dict_broadcast), VectorUDT())

    sample_ohe_dict_udf = ohe_udf_generator(sample_ohe_dict_manual_broadcast)

    sample_ohe_df = sample_data_df.select(sample_ohe_dict_udf(sample_data_df.features))
    sample_ohe_df.show(truncate=False)

    sample_ohe_data_values = sample_ohe_df.collect()
    Test.assertTrue(len(sample_ohe_data_values) == 3, "Incorrect number of elements in sample_ohe_df")
    Test.assertEquals(sample_ohe_data_values[0], (SparseVector(7, {2: 1.0, 3: 1.0}),), "Incorrect element one value")
    Test.assertEquals(sample_ohe_data_values[1], (SparseVector(7, {1: 1.0, 4: 1.0, 5:1.0}),),
                      "Incorrect element two value")
    Test.assertEquals(sample_ohe_data_values[2], (SparseVector(7, {0: 1.0, 3: 1.0, 6: 1.0}),),
                      "Incorrect element three value")
    Test.assertTrue('one_hot_encoding' in sample_ohe_dict_udf.func.func_code.co_names,
                    'ohe_udf_generator should call one_hot_encoding')

    #############################################
    #  PART 2: Construct an OHE dictionary
    #############################################
    #############################################
    #         (2a) DataFrame with row of (featureID, category)
    #############################################

    print ""
    print "(2a) DataFrame with row of (featureID, category)"
    sample_distinct_feats_df = sample_data_df.select(explode(sample_data_df.features)).distinct()
    sample_distinct_feats_df.show(truncate=False)
    #Each element is a array,

    Test.assertEquals(sorted(map(lambda r: r[0], sample_distinct_feats_df.collect())),
                      [(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'), (1, 'tabby'), (2, 'mouse'), (2, 'salmon')],
                      'incorrect value for sample_distinct_feats_df')

    #############################################
    #         (2b) OHE dictionary from distinct features
    #############################################
    print ""
    print "(2b) OHE dictionary from distinct features"
    # Each element is a array, have to map to r[0] first
    sample_ohe_dict = sample_distinct_feats_df.rdd.map(lambda r: r[0]).zipWithIndex().collectAsMap()
    Test.assertEquals(sorted(sample_ohe_dict.keys()),
                      [(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'), (1, 'tabby'), (2, 'mouse'), (2, 'salmon')],
                      'incorrect sample_ohe_dict keys')
    Test.assertEquals(sorted(sample_ohe_dict.values()), range(7), 'incorrect sample_ohe_dict values')

    #############################################
    #         (2c) Automated creation of OHE dictionary
    #############################################
    print ""
    print "(2c) Automated creation of OHE dictionary"

    def create_one_hot_dict(input_df, col_name='features'):
        """Creates a one-hot-encoder dictionary based on the input data.
        Args:
            input_df (DataFrame with 'features' column): A DataFrame where each row contains a list of
                (featureID, value) tuples.
        Returns:
            dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
                unique integers.
        """
        return input_df.select(explode(input_df[col_name])).distinct().rdd.map(lambda row: row[0]).zipWithIndex().collectAsMap()

    sample_ohe_dict_auto = create_one_hot_dict(sample_data_df)

    Test.assertEquals(sorted(sample_ohe_dict_auto.keys()),
                      [(0, 'bear'), (0, 'cat'), (0, 'mouse'), (1, 'black'), (1, 'tabby'), (2, 'mouse'), (2, 'salmon')],
                      'incorrect sample_ohe_dict keys')
    Test.assertEquals(sorted(sample_ohe_dict_auto.values()), range(7), 'incorrect sample_ohe_dict values')

    #############################################
    #  PART 3: Parse CTR data and generate OHE features
    #############################################
    #############################################
    #         (3a) Loading and splitting the data, Move it to after parse
    #############################################

    print ""
    print "(3a) Loading and splitting the data"
    raw_df = sqlContext.read.text(sys.argv[1]).withColumnRenamed('value', 'text').cache()

    #############################################
    #         (3b) Extract features
    #############################################

    def parse_point(point):
        """Converts a \t separated string into a list of (featureID, value) tuples.
        Note:
            featureIDs should start at 0 and increase to the number of features - 1.
        Args:
            point (str): A comma separated string where the first value is the label and the rest
                are features.
        Returns:
            list: A list of (featureID, value) tuples.
        """
        lst = point.split('\t')[1:]
        return zip(range(len(lst)), lst)

    print ""
    print "(3b) extract feature"
    first_parsed_point = parse_point(raw_df.first()[0]) #text is the first field anyway
    print first_parsed_point

    Test.assertTrue(len(first_parsed_point) == 39, 'Incorrect parsed arry length')
    Test.assertEquals(first_parsed_point[:5], [(0, u'1'), (1, u'1'), (2, u'5'), (3, u'0'), (4, u'1382')], 'Incorrect parsed values')

    #############################################
    #         (3c) Extracting features continued
    #############################################

    parse_point_udf = udf(parse_point, sqltype.ArrayType(sqltype.StructType([
        sqltype.StructField('_1', sqltype.ShortType()),
        sqltype.StructField('_2', sqltype.StringType())
    ])))

    def parse_raw_df(raw_df):
        """Convert a DataFrame consisting of rows of \t separated text into labels and feature.
        Args:
            raw_df (DataFrame with a 'text' column): DataFrame containing the raw comma separated data.
        Returns:
            DataFrame: A DataFrame with 'label' and 'feature' columns.
        """
        return raw_df.select(split(raw_df.text, '\t').getItem(0).cast('double'), parse_point_udf(raw_df.text)).toDF('label', 'feature')

    print ""
    print "(3c) Extracting features continued"

    parsed_df = parse_raw_df(raw_df)
    # The same split as before parsed, move here, data content does not affect randomSplit().  It does not look like the content of data
    # will affect the splits However, Spark 2 will have different splits
    weights = [.8, .1, .1]
    seed = 42
    parsed_train_df, parsed_val_df, parsed_test_df = parsed_df.randomSplit(weights, seed)
    parsed_train_df.cache()
    parsed_val_df.cache()
    parsed_test_df.cache()
    n_train = parsed_train_df.count()
    n_val = parsed_val_df.count()
    n_test = parsed_test_df.count()

    print n_train, n_val, n_test, n_train + n_val + n_test

    Test.assertTrue(all([parsed_train_df.is_cached, parsed_val_df.is_cached, parsed_test_df.is_cached]),
                    'you must cache the split data')
    Test.assertEquals(n_train, 80178, 'incorrect value for n_train')
    Test.assertEquals(n_val, 9887, 'incorrect value for n_val')
    Test.assertEquals(n_test, 9935, 'incorrect value for n_test')

    # If I don't assign an aias to a field after explode, the field name is
    #print parsed_train_df.select(sqlfunc.explode('feature')).first()

    #num_categories should use count(), otherwise, it will calculate count() * featureID.  That's why the answer is always the multiplier of featureID
    #Use originally it uses sum does not make sense.
    #Get all unique combination of (featureID, cataegory)
    num_categories = (
        parsed_train_df.select(explode('feature').alias('feature')).distinct()
                               .select(sqlfunc.col('feature').getField('_1').alias('featureID'))
                                       .groupBy('featureID').count().orderBy('featureID').collect()
    )

    # I can use it for cross validation again the dict count
    total_category =  reduce(lambda x,y: x+y, np.array(num_categories)[:,1])
    print '\ntotal # categories= {0}'.format(total_category)

    Test.assertTrue(parsed_train_df.is_cached, 'parse_raw_df should return a cached DataFrame')
    Test.assertEquals(num_categories[2][1], 853, 'incorrect implementation of parse_point or parse_raw_df')
    Test.assertEquals(num_categories[32][1], 4, 'incorrect implementation of parse_point or parse_raw_df')
    Test.assertEquals(total_category, 233941, 'total category has wrong value')

    #############################################
    #         (3d) Create OHE dictionary from dataset
    #############################################
    ctr_ohe_dict = create_one_hot_dict(parsed_train_df, 'feature')
    num_ctr_ohe_feats = len(ctr_ohe_dict)

    print ""
    print "(3d) Create OHE dictionary from dataset"
    print num_ctr_ohe_feats
    print ctr_ohe_dict[(0, '')]

    Test.assertEquals(total_category, num_ctr_ohe_feats, 'The length of ohe dictionary should be equal to total categories from group by')
    Test.assertTrue((0, '') in ctr_ohe_dict, 'incorrect features in ctr_ohe_dict')

    #############################################
    #         (3e) Apply OHE to the dataset
    #############################################
    print ""
    print "(3e) Apply OHE to the dataset"

    ohe_dict_broadcast = sc.broadcast(ctr_ohe_dict)
    ohe_dict_udf = ohe_udf_generator(ohe_dict_broadcast)
    #  Make sure that ohe_train_df contains a label and features column and is cached
    ohe_train_df = parsed_train_df.select('label', ohe_dict_udf(parsed_train_df.feature).alias('features')).cache()

    print ohe_train_df.count()
    print ohe_train_df.take(1)

    Test.assertTrue('label' in ohe_train_df.columns and 'features' in ohe_train_df.columns,
                    "ohe_train_df should have 'label' and 'features' ")
    Test.assertTrue(ohe_train_df.is_cached, 'ohe_train_df should be cache') #isCached is a flag field not a function
    num_feat_parsed = len(parsed_train_df.first()[1])
    print num_feat_parsed
    Test.assertTrue(num_feat_parsed == 39, 'parsed_train_df should have 39 features')
    #features is a SparseVector indices
    first_ohe_train_vector = ohe_train_df.first()[1]
    num_feat_ohe = len(first_ohe_train_vector.indices)
    Test.assertTrue(num_feat_ohe == num_feat_parsed, 'ohe_train_df should have the same # features as parsed_train_df')
    Test.assertTrue(first_ohe_train_vector.size == num_ctr_ohe_feats, 'The size of features SparseVector of OHE encoding should equals to the size of OHE dictionary ')

    ohe_val_df = parsed_val_df.select('label', ohe_dict_udf(parsed_val_df.feature).alias('features')).cache()
    print ohe_val_df.count()
    print ohe_val_df.take(1)


    #############################################
    #  PART 4: CTR prediction and logloss evaluation
    #############################################
    #############################################
    #         (4a) Logistic regression
    #############################################
    # A natural classifier to use in this setting is logistic regression since it models the probability of a click-through event
    # rather than returning a binary response, and when working with rare events, probabilistic predictions are useful.

    # elasticNetParam: 'the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty.
    # For alpha = 1, it is an L1 penalty.')
    standardization = False
    elastic_net_param = 0.0
    reg_param = .01
    max_iter = 20

    lr = LogisticRegression(standardization=standardization, elasticNetParam=elastic_net_param, regParam=reg_param, maxIter=max_iter)
    lr_model_basic = lr.fit(ohe_train_df)
    print ""
    print "(4a) Logistic regression"
    print 'intercept: {0}'.format(lr_model_basic.intercept)
    print 'The length of coefficients: {0}'.format(len(lr_model_basic.coefficients))
    Test.assertTrue(len(lr_model_basic.coefficients) == 233941, 'The model coefficients should equals to the size of features SparseVector')

    #############################################
    #         (4b) Log loss
    #############################################
    print ""
    print "(4b) Log loss"
    example_log_loss_df = sqlContext.createDataFrame([(.5, 1), (.5, 0), (.99, 1), (.99, 0), (.01, 1),
                                                        (.01, 0), (1., 1), (.0, 1), (1., 0)], ['p', 'label'])
    example_log_loss_df.show(truncate=False)

    epsilon = 1e-16
    def add_log_loss(df):
        """Computes and adds a 'log_loss' column to a DataFrame using 'p' and 'label' columns.
        Note:
            log(0) is undefined, so when p is 0 we add a small value (epsilon) to it and when
            p is 1 we subtract a small value (epsilon) from it.
        Args:
            df (DataFrame with 'p' and 'label' columns): A DataFrame with a probability column
                'p' and a 'label' column that corresponds to y in the log loss formula.
        Returns:
            DataFrame: A new DataFrame with an additional column called 'log_loss' where 'log_loss' column contains the loss value as explained above.
        """
        return df.withColumn('log_loss', when(df.label == 1, when(df.p == 0, -log(df.p + epsilon)).otherwise(-log(df.p)))
                      .otherwise(when(df.p == 1, -log(1 - df.p + epsilon)).otherwise(-log(1 - df.p))))


    add_log_loss(example_log_loss_df).show(truncate=False)

    log_loss_values = add_log_loss(example_log_loss_df).select('log_loss').rdd.map(lambda r: r[0]).collect()
    Test.assertTrue(np.allclose(log_loss_values[:-2],
                                [0.6931471805599451, 0.6931471805599451, 0.010050335853501338, 4.60517018598808,
                                 4.605170185988081, 0.010050335853501338, -0.0]), 'log loss is not correct')
    Test.assertTrue(not(any(map(lambda x: x is None, log_loss_values[-2:]))),
                    'log loss needs to bound p away from 0 and 1 by epsilon')

    #############################################
    #         (4c) Baseline log loss
    #############################################
    print ""
    print "(4c) Baseline log loss"
    # This SQL way to write it
    #class_one_frac_train = ohe_train_df.selectExpr('mean(label)').first()[0]
    class_one_frac_train = ohe_train_df.groupBy().mean('label').first()[0]
    print 'Training class one fraction = {0:.3f}'.format(class_one_frac_train)
    ohe_train_p_df = ohe_train_df.withColumn('p', sqlfunc.lit(class_one_frac_train))
    log_loss_tr_base = add_log_loss(ohe_train_p_df).groupBy().mean('log_loss').first()[0]
    print 'Baseline Train Logloss = {0:.3f}'.format(log_loss_tr_base)

    expected_frac = 0.226209184564
    expected_log_loss = 0.53465512785
    Test.assertTrue(np.allclose(class_one_frac_train, expected_frac),
        'incorrect value for class_one_frac_train, Got{0}, Expected{1}'.format(class_one_frac_train, expected_frac))
    Test.assertTrue(np.allclose(log_loss_tr_base, expected_log_loss),
        'incorrect value for log_loss_tr_base, Got{0}, Expected{1}'.format(log_loss_tr_base, expected_log_loss))

    #############################################
    #         (4d) Predicted Probability
    #############################################
    print ""
    print "(4d) Predicted Probability"

    def add_probability(df, model):
        """Adds a probability column ('p') to a DataFrame given a model"""
        intercept = model.intercept
        coefficients_broadcast = sc.broadcast(model.coefficients)

        def get_p(features):
            """Calculate the probability for an observation given a list of features.
            Note:
                We'll bound our raw prediction between 20 and -20 for numerical purposes.
            Args:
                features: the features
            Returns:
                float: A probability between 0 and 1.
            """
            raw_pred = features.dot(coefficients_broadcast.value) + intercept
            x = min(max(-20, raw_pred), 20)
            return (1 / (1 + exp(-x)))

        get_p_udf = udf(get_p, sqltype.DoubleType())
        return df.withColumn('p', get_p_udf('features'))

    add_probability_model_basic = lambda df: add_probability(df, lr_model_basic)
    training_predictions = add_probability_model_basic(ohe_train_df).cache()
    training_predictions.show(5)

    expected = 18168.793778
    got = training_predictions.groupBy().sum('p').first()[0]
    Test.assertTrue(np.allclose(got, expected),
                    'incorrect value for training_predictions. Got {0}, expected {1}'.format(got, expected))

    #############################################
    #         (4e) Evaluate the model
    #############################################
    print ""
    print "(4e) Evaluate the model"
    def evaluate_results(df, model, baseline=None):
        """Calculates the log loss for the data given the model.
        Note:
            If baseline has a value the probability should be set to baseline before
            the log loss is calculated.  Otherwise, use add_probability to add the
            appropriate probabilities to the DataFrame.
        Args:
            df (DataFrame with 'label' and 'features' columns): A DataFrame containing
                labels and features.
            model (LogisticRegressionModel): A trained logistic regression model. This
                can be None if baseline is set.
            baseline (float): A baseline probability to use for the log loss calculation.
        Returns:
            float: Log loss for the data.
        """
        df_with_probability =  add_probability(df, model) if model != None else df.withColumn('p', sqlfunc.lit(baseline))
        df_with_log_loss = add_log_loss(df_with_probability)
        log_loss = df_with_log_loss.groupBy().mean('log_loss').first()[0]
        return log_loss

    log_loss_train_model_basic = evaluate_results(ohe_train_df, lr_model_basic)
    print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'.format(log_loss_tr_base, log_loss_train_model_basic))

    expected_log_loss = 0.474685722542
    Test.assertTrue(np.allclose(log_loss_train_model_basic, expected_log_loss),
                    'incorrect value for log_loss_train_model_basic. Got {0}, expected {1}'.format(
                        log_loss_train_model_basic, expected_log_loss))
    expected_res = 0.6931471805600546
    res = evaluate_results(ohe_train_df, None, 0.5)
    Test.assertTrue(np.allclose(res, expected_res),
                    'evaluate_results needs to handle baseline models. Got {0}, expected {1}'.format(res, expected_res))

    #############################################
    #         (4f) Validation log loss
    #############################################
    print ""
    print "(4f) Validation log loss"

    log_loss_val_base = evaluate_results(ohe_val_df, None, class_one_frac_train)
    log_loss_val_l_r0 = evaluate_results(ohe_val_df, lr_model_basic)
    print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'.format(log_loss_val_base,
                                                                                               log_loss_val_l_r0))
    expected_val_base = 0.544167185783
    Test.assertTrue(np.allclose(log_loss_val_base, expected_val_base),
                    'incorrect value for log_loss_val_base. Got {0}, expected {1}'.format(log_loss_val_base,                                                                                        expected_val_base))
    expected_val_model_basic = 0.487967958915
    Test.assertTrue(np.allclose(log_loss_val_l_r0, expected_val_model_basic),
                    'incorrect value for log_loss_val_l_r0. Got {0}, expected {1}'.format(log_loss_val_l_r0,
                                                                                          expected_val_model_basic))

    #############################################
    #  PART 4: Reduce feature dimension via feature hashing
    #############################################
    #############################################
    #         (5a) Hash function
    #############################################

    def hash_function(raw_feats, num_buckets, print_mapping=False):
        """Calculate a feature dictionary for an observation's features based on hashing.
        Note:
            Use print_mapping=True for debug purposes and to better understand how the hashing works.
        Args:
            raw_feats (list of (int, str)): A list of features for an observation.  Represented as
                (featureID, value) tuples.
            num_buckets (int): Number of buckets to use as features.
            print_mapping (bool, optional): If true, the mappings of featureString to index will be
                printed.
        Returns:
            dict of int to float:  The keys will be integers which represent the buckets that the
                features have been hashed to.  The value for a given key will contain the count of the
                (featureID, value) tuples that have hashed to that key.
        """
        mappings =  { category+':'+str(ind): int(int(hashlib.md5(category+':'+str(ind)).hexdigest(), 16) % num_buckets) for ind, category in raw_feats }
        if print_mapping:
            print mappings

        def map_update(l, r):
            l[r] += 1.0
            return l

        sparse_features = reduce(map_update, mappings.values(), defaultdict(float))
        return dict(sparse_features)

    # Reminder of the sample values:
    # sample_one = [(0, 'mouse'), (1, 'black')]
    # sample_two = [(0, 'cat'), (1, 'tabby'), (2, 'mouse')]
    # sample_three =  [(0, 'bear'), (1, 'black'), (2, 'salmon')]
    # mapping example
    #     {'mouse:0': 3, 'black:1': 3}
    #     {'mouse:0': 99, 'black:1': 51}
    print ""
    print "(5a) Hash function"
    sample_one_four_buckets = hash_function(sample_one, 4, True)
    sample_two_four_buckets = hash_function(sample_two, 4, True)
    sample_three_four_buckets = hash_function(sample_three, 4, True)
    sample_one_hundred_buckets = hash_function(sample_one, 100, True)
    sample_two_hundred_buckets = hash_function(sample_two, 100, True)
    sample_three_hundred_buckets = hash_function(sample_three, 100, True)

    print '\n\t\t 4 Buckets \t\t\t 100 Buckets'
    print 'SampleOne:\t {0}\t\t\t {1}'.format(sample_one_four_buckets, sample_one_hundred_buckets)
    print 'SampleTwo:\t {0}\t\t\t {1}'.format(sample_two_four_buckets, sample_two_hundred_buckets)
    print 'SampleThree:\t {0}\t\t\t {1}'.format(sample_three_four_buckets, sample_three_hundred_buckets)

    Test.assertEquals(sample_one_four_buckets, {3: 2.0}, 'incorrect value for samp_one_four_buckets')
    Test.assertEquals(sample_three_hundred_buckets, {80: 1.0, 82: 1.0, 51: 1.0},
                      'incorrect value for samp_three_hundred_buckets')

    #############################################
    #         (5b) Creating hash features
    #############################################
    print ""
    print "(5b) Creating hash features"
    num_hash_buckets = 2 ** 15
    # UDF that returns a vector of hashed features given an Array of tuples
    tuples_to_hash_features_udf = udf(lambda x: Vectors.sparse(num_hash_buckets, hash_function(x, num_hash_buckets)), VectorUDT())

    def add_hashed_features(df):
        """Return a DataFrame with labels and hashed features.
        Note:
            Make sure to cache the DataFrame that you are returning.
        Args:
            df (DataFrame with 'tuples' column): A DataFrame containing the tuples to be hashed.
        Returns:
            DataFrame: A DataFrame with a 'label' column and a 'features' column that contains a
                SparseVector of hashed features.
        """
        return df.select(df.label, tuples_to_hash_features_udf(df.feature).alias('features'))

    hash_train_df = add_hashed_features(parsed_train_df)
    hash_val_df = add_hashed_features(parsed_val_df)
    hash_test_df = add_hashed_features(parsed_test_df)
    hash_train_df.show(5)

    # In this case it is sum of values
    first_hash_train_vector = hash_train_df.first()[1]
    num_feat_hash = sum(first_hash_train_vector.values)
    Test.assertTrue(num_feat_hash == 39, 'ohe_train_df should have the same # features as parsed_train_df')
    Test.assertTrue(first_hash_train_vector.size == num_hash_buckets,
                    'The size of features SparseVector of hash train DF should equals to the num_hash_buckets')

    #############################################
    #         (5c) Sparsity
    #############################################
    def vector_feature_sparsity(sparse_vector):
        """Calculates the sparsity of a SparseVector.
        Args:
            sparse_vector (SparseVector): The vector containing the features.
        Returns:
            float: The ratio of features found in the vector to the total number of features.
        """
        return len(sparse_vector.indices) / float(sparse_vector.size)

    a_sparse_vector = Vectors.sparse(5, {0: 1.0, 3: 1.0})
    a_sparse_vector_sparsity = vector_feature_sparsity(a_sparse_vector)
    print ""
    print "(5c) Sparsity"
    print 'Sparsity = {0:.2f}.'.format(a_sparse_vector_sparsity)

    Test.assertEquals(a_sparse_vector_sparsity, 0.4, 'Incorrect value of a_sparse_vector_sparsity')

    #############################################
    #         (5d) Sparsity continued
    #############################################
    feature_sparsity_udf = udf(vector_feature_sparsity, sqltype.DoubleType())

    def get_sparsity(df):
        """Calculates the average sparsity for the features in a DataFrame.
        Args:
            df (DataFrame with 'features' column): A DataFrame with sparse features.
        Returns:
            float: The average feature sparsity.
        """
        return df.select(feature_sparsity_udf(df.features).alias('sparsity')).groupBy().mean('sparsity').first()[0]

    average_sparsity_ohe = get_sparsity(ohe_train_df)
    average_sparsity_hash = get_sparsity(hash_train_df)
    print ""
    print "(5d) Sparsity continued"
    print 'Average OHE Sparsity: {0:7e}'.format(average_sparsity_ohe)
    print 'Average HASH Sparsity: {0:7e}'.format(average_sparsity_hash)

    Test.assertTrue(average_sparsity_hash > average_sparsity_ohe, 'average_sparsity_hash should be denser than average_sparsity_ohe')

    #############################################
    #         (5e) Logistic model with hashed features
    #############################################
    # It has different hyper-parameters, we should not compare using different ones
    standardization = False
    elastic_net_param = 0.7
    reg_param = .001
    max_iter = 20

    lr_hash = LogisticRegression(standardization=standardization, elasticNetParam=elastic_net_param, regParam=reg_param, maxIter=max_iter)
    lr_model_hashed = lr_hash.fit(hash_train_df)
    print ""
    print "(5e) Logistic model with hashed features"
    print 'intercept: {0}'.format(lr_model_hashed.intercept)
    print len(lr_model_hashed.coefficients)

    log_loss_train_model_hashed = evaluate_results(hash_train_df, lr_model_hashed)

    print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}\n\tHashed = {2:.3f}'
           .format(log_loss_tr_base, log_loss_train_model_basic, log_loss_train_model_hashed))

    expected = 0.46545524487
    Test.assertTrue(np.allclose(log_loss_train_model_hashed, expected),
                    'incorrect value for log_loss_train_model_hashed. Got {0}, expected {1}'.format(
                        log_loss_train_model_hashed, expected))

    #############################################
    #         (5f) Evaluate on the test set
    #############################################
    print ""
    print "(5f) Evaluate on the test set"
    log_loss_test = evaluate_results(hash_test_df, lr_model_hashed)
    class_one_frac_test = parsed_test_df.selectExpr('mean(label)').first()[0]
    log_loss_test_baseline = evaluate_results(hash_test_df, None, class_one_frac_test)

    print ('Hashed Features Test Log Loss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
        .format(log_loss_test_baseline, log_loss_test))

    expected_test_baseline = 0.530363901139
    Test.assertTrue(np.allclose(log_loss_test_baseline, expected_test_baseline),
                    'incorrect value for log_loss_test_baseline. Got {0}, expected {1}'.format(log_loss_test_baseline,
                                                                                               expected_test_baseline))
    expected_test = 0.458838771351
    Test.assertTrue(np.allclose(log_loss_test, expected_test),
                    'incorrect value for log_loss_test. Got {0}, expected {1}'.format(log_loss_test, expected_test))







































































