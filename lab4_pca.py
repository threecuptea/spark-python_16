import sys
from pyspark import SparkContext
import numpy as np
from numpy.linalg import eigh

from test_helper import Test
#############################################
# This is my re-write based upon
# This lab4 of the curriculum  of edx.org BerkeleyX -
#      CS120x: Distrubuted Machine Learning with Apache Spark
# I skip Part3 up because it's specific to neuro science field
#
# This lab delves into exploratory analysis of data, specifically using principal component analysis (PCA)
# and feature-based aggregation.
#
# Originally the lab was done in python notebook hosted in community.cloud.databricks.com. See the link for original assignment
# https://raw.githubusercontent.com/spark-mooc/mooc-setup/master/cs120_lab4_pca.dbc
# You can get a free account in community.databrick.com then import using the above link to see the assignment
#
# I wrote this lab using using Spark 1.6 and numpy.linalg.eigh for Eigendecomposition that PCA depends upon it to
# generate principal component(s) on Jan. 16, 2017.  This lab does not use DataFrame
#############################################
if __name__ == '__main__':

    sc = SparkContext(appName='lab4_pca')

    #############################################
    #  PART 1: Work through the steps of PCA on a sample dataset
    #############################################
    #############################################
    #         (1a) intepreting PCA
    #############################################

    def create_2D_gaussian(mn, variance, cov, n):
        """Randomly sample points from a two-dimensional Gaussian distribution"""
        np.random.seed(142)
        return np.random.multivariate_normal(np.array([mn, mn]), np.array([[variance, cov], [cov, variance]]), n)

    # Data with no co-variance.  It's
    data_random = create_2D_gaussian(mn=50, variance=1, cov=0, n=100)
    data_correlated = create_2D_gaussian(mn=50, variance=1, cov=.9, n=100)

    correlated_data = sc.parallelize(data_correlated)
    mean_correlated = correlated_data.mean()
    correlated_data_zero_mean = correlated_data.map(lambda x: x - mean_correlated)
    print "(1a) intepreting PCA"
    print mean_correlated
    print correlated_data.take(1)
    print correlated_data_zero_mean.take(1)

    Test.assertTrue(np.allclose(mean_correlated, [49.95739037, 49.97180477]),
                    'incorrect value for mean_correlated')
    Test.assertTrue(np.allclose(correlated_data_zero_mean.take(1)[0], [-0.28561917, 0.10351492]),
                    'incorrect value for correlated_data_zero_mean')

    #############################################
    #         (1b) Sample covariance matrix
    #############################################
    print ""
    print "(1b) Sample covariance matrix"

    correlated_cov = correlated_data_zero_mean.map(lambda x: np.outer(x, x)).mean()
    print correlated_cov

    cov_result = [[0.99558386, 0.90148989], [0.90148989, 1.08607497]]
    Test.assertTrue(np.allclose(cov_result, correlated_cov), 'incorrect value for correlated_cov')

    #############################################
    #         (1c) Covariance function
    #############################################
    print ""
    print "(1c) Covariance function"

    def estimate_covariance(data):
        """Compute the covariance matrix for a given rdd.
        Note:
            The multi-dimensional covariance array should be calculated using outer products.  Don't
            forget to normalize the data by first subtracting the mean.

        Args:
            data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.

        Returns:
            np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
                length of the arrays in the input `RDD`.
        """
        data_mean = data.mean()
        data_zero_mean = data.map(lambda x: x - data_mean)
        return data_zero_mean.map(lambda x: np.outer(x, x)).mean()

    correlated_cov_auto = estimate_covariance(correlated_data)
    print correlated_cov_auto

    Test.assertTrue(np.allclose(correlated_cov, correlated_cov_auto),
                    'incorrect value for correlated_cov_auto')

    #############################################
    #         (1d) Eigendecomposition
    #############################################
    # eig_vals represents the variances that each dimension of data accounts for.
    # eigenvectors represents coefficients(weights) that each dimension of data can dot product with to create
    # new representation of data of that dimension.
    eig_vals_manual, eig_vecs_manual = eigh(correlated_cov_auto)
    print ""
    print "(1d) Eigendecomposition"
    print 'eigenvalues: {0}'.format(eig_vals_manual)
    print '\neigenvectors: \n{0}'.format(eig_vecs_manual)

    # np.argsort will return the indices of eig_vals in ascending order
    # https://docs.python.org/2.3/whatsnew/section-slices.html, param 1 start, param 2 end,
    # param 3 step, -1 in param 3 means in reverse order,
    # It does not care about the row, pick the column using the column return by the index
    inds_manual = np.argsort(eig_vals_manual)[::-1]
    top_component_manual = eig_vecs_manual[:, inds_manual[0]]

    def check_basis(vectors, correct):
        return np.allclose(vectors, correct) or np.allclose(np.negative(vectors), correct)

    Test.assertTrue(check_basis(top_component_manual, [0.68915649, 0.72461254]),
                    'incorrect value for top_component')

    correlated_data_scores_manual = correlated_data.map(lambda x: np.dot(x, top_component_manual))
    print 'one-dimensional data (first three):\n{0}'.format(correlated_data_scores_manual.take(3))

    first_three = [70.51682806, 69.30622356, 71.13588168]
    Test.assertTrue(check_basis(correlated_data_scores_manual.take(3), first_three),
                    'incorrect value for correlated_data_scores')


    ###########################################################
    #  PART 2: Write a PCA function and evaluate PCA on sample datasets
    ###########################################################
    #############################################
    #         (2a) PCA function
    #############################################
    print ""
    print "(2a) PCA function"
    def pca(data, identifier, k=2, printDebug=True):
        """Computes the top `k` principal components, corresponding scores, and all eigenvalues.
        Note:
            All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
            each eigenvectors as a column.  This function should also return eigenvectors as columns.
        Args:
            data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
            k (int): The number of principal components to return.
        Returns:
            tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
                scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
                rows equals the length of the arrays in the input `RDD` (d) and the number of columns equals
                `k` (top k).  The `RDD` of scores has the same number of rows as `data` and consists of arrays
                of length `k`.  Eigenvalues is an array of length d (the number of features).
        """
        cov = estimate_covariance(data)
        eig_vals, eig_vecs = eigh(cov)
        if printDebug:
            print "\nThe PCA for '{0}'-".format(identifier)
            print 'original eigh_vals: \n{0}'.format(eig_vals)
            print 'original eigh_vecs: \n{0}\n'.format(eig_vecs)
        inds_desc = np.argsort(eig_vals)[::-1]
        top_comps = eig_vecs[:, inds_desc[:k]]
        top_scores = data.map(lambda x: np.dot(x, top_comps))

        return top_comps, top_scores, eig_vals[inds_desc]

    top_comps_correlated, correlated_data_scores_auto, eig_vals_correlated = pca(correlated_data, 'Correlated_data, k = 2')
    print 'top_components_correlated: \n{0}'.format(top_comps_correlated)
    print ('\ncorrelated_data_scores_auto (first three): \n{0}'
           .format('\n'.join(map(str, correlated_data_scores_auto.take(3)))))
    print '\neigenvalues_correlated: \n{0}'.format(eig_vals_correlated)

    # Create a higher dimensional test set
    pca_test_data = sc.parallelize([np.arange(x, x + 4) for x in np.arange(0, 20, 4)])
    # co-variance matrix is (4 x 4 matrix) with each element is 32 top component [0.5, 0.5, 0.5, 0.5], top eig_val = 128
    # 32 * .5 * 4 = 64 = 128 * .5, proof A * v = lambda * v (A is square matrix is co-variance matrix here)
    components_test, test_scores, eigenvalues_test = pca(pca_test_data, 'PCA test data, k = 2', 3)

    print '\npca_test_data: \n{0}'.format(np.array(pca_test_data.collect()))
    print '\ncomponents_test: \n{0}'.format(components_test)
    print ('\ntest_scores (first three): \n{0}'
           .format('\n'.join(map(str, test_scores.take(3)))))
    print '\neigenvalues_test: \n{0}'.format(eigenvalues_test)

    Test.assertTrue(check_basis(top_comps_correlated.T,
                                [[0.68915649, 0.72461254], [-0.72461254, 0.68915649]]),
                    'incorrect value for top_components_correlated')
    first_three_correlated = [[70.51682806, 69.30622356, 71.13588168], [1.48305648, 1.5888655, 1.86710679]]
    Test.assertTrue(np.allclose(first_three_correlated,
                                np.vstack(np.abs(correlated_data_scores_auto.take(3))).T),
                    'incorrect value for first three correlated values')
    Test.assertTrue(np.allclose(eig_vals_correlated, [1.94345403, 0.13820481]),
                    'incorrect values for eigenvalues_correlated')
    top_components_correlated_k1, correlated_data_scores_k1, eigenvalues_correlated_k1 = pca(correlated_data, 'Correlated_data, k = 1', 1)
    Test.assertTrue(check_basis(top_components_correlated_k1.T, top_comps_correlated.T[0]),
                    'incorrect value for components when k=1')
    Test.assertTrue(np.allclose(np.vstack(np.abs(correlated_data_scores_auto.take(3))).T[0],
                                np.vstack(np.abs(correlated_data_scores_k1.take(3))).T),
                    'incorrect value for scores when k=1')
    Test.assertTrue(np.allclose(eigenvalues_correlated_k1, eig_vals_correlated),
                    'incorrect values for eigenvalues when k=1')
    Test.assertTrue(check_basis(components_test.T[0], [.5, .5, .5, .5]),
                    'incorrect value for components_test')
    Test.assertTrue(np.allclose(np.abs(test_scores.first()[0]), 3.),
                    'incorrect value for test_scores')
    Test.assertTrue(np.allclose(eigenvalues_test, [128, 0, 0, 0]), 'incorrect value for eigenvalues_test')

    #############################################
    #         (2b) PCA on data random
    #############################################
    print ""
    print "(2a) PCA on data random"
    random_data_rdd = sc.parallelize(data_random)
    top_components_random, random_data_scores_auto, eigenvalues_random = pca(random_data_rdd, 'Random_data, k = 2')

    print 'top_components_random: \n{0}'.format(top_components_random)
    print ('\nrandom_data_scores_auto (first three): \n{0}'
           .format('\n'.join(map(str, random_data_scores_auto.take(3)))))
    print '\neigenvalues_random: \n{0}'.format(eigenvalues_random)

    Test.assertTrue(check_basis(top_components_random.T,
                                [[-0.2522559, 0.96766056], [-0.96766056, -0.2522559]]),
                    'incorrect value for top_components_random')
    first_three_random = [[36.61068572, 35.97314295, 35.59836628],
                          [61.3489929, 62.08813671, 60.61390415]]
    Test.assertTrue(np.allclose(first_three_random, np.vstack(np.abs(random_data_scores_auto.take(3))).T),
                    'incorrect value for random_data_scores_auto')
    Test.assertTrue(np.allclose(eigenvalues_random, [1.4204546, 0.99521397]),
                    'incorrect value for eigenvalues_random')

    #############################################
    #         (2c) 3D to 2D
    #############################################

    print ""
    print "(2c) 3D to 2D"
    m = 100
    mu = np.array([50, 50, 50])
    # I have to change co-variance from 0.9, 0.7 to .8 to .6; otherwise, I kept getting warning that co variance matrix is
    # not positive definitive
    r1_2 = 0.8
    r1_3 = 0.6
    r2_3 = 0.1
    sigma1 = 5
    sigma2 = 20
    sigma3 = 20
    c = np.array([[sigma1 ** 2, r1_2 * sigma1 * sigma2, r1_3 * sigma1 * sigma3],
                  [r1_2 * sigma1 * sigma2, sigma2 ** 2, r2_3 * sigma2 * sigma3],
                  [r1_3 * sigma1 * sigma3, r2_3 * sigma2 * sigma3, sigma3 ** 2]])
    np.random.seed(142)
    data_threeD = np.random.multivariate_normal(mu, c, m)

    threeD_data = sc.parallelize(data_threeD)


    components_threeD, threeD_scores, eigenvalues_threeD = pca(threeD_data, '3D correlated data, k = 2')

    print 'components_threeD: \n{0}'.format(components_threeD)
    print ('\nthreeD_scores (first three): \n{0}'
           .format('\n'.join(map(str, threeD_scores.take(3)))))
    print '\neigenvalues_threeD: \n{0}'.format(eigenvalues_threeD)

    Test.assertEquals(components_threeD.shape, (3, 2), 'incorrect shape for components_threeD')

    #############################################
    #         (2d) Variance explained
    #############################################
    def variance_explained(data, k=1):
        """Calculate the fraction of variance explained by the top `k` eigenvectors.

        Args:
            data (RDD of np.ndarray): An RDD that contains NumPy arrays which store the
                features for an observation.
            k: The number of principal components to consider.

        Returns:
            float: A number between 0 and 1 representing the percentage of variance explained
                by the top `k` eigenvectors.
        """
        _, _, eigenvalues = pca(data, '', k, False)
        return (sum(eigenvalues[:k]) / float(sum(eigenvalues)))


    print ""
    print "(2d) Variance explained"
    variance_random_1 = variance_explained(random_data_rdd, 1)
    variance_correlated_1 = variance_explained(correlated_data, 1)
    variance_random_2 = variance_explained(random_data_rdd, 2)
    variance_correlated_2 = variance_explained(correlated_data, 2)
    variance_threeD_2 = variance_explained(threeD_data, 2)

    print ('Percentage of variance explained by the first component of random_data_rdd: {0:.1f}%'
           .format(variance_random_1 * 100))
    print ('Percentage of variance explained by both components of random_data_rdd: {0:.1f}%'
           .format(variance_random_2 * 100))
    print ('\nPercentage of variance explained by the first component of correlated_data: {0:.1f}%'.
           format(variance_correlated_1 * 100))
    print ('Percentage of variance explained by both components of correlated_data: {0:.1f}%'
           .format(variance_correlated_2 * 100))
    print ('\nPercentage of variance explained by the first two components of threeD_data: {0:.1f}%'
           .format(variance_threeD_2 * 100))

    Test.assertTrue(np.allclose(variance_random_1, 0.588017172066), 'incorrect value for variance_random_1')
    Test.assertTrue(np.allclose(variance_correlated_1, 0.933608329586),
                    'incorrect value for varianceCorrelated1')
    Test.assertTrue(np.allclose(variance_random_2, 1.0), 'incorrect value for variance_random_2')
    Test.assertTrue(np.allclose(variance_correlated_2, 1.0), 'incorrect value for variance_correlated_2')
    Test.assertTrue(np.allclose(variance_threeD_2, 0.997106764), 'incorrect value for variance_threeD_2')











