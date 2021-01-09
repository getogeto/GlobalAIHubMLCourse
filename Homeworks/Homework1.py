
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 00:51:05 2021

@author: GeTo
"""

# =============================================================================
# # ## Homework 1
#
#
# # 1) How would you define Machine Learning?
#  Machine Learning is a sustainable system that can predict benefits from massive
#  amount of data.
#
# # 2) What are the differences between Supervised and Unsupervised Learning?
#  Specify example 3 algorithms for each of these.
#  Mainly difference between Supervised and Unsupervised learning is that Supervised Learning
#  involves the mapping from the input to the essential output while Unsupervised Learning
#  discovers patterns in data instead of aiming to produce output.
#  Problems can be solved by classification and regression for Supervised Learning
#  in reply to clustering and K-means for Unsupervised learning.
#
# # 3) What are the test and validation set, and why would you want to use them?
#  Test set should include %20 of the original dataset that used to evaluate the final model performance.
#  Validation set is the sample of data used to provide an unbiased evaluation of a model
#  fit on the training dataset while tuning model hyperparameters.
#
# # 4) What are the main preprocessing steps? Explain them in detail.
#  Why we need to prepare our data?
#  We need to prepare data before analyzing it. This help us to understand the quality and
#  quantity of data. 
#  - First we need to Gather the data.
#  - Then we need to focus on Duplicate Values which are removed in most cases.
#  - After cleaning duplicates we should Balance our data with Undersampling that
#  takes samples of majority class and Oversampling that copies the minority class.
#  - Then we should detect Missing Values and simply eliminate them or fill them with mean or median. 
#  - So afterall we are ready for Outlier Detection which means samples that are exceptionally
#  far from the mainstream of data. We have several methods to detect them such as Standart Deviation,
#  Box Plots, Isolation Forests and Z-Score.  
#  - After handling outliers we should Scale the Features so we can normalize the range of independent
#  variables or features of data. I think this is one of the most critical step of Pre-Processing.
#  We can use Normalization that bounds values between two numbers like between [0,1] or [-1.1]
#  Standardization transforms data to have zero mean and a variance of 1.
#  - Bucketing is used to minimize the effects of small observation errors.
#  - Feature Extraction is simply a process that identifies important features or attributes of the data.
#  - Feature Encoding is basically performing transformations on the data such that it can be easily
#  accepted as input for machine learning algorithms while still retaining its original meaning.
#    - Nominal : Any one-to-one mapping can be done which retains the meaning.
#  For instance, a permutation of values like in One-Hot Encoding.
#    - Ordinal : An order-preserving change of values. The notion of small, medium and large can be
#  represented equally well with the help of a new function. For example, we can encode this S, M and L
#  sizes into {0, 1, 2} or maybe {1, 2, 3}.
#  - Train / Validation / Test Split: Machine Learning algorithms, or any algorithm for that matter,
#  has to be first trained on the data distribution available and then validated and tested, before it
#  can be deployed to deal with real-world data.
#
# # 5) How you can explore countionus and discrete variables?
#  Discrete Variable refers to the variable that assumes a finite number of isolated values.
#  Values are obtained by counting.
#  Continuous variable alludes to the a variable which assumes infinite number of different values.
#  Values are obtained by measuring.
#
# # 6) Analyse the plot given below. (What is the plot and variable type, check the distribution
#  and make comment about how you can preproccess it.) 
# 
# # from IPython.display import Image   #displaying the sixth questions plot.
# # img = "D:\Documents\Global AI Hub\Homeworks\download.png"
# # Image(filename=img)
# 
#  Variable type : Continous
#  Distribution  : Bimodal
#  Plot - Histogram
#  Preproccess
#  - Add Missing Value (Median)
#  - Outlier Detection (Standard Deviation)
#  - Feature Scalling (Normalization)
# =============================================================================




