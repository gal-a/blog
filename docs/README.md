<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-123065321-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-123065321-1');
</script>

## Intro
Welcome to my new blog! This is a place for me to share Data Science resources where I aim to provide lots of **simple python examples directly in Google Colab**.

**NEW:** <a href="https://github.com/gal-a/blog/tree/master/docs/notebooks/nlp/nlp_tf-idf_clustering_post.md" target="_blank">Comparing the performance of non-supervised vs supervised learning methods for NLP text classification</a> 

If you have a cool data science project that you'd like to discuss (ideally, over a beer), feel free to reach out and connect via LinkedIn:**
<a href="https://il.linkedin.com/in/galarav" target="_blank">


## Notebooks

### Pandas I
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_basic_operations.ipynb" target="_blank">Basic pandas operations</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_build_dataframes.ipynb" target="_blank">Build dataframes</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_modify_series.ipynb" target="_blank">Modify series</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_merge_concat.ipynb" target="_blank">Merge and concat</a>  

### Pandas II
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_handle_missing_data.ipynb" target="_blank">Handle missing data</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_convert_types.ipynb" target="_blank">Convert types</a>   
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_groupby.ipynb" target="_blank">Group by</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_agg.ipynb" target="_blank">Aggregate</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/pandas/pandas_datetime.ipynb" target="_blank">DateTime</a>  

### Plotting Data
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_scatter_for_EDA.ipynb" target="_blank">Scatter plots for EDA</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_kmeans.ipynb" target="_blank">K-means clusters</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_seaborn_heatmap.ipynb" target="_blank">Seaborn heatmap</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_seaborn_distplots.ipynb" target="_blank">Seaborn distplots</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_boxplot.ipynb" target="_blank">Pandas boxplot</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_interpolate.ipynb" target="_blank">Scipy interpolate</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/plot/plot_quantile_quantile_plots.ipynb" target="_blank">Scipy Q-Q plots</a>  

### Miscellaneous
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/class_framework_in_notebooks_1_of_2.ipynb" target="_blank">Setup a class framework for use in notebooks, part 1 of 2</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/class_framework_in_notebooks_2_of_2.ipynb" target="_blank">Setup a class framework for use in notebooks, part 2 of 2</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/nlp/nltk_preprocess.ipynb" target="_blank">Process text files</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/html_extracting_table_data.ipynb" target="_blank">Extracting table data from the web</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/list_comprehensions.ipynb" target="_blank">List comprehensions</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/numpy_basics.ipynb" target="_blank">Numpy basics</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/sklearn/sklearn_precision_recall_vs_roc_curves.ipynb" target="_blank">Precision-Recall vs ROC curves</a>  

### Machine Learning - Example walkthroughs
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/sklearn/sklearn_logistic_regression_vs_gbm.ipynb" target="_blank">Breast cancer prediction</a>  -  python notebook that demonstrates the following techniques:
1. Univariate feature reduction (remove low correlations with the target).
2. Feature reduction based on collinearity (for each highly correlated pair of features, leave only the feature that correlates better with the target value).
3. Compare two different model types for supervised learning (Logistic Regression and GBM), including the testing and ranking of feature importance.
4. Calculate percentile bins for each model in order to determine the ratio of positive classes for each percentile bin.
5. Plot ROC and Precision-Recall curves. 

<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/nlp/nlp_tf-idf_clustering.ipynb" target="_blank">TF-IDF based topic clustering using PCA with K-means, NMF, LDA</a>  - python notebook that demonstrates the following techniques:

1. Vectorize text to a numeric matrix using TF-IDF (Term Frequency - Inverse Document Frequency)
2. Dimensionality Reduction using PCA
3. Unsupervised classification: Calculate K-means clusters based on PCA (a reduced version of TF-IDF)
4. Unsupervised classification: Calculate NMF (Non-negative Matrix Factorization) based on TF-IDF
5. Unsupervised classification: Calculate LDA (Latent Derilicht Analysis) based on TF

### Article for the above python notebook:
<a href="https://github.com/gal-a/blog/tree/master/docs/notebooks/nlp/nlp_tf-idf_clustering_post.md" target="_blank">Comparing the performance of non-supervised vs supervised learning methods for NLP text classification</a>

<a href="https://il.linkedin.com/in/galarav" target="_blank">
Copyright © Gal Arav, 2018