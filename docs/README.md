<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-123065321-1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-123065321-1');
</script>

## Intro
Welcome to my new blog! This is a place for me to share Data Science resources where I will try to provide lots of **simple python examples directly in Google Colab**. In particular, I will be focusing on the following libraries: sklearn, tensorflow, keras, matplotlib, nltk, scipy, numpy and pandas.  

**NEW:** Non-supervised text classification using TF-IDF method (scroll down to bottom of this page).  

Stay tuned as I continue to add more Machine Learning examples ~Gal, August 2018.  

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
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/nlp/nltk_preprocess.ipynb" target="_blank">Process text files</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/html_extracting_table_data.ipynb" target="_blank">Extracting table data from the web</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/list_comprehensions.ipynb" target="_blank">List comprehensions</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/misc/numpy_basics.ipynb" target="_blank">Numpy basics</a>  
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/sklearn/sklearn_precision_recall_vs_roc_curves.ipynb" target="_blank">Precision-Recall vs ROC curves</a>  

### Machine Learning - Example use cases
<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/sklearn/sklearn_logistic_regression_vs_gbm.ipynb" target="_blank">Breast Cancer Prediction</a>  - demonstrates the following techniques:
1. Univariate feature reduction (remove low correlations with the target).
2. Paired feature reduction (for each highly correlated pair, use only the feature that correlates better with the target).
3. Compare two different model types for supervised learning (Logistic Regression and GBM), including the testing and ranking of feature importance.
4. Calculate percentile bins for each model in order to analyze the ratio of positive classes for each percentile bin.
5. Plot ROC and Precision-Recall curves.  

<a href="https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/nlp/nltk_tf-idf_clustering.ipynb" target="_blank">Non-supervised text classification using TF-IDF method</a>  - demonstrates the following techniques:

1. Load and process text (for a simplified toy dataset)  
2. Vectorize text to a numeric matrix  
(i) Transform input sentences using count vectorizer  
(ii) Transform count vectorizer to a bag-of-words  
(iii) Transform bag-of-words to TF-IDF  
(iv) Build weighted word counts from TF-IDF  
(v) Build cosine similarity of sentences from TF-IDF  
(vi) Build a word cloud from the weighted word counts  
3. Dimensionality Reduction using PCA  
4. Calculate K-means clusters (unsupervised classification)  
