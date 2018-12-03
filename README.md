## PyiSA
### iSA(X) Aggregated Text Classification in Python


PyiSA is a Python package providing access to the iSAX algorithm for supervised, aggregated text classification 
developed by [VOICES from the Blogs](http://www.voices-int.com/). 

This package is for **academic use only**; commercial use of iSA/iSAX is protected by U.S. provisional patent 
application No. 62/215264. PyiSA is distributed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License](http://creativecommons.org/licenses/by-nc-nd/4.0/). 


What is iSA/iSAX?
----

The iSA(X) algorithm is described in the paper "[iSA: A fast, scalable and accurate algorithm for sentiment analysis 
of social media content](http://dx.doi.org/10.1016/j.ins.2016.05.052)", Information Sciences (2016).

In essence, iSA is an algorithm for providing _aggregated_ classification of text documents based on a supervised
(human coded) sample. Unlike general-purpose classifiers (e.g. naive Bayes, support vector machines, decision trees
or neural networks), iSA is designed to give an good estimate of the distribution of categories across a full
corpus of documents, _not_ to provide accurate per-document catgeory predictions. 

iSA is particularly effective when working with small training corpora (i.e. the number of human-coded documents is 
relatively low), and the iSAX variant adds a step which augments small documents (e.g. Tweets or other short texts),
compensating for their short length and improving prediction accuracy.


PyiSA and the iSAX R Package
-----

The original implementation of iSAX by the paper authors (Andrea Ceron, Luigi Curini and Stefano Iacus) is the [iSAX R
package](https://github.com/blogsvoices/iSAX). 

PyiSA is a Python implementation which replicates the core functionality of iSAX; it has been designed to mimic the 
arguments and keywords of the R version where this is possible and sensible. However, a number of changes have been
made to render the package more "pythonic"; most notably, the `prep_data()` function has been reduced in its scope,
allowing users to choose their own approach to building a Term-Document Matrix from the several excellent packages
available for this purpose in Python. The `test_isa.py` file gives one example of this using Scikit-Learn and NLTK to
effectively construct a matrix for the English language, but other languages will require different approaches.


Installing PyiSA
----

At present, PyiSA may be installed simply by dropping the `./pyisax/` folder and its contents in your project. Please
ensure that you install the following dependencies (using `pip` or another package manager):
    
    pandas
    numpy
    scipy
    quadprog

A future version of PyiSA will be installable automatically using the `pip` command.


Using PyiSA
----

Import PyiSA to your project using the following command:

```python
from pyisax import PyiSA
```

You can now directly access the `PyiSA.prep_data()` function. This function expects to be passed a term-document 
matrix (documents in rows, features/vocabulary in columns), and accepts either a Numpy array or a Scipy CSR sparse
matrix. It returns a list of string representations of each document which may be passed directly to the main iSA
object.

To use iSA, first create an instance of the algorithm object with the settings you wish to use:

```python
my_isa = PyiSA(boot_count=1000, predict=False, sequence_length=5, sparse=False,
               verbose=False, tolerance=0)
``` 

**Parameters**: (all are optional)
- `boot_count`: Controls the number of runs the algorithm will attempt - higher figures may
yield more accurate results at the expense of processing time. 
- `predict`: If True, will populate the `predict_cats` attribute with each stem's predicted category
after processing. These predictions are provided for informational purposes only.
- `sequence_length`: Controls the length of the sub-sequences which should be used to augment the feature space of the
data in the iSAX step. Set this to 0 to skip iSAX and perform "vanilla" iSA on un-augmented data.
- `sparse`: Experimental; if set to True, will use Pandas sparse arrays for some internal processing. May help if memory
constraints are very tight. Probably doesn't, though.
- `verbose`: Give text feedback on various elements of the iSA algorithm's progress through stdout.
- `tolerance`: Lower bound for the determinant of P'*P; the feature space matrix is considered uninvertible below this,
raising an error in the algorithm. A value of 0 means this will never raise an error; generally best left alone.


Once you have created a PyiSA object with your required data, use it to predict category distribution as follows:

```python
my_isa.fit(X_train, X_test, y_train)
```

`X_train` and `X_test` are lists of document strings received from the `prep_data` function, the former being the documents
for which categorisation data exists, the latter being the remainder of the corpus. `y_train` is a list of categories for
the `X_train` documents (i.e. target variables).

After fitting, the results of the algorithm can be accessed from the following attributes:

| Attribute | Description |
|:----------|:------------|
| `best` | Best estimation of incidence of all categories across the corpus |
| `best_table` | Details of best estimation, including standard error and P-value |
| `estimate` | First estimate of category incidence. If `boot_count` is 0, this will be the same as `best`. | 
| `estimate_table` | Detailed statistics for `estimate`. |
| `boot` | Results from each run of the equation (averaged to discover `best`). |
| `predict_cats` | If `predict` parameter is set, predicted category for each 'stem'. |
| `elapsed_time` | Time in seconds it took to fit the model to the data. |


Non-European Languages
----
iSAX has been used successfully with languages such as Chinese and Japanese; the core algorithm is perfectly suited to
these languages, but the process of creating a Term-Document Matrix is more challenging, requiring the use of custom
tokenisation software to extract stems from the corpus. 

For Japanese, either the `mecab-python3` package (requiring the
installation of external software - easy on Linux or macOS, very challenging on Windows) or the `janome` package (no
external dependencies) are recommended.


Questions?
----
Questions or bug reports specific to the Python package should be directed through Github.

If you have a general question about iSA, or  are interested in using iSA technology in an enterprise environment, 
please contact [iSA@voices-int.com](mailto:iSA@voices-int.com).