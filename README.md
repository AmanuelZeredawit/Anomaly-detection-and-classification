# Arcelo_mittal

### General description
 
This project is developed for [ArcelorMittal](https://en.wikipedia.org/wiki/ArcelorMittal), an international steel manufacturer. ArcelorMittal is the second-largest steel producer in the world, with an annual 
crude steel production of 88 million metric tonnes as of 2022.

In this project, an algorithm is developed for the client to determine whether 
there is a constriction or not on a steel coil based on B3, B4, and B5 measurements. 
Then a model is developed to predict constriction.

![B3, B4, B5 measurement plot ](images/constriction/283805.png)


### Algorithm for detection Constriction

The measurement taken at B3, B4, and B5 are not taken at the same point, hence
we don't have the same length in a set of (length, width), which makes it difficult
to compare the width of B3, B4, and B5. To tackle that we first grouped the dataset
to the int(length) and aggregated them by getting their average, which makes 
comparing B3, B4, and B5 measurements easier. Then we get a new measurement B3B4 
which is the mean of B3 and B4. And we take the absolute value of the difference
between B3B4 and B5 to determine the width difference. Finally, we count the
number of data points in which the difference is greater or equal to 5.
Our algorithm detects a coil as constriction if the count is greater than or 
equal to 6, not constriction if the count is 0, and 'not sure if the count is 
between 0 to 6.


```python

# length cast to int
# B3, B4, and B5 are grouped and merged to form a dataframe df.

df.B3B4_width = df['B3_width', 'B4_width'].mean()
df.constriction_width = df.B5_width - df.B3B4_width
count = len(df.constriction_width > 5)

if count > 6:
      constriction
elsif count == 0:
      not constriction
else:
      not sure


```



So for analysis and modeling, we only use coils that are determined either as constriction or not. 

Result of our algorithm for constriction :

| Is_constriction|number of coils |
| -------------  | -------------- |
| Yes            |  1725          |
| NO             |  18445         |
| Not-sure       |  4541          |
| Total          |  25111         |



### Modeling

The dataset we have for modeling is unbalanced (18845, 1725) so we downsample the majority class to develop our models. Different models have been tried and evaluated. The metrics that we use for evaluation are F-score and confusion matrix as the business need of our client is to minimize false negative(FN) and false positive(FP).
Minimizing FN means minimizing the loss caused by manufacturing coils with constriction. Minimizing FP keeps our client "ArcelorMittal" in business by keeping its reputation of manufacturing and supplying constriction-free steel coils. In this project, we use a random forest classifier to predict prediction as FP and FN is lower than in other models.

confusion matrix result on downsampled test data:

|                |  Positive      | Negative |
| -------------  | -------------- |----------|
| Positive       |  302           | 54       |
| Negative       |  51            | 283      |

### Usage

This project is a private property of ArcelorMittal. So using the files without the consent of
ArcelorMittal is not allowed.


### Installation

To deploy and use the project first clone it to your machine. And use the package manager
[pip](https://pip.pypa.io/en/stable/) to install the virtual environment and libraries.



1. Install virtualenv

```bash
pip install virtualenv
```
2. Create a virtual environment and activate it
```bash
virtualenv venv
> On windows -> venv\Scripts\activate
> On Linux -> . env/bin/activate

```
3. Install the necessary libraries
```bash
pip install -r requirements.txt
```


### Future development

* The algorithm is to be improved by applying some statistical concepts
* The dataset is also further to be preprocessed so that only features having importance used
* The models also have to be tuned further by tuning their parameters
* Different models also to be tried



### Collaborators

Developer Team
* Olga
* Shakil
* Amanuel
* Data Engineering team

Becode coaches
* Chrysanthi
* Louis
* Vanessa

ArcelorMittal
* Thomas




© 2022 Becode, Ghent.
















