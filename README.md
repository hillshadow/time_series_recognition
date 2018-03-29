# Time Series Recognition

This project repository is not enterely built ! 

## Introduction

The aim is to recognize the activity of a person by looking at the acceleration of his arms and legs.

For this purpose, one has to recognize time series in real time of the data acquisition.

The project is splitted in few steps :

1. Data Preparation
2. Series segmentation : automatic or manual
3. Template Construction 
4. Times Series Comparaison
5. Series Classification 
6. Times Series Recognition

## Data Preparation 

In order to exploit the acceleration measures we compute the Sum of the Square values of the acceleration series:
	
## Series segmentation 

The segmentation is needed to compute afterward the template of the classe. There is two approach to segment the SSQ series : 
manually or automatically.

This section is based on the works of Derquenne.

## Template Construction

The goal is to create average centroids that are consistent with the warping behavior of DTW.  The template construction  is based on the DBA method wich iteratively refines an average sequence $a$ and follows an expectation-maximization scheme:
1. Consider the average sequence $a$ fixed and find the best multiple alignment $M$ of the set of sequences $S$ with regard to $a$, by individually aligning each sequence of $S$ to $a$.
2. Now consider $M$ fixed and update $a$ as the best average sequence consistent with $M$.

This section is based on the works of Petitjean.

## Times Series Comparaison

To recognize and classify the time series, one has to be able to compare them. A time serie must be compared to the pattern of all the classes so as to detect which pattern it most looks like.
In other words : the comparaison algorithms compute the feature of a time series template after template and then joind them into a big features vector.

This section is based on the works of Strijov.

## Time Series Classification

After computing the parameters, one must train a classifier which will discriminate the time series.

## Time Series Recognition

Finally, one can carry out the recognition process on it's data.

