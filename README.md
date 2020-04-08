
# Confidence Intervals - Lab

## Introduction

As we begin to talk about confidence intervals, it’s important to remember the difference between a **sample distribution** and a **sampling distribution**. 

Remember that a sample data distribution is the distribution of the data points within a single sample.  A sampling distribution is the probability distribution a statistic can take. Also recall that, by the Central Limit Theorem, the sampling distribution of the sample mean (x_hat) is approximately normal regardless of the shape of the original distribution of the variable.

As seen in a previous lab,  point estimates can provide some idea of a population parameter like the mean, but estimates are prone to error and also, taking multiple samples to get improved estimates may not be feasible all the time. 

## Objectives
You will be able to:
* Calculate confidence intervals
* Interpret confidence intervals in relation to true population parameters

## Data 

If we wanted to know the average age of registered voters in some elections, we could take a survey of registered voters and then use the average age of the respondents as a point estimate of the average age of the population as a whole.

In a previous lab, we saw that the sample mean is usually not exactly the same as the population mean. This difference can be caused by many factors including poor survey design, biased sampling methods and the randomness inherent to drawing a sample from a population. Let's run a simulation to investigate point estimates by generating a population of random age data and then drawing a sample from it to estimate the mean:


```python
# Import necessary libraries
import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Plot styles
plt.style.use('fivethirtyeight')
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```

The Poisson distribution is the discrete probability distribution of the number of events occurring in a given time period, given the average number of times the event occurs over that time period. We shall use a Poisson distribution to construct a bimodal distribution.


```python
np.random.seed(15)
population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages = np.concatenate((population_ages1, population_ages2))
```


```python
population_ages
```




    array([60, 40, 49, ..., 24, 28, 31])



Calculate the mean for the population and visualize the distribution. Also, describe the solution using five point statistics. (Hint: convert the population_ages array into a dataframe).


```python

# Convert the population into a pandas dataframe pop_ages
pop_ages = pd.DataFrame(population_ages)

# Calculate and print descriptive statistics for pop_ages
print(pop_ages.describe())
# The mean of population is: 42.985332

# count	250000 
# mean	42.985332
# std	13.211628
# min	18.000000
# 25%	29.000000
# 50%	47.000000
# 75%	54.000000
# max	84.000000

```

                       0
    count  250000.000000
    mean       42.985332
    std        13.211628
    min        18.000000
    25%        29.000000
    50%        47.000000
    75%        54.000000
    max        84.000000



```python
# Draw a histogram for pop ages
pop_ages.hist(bins=40)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fddd907f080>]],
          dtype=object)




![png](index_files/index_8_1.png)


Let's take a random sample of size 500 from this distribution and calculate the sample mean and standard deviation. Also, work out the difference between the population and sample mean. 


```python
np.random.seed(15)

# Take random sample of size 500
sample_size = 500
sample = np.random.choice(pop_ages[0], size=sample_size)

# Calculate sample mean and standard deviation
sample_mean = np.mean(sample)
sample_std = np.std(sample)

print("Sample mean:", sample_mean)
print("Sample std.:", sample_std)
print("Difference between means:", population_ages.mean() - sample_mean)

# Sample mean: 43.49 
# Sample std.: 12.98529552994463
# Difference between means: -0.5046680000000023
```

    Sample mean: 43.49
    Sample std.: 12.98529552994463
    Difference between means: -0.5046680000000023


We can see there is a small difference between sample mean and population mean. An increase in sample size can help reduce this difference. 

The central limit theorem states the distribution of **many sample means**, known as a **sampling distribution**, will be normally distributed. This rule holds even if the underlying distribution itself is not normally distributed as we saw above. 

As a result, we can treat the sample mean as if it were drawn from a normal distribution. 

To illustrate, let's create a sampling distribution by taking 100 samples from our population and then making 100 point estimates of the mean. Calculate the mean and standard deviation of sample means.



```python
np.random.seed(15)

point_estimates = []         # Make empty list to hold point estimates

# Take 100 samples and generate 100 point estimates using a for loop. 
# append sample means to get point estimates
for i in range(100):
    sample = np.random.choice(pop_ages[0], size=500)
    point_estimates.append(sample.mean())

# Convert the point_estimates into a pandas dataframe
# Calculate and print descriptive statistics for the dataframe
sampling_dist = pd.DataFrame(point_estimates)

# count	100.000000
# mean	42.959380
# std	0.586404
# min	41.296000
# 25%	42.530000
# 50%	42.960000
# 75%	43.356000
# max	44.492000
sampling_dist.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>42.959380</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.586404</td>
    </tr>
    <tr>
      <td>min</td>
      <td>41.296000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>42.530000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>42.960000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>43.356000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>44.492000</td>
    </tr>
  </tbody>
</table>
</div>



Let's visualize the distribution of sample means to check for normality.


```python
#Visualize the point estimates by plotting a density plot (use pandas)
sampling_dist[0].plot.kde();
```


![png](index_files/index_14_0.png)


The sampling distribution appears to be roughly normal, despite the bimodal population distribution that the samples were drawn from. This is where the central limit theorem comes into play. In addition, the mean of the sampling distribution approaches the true population mean. The more samples we take, the better our estimate of the population parameter is likely to be. 

## What is a Confidence Interval?

A **Confidence Interval** is a range of values above and below the point estimate that captures the true population parameter at some predetermined confidence level. If we **want** to have a 95% chance of capturing the true population parameter with a point estimate and a corresponding confidence interval, we would set the confidence level to 95%. *Higher confidence levels result in wider confidence intervals.*

We calculate a confidence interval by taking a point estimate and then adding and subtracting a **margin of error** to create a range. The margin of error is based on your desired confidence level, the spread of the data and the size of your sample. The way you calculate the margin of error depends on whether you know the standard deviation of the population or not.

The margin of error for a known population standard deviation is:

> **Margin of Error = z ∗ σ / √n**

Where σ (sigma) is the population standard deviation, n is sample size, and z is a number known as the z-critical value. 

>The z-critical value is the number of standard deviations you'd have to go from the mean of the normal distribution to capture the proportion of the data associated with the desired confidence level. 

For instance, we know that roughly 95% of the data in a normal distribution lies within 2 standard deviations of the mean, so we could use 2 as the z-critical value for a 95% confidence interval as shown in this image:
![](images/Normal-Distribution.png)

**If we want our confidence level — i.e., how confident we are that the true value of the parameter lies within the confidence interval — to be:**
- **90%**: The z-score multiplier should be **z = 1.645**, because 90% of the area under the $Z \text{~} N(0, 1)$ normal distribution lies between -1.645 and 1.645.
- **95%**: The z-score multiplier should be **z = 1.96**, because 95% of the area under the $Z \text{~} N(0, 1)$ normal distribution lies between -1.96 and 1.96.
- **99%**: The z-score multiplier should be **z = 2.575**, because 99% of the area under the $Z \text{~} N(0, 1)$ normal distribution lies between -2.575 and 2.575.

It is more suitable to get z-critical values with `stats.norm.ppf()` as the results are more accurate.
> **`stats.norm.ppf(q, loc=0, scale=1)`**	is a percent point function (inverse of cdf — percentiles).

Create a function to input population and sample data to calculate the confidence intervals:


```python
def conf_interval(pop, sample):
    '''
    Function input: population , sample 
    Function output: z-critical, Margin of error, Confidence interval
    '''
    sample_size = 500
    n = len(sample)
    x_hat = sample.mean()

    # Calculate the z-critical value using stats.norm.ppf()
    # Note that we use stats.norm.ppf(q = 0.975) to get the desired z-critical value 
    # instead of q = 0.95 because the distribution has two tails.
    q = 0.975
    z = stats.norm.ppf(q, loc=0, scale=1)  #  z-critical value for 95% confidence

    #Calculate the population std from data
    pop_stdev = pop.std()

    # Calculate the margin of error using formula given above
    moe = z * ( pop_stdev / np.sqrt(n))

    # Calculate the confidence interval by applying margin of error to sample mean 
    # (mean - margin of error, mean+ margin of error)
    conf = x_hat - moe , x_hat + moe
    
    return z, moe, conf

# Call above function with sample and population 
z_critical, margin_of_error, confidence_interval = conf_interval(population_ages, sample)       

print("z-critical value:")              
print(z_critical)         
print ('\nMargin of error')
print(margin_of_error)
print("\nConfidence interval:")
print(confidence_interval)

# z-critical value:
# 1.959963984540054

# Margin of error
# 1.158026699800684

# Confidence interval:
# (41.86997330019931, 44.186026699800685)
```

    z-critical value:
    1.959963984540054
    
    Margin of error
    1.158026699800684
    
    Confidence interval:
    (41.86997330019931, 44.186026699800685)


> Note that with calculated confidence intervals, we captured the true population mean of 42.9

We can create several such confidence intervals and visualize them to get a better sense of what it means to "capture" the true mean. 

Let's set a sample size of 1000 and take 25 samples to calculate the confidence intervals using the function above. 


```python
np.random.seed(12)

# Select the sample size 
sample_size = 1000

# Initialize lists to store interval and mean values
intervals = []
sample_means = []

# Run a for loop for sampling 25 times and calculate + store confidence interval and sample mean values

for i in range(25):
    # Take a random sample of chosen size 
    sample = np.random.choice(population_ages, size=sample_size)
    
    # Calculate z_critical, margin_of_error, confidence_interval from function above
    z_critical, margin_of_error, confidence_interval = conf_interval(population_ages, sample)    


    
    # Calculate and append sample means and conf intervals for each iteration
    sample_mean = sample.mean()
    sample_means.append(sample_mean)
    intervals.append(confidence_interval)
print(sample_means)
print(intervals)
```

    [42.895, 43.032, 43.99, 42.789, 42.488, 43.568, 42.52, 42.809, 42.385, 43.28, 43.066, 42.961, 42.867, 42.997, 43.011, 43.231, 42.522, 43.457, 43.098, 43.082, 42.305, 42.646, 43.953, 42.805, 43.176]
    [(42.07615146777586, 43.71384853222415), (42.21315146777585, 43.85084853222414), (43.17115146777586, 44.80884853222415), (41.97015146777586, 43.607848532224146), (41.669151467775855, 43.306848532224144), (42.74915146777585, 44.38684853222414), (41.70115146777586, 43.33884853222415), (41.99015146777585, 43.62784853222414), (41.56615146777585, 43.20384853222414), (42.461151467775856, 44.098848532224146), (42.24715146777586, 43.88484853222415), (42.142151467775854, 43.77984853222414), (42.04815146777585, 43.68584853222414), (42.178151467775855, 43.815848532224145), (42.19215146777586, 43.82984853222415), (42.41215146777586, 44.049848532224146), (41.703151467775854, 43.34084853222414), (42.638151467775856, 44.275848532224146), (42.279151467775854, 43.916848532224144), (42.263151467775856, 43.900848532224146), (41.486151467775855, 43.123848532224144), (41.827151467775856, 43.464848532224146), (43.13415146777586, 44.77184853222415), (41.986151467775855, 43.623848532224144), (42.35715146777586, 43.99484853222415)]



```python
# plot the mean and confidence interval for each sample as error bars
# plot the population mean 

plt.figure(figsize=(15,9))
plt.errorbar(x=np.arange(0.1, 25, 1), y=sample_means, yerr=[ ( top - bot )/2.0 for bot, top in intervals], fmt='o')
plt.hlines(pop_ages[0].mean(), xmin=0, xmax=25, color='r');
```


![png](index_files/index_21_0.png)


Notice that in the plot above, most of the 95% confidence intervals overlap the red line marking the true mean. This is to be expected: since a 95% confidence interval captures the true mean 95% of the time, we'd expect our interval to miss the true mean 5% of the time.

## Interpreting the Confidence Interval

There are two interpretations we can derive from the above simulation, and one of them is wrong.


**Interpretation 1 (incorrect):**
> There is a 95% probability that the mean voter age is between 41.6 and 43.9.

**Interpretation 2 (correct):**
 >If we pulled 100 samples and constructed confidence intervals in the same manner, we would expect that 95 of the intervals would contain the true mean of population age. 

Why is interpretation 1 wrong? It assigns a probability to the true value. In Frequentist statistics, the true value is fixed and the data is random. Confidence intervals make a statement of probability about the confidence interval range that could contain the true value.

## Summary

In this lab, we saw an introduction to confidence intervals and how to construct them from random samples. We also saw how due to the central limit theorem, the mean of sample means tends to be a normal distribution even if the population is bimodal. We also saw how the z-critical value can be used to define the confidence interval based on a confidence level required and the size of samples. We learned how to correctly interpret a confidence interval. 
