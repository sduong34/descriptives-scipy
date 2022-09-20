import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from pandas.plotting import scatter_matrix
from scipy import stats
from statsmodels.formula.api import ols

#import csv file
data = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
data

#create dataframe from array
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t}) 

#manipulating data
data.shape
data.columns 
print(data['Gender'])
data[data['Gender'] == 'Female']['VIQ'].mean()

#box plots for each column grouped by gender
groupby_gender = data.groupby('Gender')
groupby_gender.boxplot(column=['FSIQ', 'VIQ', 'PIQ'])

#scatter matrices
scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

#1-sample t-test
stats.ttest_1samp(data['VIQ'], 0)

#2-sample t-test
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)   

#paired tests
stats.ttest_ind(data['FSIQ'], data['PIQ']) 
stats.ttest_rel(data['FSIQ'], data['PIQ'])
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0) 
stats.wilcoxon(data['FSIQ'], data['PIQ'])

#simple linear regression
x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
data = pd.DataFrame({'x': x, 'y': y})
#specify OLS model and fit it
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
#inspect statistics derived from fit
print(model.summary())

#multiple regression
data = pd.read_csv('data/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())  
print(model.f_test([0, 1, -1, 0]))

#pairplot: scatter matrices ## missing dataset
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg')  
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg', hue='SEX')

#lmplot
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)  

#testing for interactions
result = sm.ols(formula='wage ~ education + gender + education * gender',
                data=data).fit()    
print(result.summary()) 