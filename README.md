# Arya.ai Assignemnt June 2022

Hello there! This is my submission for the Arya.ai - Data Scientist Assignement Round.

# About Me

Hi! My name is Shikhar Shukla, a machine learning engineer and today I am trying to attempt to do an EDA on the dataset which is provided to me by Arya.ai as an assignment.

# The Assignment

The problem statement is as follows:

> Binary Classification - Do an exploratory analysis of the dataset provided, and decide on feature selection, and preprocessing before training a model to classify as class ‘0’ or class ‘1’

Right off the bat we can say that in this assignment, we will have to apply classical Machine Learning algorithms to train a simple classification model for binary prediction.

# What I tried

- Standard Logistic Regression using sklearn gave 93% accuracy. (see main.py)
- I tried selecting features using Correlation metric but the results were unsatisfactory.
- I also tried ANOVA method but I couldn't get better results.
- I decided to do some EDA and for that I used 'sweetviz' library.
- The resulting graphs are present in 'visualizations' folder.
- I finally used a simple neural network built using pytorch to tackle this problem.
- The NN was able to give me better results than sklearn library. (see eda_and_nn.ipynb)

# Final Result

NN was able to give the best results at 94% accuracy.

- The results are in the 'submission.csv' file
