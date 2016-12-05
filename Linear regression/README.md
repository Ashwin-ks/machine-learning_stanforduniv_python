Linear regression is an approach to modeling the relationship between a dependent variable and one or more independent variables (if there's one independent variable then it's called simple linear regression, and if there's more than one independent variable then it's called multiple linear regression). 

linear_regression.py :-Implementing simple linear regression to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and we have data for profits and populations from the cities(ex1data1.txt). We'd like to figure out what the expected profit of a new food truck might be given only the population of the city that it would be placed in.

In this implementation we're going to use an optimization technique called gradient descent to find the parameters theta. If you're familiar with linear algebra, you may be aware that there's another way to find the optimal parameters for a linear model called the "normal equation" which basically solves the problem at once using a series of matrix calculations. However, the issue with this approach is that it doesn't scale very well for large data sets. In contrast, we can use variants of gradient descent and other optimization methods to scale to data sets of unlimited size, so for machine learning problems this approach is more practical.

multiple_linear_regression.py: -predict the price that a house will sell for.We have more than one dependent variable(ex1data2.txt). We're given both the size of the house in square feet, and the number of bedrooms in the house.We'll use Gradient descent algorithm to fit the linear model for multivariate linear regression

linear_regression_scikitlearn.py:- simple linear regression task from part 1 using scikit-learn's linear regression class.