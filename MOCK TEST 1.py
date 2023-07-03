#!/usr/bin/env python
# coding: utf-8

# In[ ]:


17.  Write a function that takes a list of numbers as input and returns a new list containing only the even numbers from the input list. Use list comprehension to solve this problem.


# In[ ]:


# using list comprehension

even_nos = [num for num in list1 if num % 2 == 0]
 
print("Even numbers in the list: ", even_nos)


# In[ ]:


Time Complexity: O(N)
Auxiliary Space: O(N), As constant extra space is used.


# In[ ]:


18. Implement a decorator function called ‘timer’ that measures the execution time of a function. The ‘timer’ decorator should print the time taken by the decorated function to execute. Use the ‘time’ module in Python to calculate the execution time.


# In[ ]:


from time import time
  
  
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
  
  
@timer_func
def long_time(n):
    for i in range(n):
        for j in range(100000):
            i*j
  
  
long_time(5)


# In[ ]:


19. Write a function called ‘calculate_mean’ that takes a list of numbers as input and returns the mean (average) of the numbers. The function should calculate the mean using the sum of the numbers divided by the total count.


# In[ ]:


importing mean()
from statistics import mean
 
def Average(lst):
    return mean(lst)
 
# Driver Code
lst = [15, 9, 55, 41, 35, 20, 62, 49]
average = Average(lst)
 
# Printing average of the list
print("Average of the list =", round(average, 2))


# In[ ]:


20. Write a function called ‘perform_hypothesis_test’ that takes two lists of numbers as input, representing two samples. The function should perform a two-sample t-test and return the p-value. Use the ‘scipy.stats’ module in Python to calculate the t-test and p-value.


# In[ ]:


# Import the library
import scipy.stats as stats
 
# Creating data groups
data_group1 = np.array([14, 15, 15, 16, 13, 8, 14,
                        17, 16, 14, 19, 20, 21, 15,
                        15, 16, 16, 13, 14, 12])
 
data_group2 = np.array([15, 17, 14, 17, 14, 8, 12,
                        19, 19, 14, 17, 22, 24, 16,
                        13, 16, 13, 18, 15, 13])
 
# Perform the two sample t-test with equal variances
stats.ttest_ind(a=data_group1, b=data_group2, equal_var=True)


# In[ ]:


# Importing scipy library
import scipy.stats
  
# Determining the p-value
scipy.stats.t.sf(abs(1.87), df=24)

