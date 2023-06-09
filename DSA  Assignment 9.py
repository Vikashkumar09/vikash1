#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Question 1**

Given an integer `n`, return *`true` if it is a power of two. Otherwise, return `false`*.

An integer `n` is a power of two, if there exists an integer `x` such that `n == 2x`.

**Example 1:**
Input: n = 1 

Output: true

**Example 2:**
Input: n = 16 

Output: true

**Example 3:**
Input: n = 3 

Output: false


# In[ ]:


class Solution:
  def isPowerOfTwo(self, n: int) -> bool:
    return False if n < 0 else bin(n).count('1') == 1


# In[ ]:


**Question 2**

Given a number n, find the sum of the first natural numbers.

**Example 1:**

Input: n = 3 

Output: 6

**Example 2:**

Input  : 5 

Output : 15


# In[ ]:


def findSum(n):
    sum = 0
    x = 1
    while x <= n:
        sum = sum + x
        x = x + 1
    return sum
 
 
# Driver code
n = 5
print findSum(n)


# In[ ]:


**Question 3**

****Given a positive integer, N. Find the factorial of N. 

**Example 1:**

Input: N = 5 

Output: 120

**Example 2:**

Input: N = 4

Output: 24


# In[ ]:


def factorial(n):
     
    # single line to find factorial
    return 1 if (n==1 or n==0) else n * factorial(n - 1)
 
# Driver Code
num = 5
print("Factorial of",num,"is",factorial(num))


# In[ ]:


**Question 4**

Given a number N and a power P, the task is to find the exponent of this number raised to the given power, i.e. N^P.

**Example 1 :** 

Input: N = 5, P = 2

Output: 25

**Example 2 :**
Input: N = 2, P = 5

Output: 32


# In[ ]:


Method 1: Iterative approach
    
Here, we have used a for-loop to calculate the power, by iteratively multiplying the number for a given number of times.


def CalculatePower(N,X):
  P=1
  for i in range(1, X+1):
    P=P*N
  return P
 
N,X=2,3
print(CalculatePower(N,X))
N,X=3,4
print(CalculatePower(N,X))


# In[ ]:


Method 2: Recursive Approach
    
Here we have used a recursive function calc_power() to calculate the power of a given number using recursion. This function will return 1 if power is 0, which is base condition for the function.


def calc_power(N, p):
    if p == 0:
        return 1
    return N * calc_power(N, p-1)
 
print(calc_power(4, 2))


# In[ ]:


**Question 5**

Given an array of integers **arr**, the task is to find maximum element of that array using recursion.

**Example 1:**

Input: arr = {1, 4, 3, -5, -4, 8, 6};
Output: 8

**Example 2:**

Input: arr = {1, 4, 45, 6, 10, -8};
Output: 45


# In[ ]:


function to print Minimum element
# using recursion


def findMinRec(A, n):
     
    # if size = 0 means whole array
    # has been traversed
    if (n == 1):
        return A[0]
    return min(A[n - 1], findMinRec(A, n - 1))
 
# Driver Code
if __name__ == '__main__':
    A = [1, 4, 45, 6, -50, 10, 2]
    n = len(A)
    print(findMinRec(A, n))


# In[ ]:


**Question 6**

Given first term (a), common difference (d) and a integer N of the Arithmetic Progression series, the task is to find Nth term of the series.

**Example 1:**

Input : a = 2 d = 1 N = 5
Output : 6
The 5th term of the series is : 6

**Example 2:**

Input : a = 5 d = 2 N = 10
Output : 23
The 10th term of the series is : 23


# In[ ]:



# Arithmetic progression
 
def Nth_of_AP(a, d, N) :
 
    # using formula to find the
    # Nth term t(n) = a(1) + (n-1)*d
    return (a + (N - 1) * d)
      
  
# Driver code
a = 2  # starting number
d = 1  # Common difference
N = 5  # N th term to be find
  
# Display the output
print( "The ", N ,"th term of the series is : ",
       Nth_of_AP(a, d, N))


# In[ ]:


**Question 7**

Given a string S, the task is to write a program to print all permutations of a given string.

**Example 1:**

***Input:***

*S = “ABC”*

***Output:***

*“ABC”, “ACB”, “BAC”, “BCA”, “CBA”, “CAB”*

**Example 2:**

***Input:***

*S = “XY”*

***Output:***

*“XY”, “YX”*


# In[ ]:


def permute(s, answer):
    if (len(s) == 0):
        print(answer, end="  ")
        return
 
    for i in range(len(s)):
        ch = s[i]
        left_substr = s[0:i]
        right_substr = s[i + 1:]
        rest = left_substr + right_substr
        permute(rest, answer + ch)
 
 
# Driver Code
answer = ""
 
s = "ABC"
 
print("All possible strings are : ")
permute(s, answer)


# In[ ]:


**Question 8**

Given an array, find a product of all array elements.

**Example 1:**

Input  : arr[] = {1, 2, 3, 4, 5}
Output : 120
**Example 2:**

Input  : arr[] = {1, 6, 3}
Output : 18


# In[ ]:


class Solution:
  def productExceptSelf(self, nums: List[int]) -> List[int]:
    n = len(nums)
    left_product = [1] * n # initialize left_product array with 1
    right_product = [1] * n # initialize right_product array with 1
    # calculate the product of elements to the left of each element
    for i in range(1, n):
        left_product[i] = left_product[i - 1] * nums[i - 1]

    # calculate the product of elements to the right of each element
    for i in range(n - 2, -1, -1):
        right_product[i] = right_product[i + 1] * nums[i + 1]

    # calculate the product of all elements except nums[i]
    result = [left_product[i] * right_product[i] for i in range(n)]

    return result


# In[ ]:




