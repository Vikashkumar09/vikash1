#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Question 1**

Given three integer arrays arr1, arr2 and arr3 **sorted** in **strictly increasing** order, return a sorted array of **only** the integers that appeared in **all** three arrays.

**Example 1:**

Input: arr1 = [1,2,3,4,5], arr2 = [1,2,5,7,9], arr3 = [1,3,4,5,8]

Output: [1,5]

**Explanation:** Only 1 and 5 appeared in the three arrays.


# In[ ]:


def findCommon(ar1, ar2, ar3, n1, n2, n3):
 
    # Initialize starting indexes for ar1[], ar2[] and ar3[]
    i, j, k = 0, 0, 0
 
    # Iterate through three arrays while all arrays have elements
    while (i < n1 and j < n2 and k < n3):
 
        # If x = y and y = z, print any of them and move ahead
        # in all arrays
        if (ar1[i] == ar2[j] and ar2[j] == ar3[k]):
            print ar1[i],
            i += 1
            j += 1
            k += 1
 
        # x < y
        elif ar1[i] < ar2[j]:
            i += 1
 
        # y < z
        elif ar2[j] < ar3[k]:
            j += 1
 
        # We reach here when x > y and z < y, i.e., z is smallest
        else:
            k += 1
 
 
# Driver program to check above function
ar1 = [1, 5, 10, 20, 40, 80]
ar2 = [6, 7, 20, 80, 100]
ar3 = [3, 4, 15, 20, 30, 70, 80, 120]
n1 = len(ar1)
n2 = len(ar2)
n3 = len(ar3)
print "Common elements are",
findCommon(ar1, ar2, ar3, n1, n2, n3)


# In[ ]:


Time complexity of the above solution is O(n1 + n2 + n3).
In the worst case, the largest sized array may have all small elements and middle-sized array has all middle elements.
Auxiliary Space:   O(1)


# In[ ]:


**Question 2**

Given two **0-indexed** integer arrays nums1 and nums2, return *a list* answer *of size* 2 *where:*

- answer[0] *is a list of all **distinct** integers in* nums1 *which are **not** present in* nums2*.*
- answer[1] *is a list of all **distinct** integers in* nums2 *which are **not** present in* nums1.

**Note** that the integers in the lists may be returned in **any** order.

**Example 1:**

**Input:** nums1 = [1,2,3], nums2 = [2,4,6]

**Output:** [[1,3],[4,6]]

**Explanation:**

For nums1, nums1[1] = 2 is present at index 0 of nums2, whereas nums1[0] = 1 and nums1[2] = 3 are not present in nums2. Therefore, answer[0] = [1,3].

For nums2, nums2[0] = 2 is present at index 1 of nums1, whereas nums2[1] = 4 and nums2[2] = 6 are not present in nums2. Therefore, answer[1] = [4,6].


# In[ ]:


class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
         set1,set2=set(nums1),set(nums2)
         return[list(set1-set2),list(set2-set1)]

#other possible answer for this problem is
class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        set1=set(nums1)
        set2=set(nums2)
        res=[[],[]]

        for i in set1:
            if i not in set2:
                res[0].append(i)
        for i in set2:
            if i not in set1:
                res[1].append(i)
        return res

#other possible answer for this problem
class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        set1,set2=set(nums1),set(nums2)
        return[set1-set2,set2-set1]


# In[ ]:


**Question 3**

Given a 2D integer array matrix, return *the **transpose** of* matrix.

The **transpose** of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.

**Example 1:**

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]

Output: [[1,4,7],[2,5,8],[3,6,9]]


# In[ ]:


class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        return [e for e in zip(*matrix)]


# In[ ]:


**Question 4**
   
Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2), ..., (an, bn) such that the sum of min(ai, bi) for all i is **maximized**. Return *the maximized sum*.

**Example 1:**

Input: nums = [1,4,3,2]

Output: 4

**Explanation:** All possible pairings (ignoring the ordering of elements) are:

1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3

2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3

3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4

So the maximum possible sum is 4.


# In[ ]:


class Solution:
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        # approach: sort list, take smaller ones and sum them up

        nums.sort()
        return sum(nums[::2])


# In[ ]:


**Question 5**

You have n coins and you want to build a staircase with these coins. The staircase consists of k rows where the ith row has exactly i coins. The last row of the staircase **may be** incomplete.

Given the integer n, return *the number of **complete rows** of the staircase you will build*.

**Example 1:**

[]()

get_ipython().system('[v2.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4bd91cfa-d2b1-47b3-8197-a72e8dcfff4b/v2.jpg)')

**Input:** n = 5

**Output:** 2

**Explanation:** Because the 3rd row is incomplete, we return 2.


# In[ ]:


class Solution:
def arrangeCoins(self, n: int) -> int:

    return (int)((2 * n + 0.25)**0.5 - 0.5)
	
    
    
    
    
	2nd solution
    
	
	left, right = 0, n
    while left <= right:
        mid = (left + right) // 2
        k = (mid*(mid+1))//2
        if k == n:
            return mid
        elif (k < n):
            left = mid + 1
        else:
            right = mid - 1
    return right
	
    
    
    
3rd Solution
	
	if n == 1:
        return 1
    temp = 0
    while n >= 0:
        n -= temp
        temp += 1
    return temp-2


# In[ ]:


**Question 6**

Given an integer array nums sorted in **non-decreasing** order, return *an array of **the squares of each number** sorted in non-decreasing order*.

**Example 1:**

Input: nums = [-4,-1,0,3,10]

Output: [0,1,9,16,100]

**Explanation:** After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100]


# In[ ]:


class Solution(object):
    def sortedSquares(self, nums):
        result = [0]*len(nums)
        i = 0
        j = len(nums)-1
        k = len(nums)-1
        while k>=0:
            if nums[i]*nums[i] <= nums[j]*nums[j]:
                result[k] = nums[j]*nums[j]
                k-=1
                j-=1
            else:
                result[k] = nums[i]*nums[i]
                i+=1
                k-=1
        return result        


# In[ ]:


**Question 7**


You are given an m x n matrix M initialized with all 0's and an array of operations ops, where ops[i] = [ai, bi] means M[x][y] should be incremented by one for all 0 <= x < ai and 0 <= y < bi.

Count and return *the number of maximum integers in the matrix after performing all the operations*

**Example 1:**

get_ipython().system('[q4.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4d0890d0-7bc7-4f59-be8e-352d9f3c1c52/q4.jpg)')

**Input:** m = 3, n = 3, ops = [[2,2],[3,3]]

**Output:** 4

**Explanation:** The maximum integer in M is 2, and there are four of it in M. So return 4.


# In[ ]:


class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        length = len(ops)
        if length == 0:
            return m*n
        result = [ops[0][0] , ops[0][1]]
        for i in range(1,length):
            result[0] = min(result[0] , ops[i][0])
            result[1] = min(result[1] , ops[i][1])
        return result[0]*result[1]


# In[ ]:


**Question 8**


Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].

*Return the array in the form* [x1,y1,x2,y2,...,xn,yn].

**Example 1:**

**Input:** nums = [2,5,1,3,4,7], n = 3

**Output:** [2,3,5,4,1,7]

**Explanation:** Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].


# In[ ]:


def shuffle(self, nums: List[int], n: int) -> List[int]:
        ls=[]
        for i in range(n):
            ls+=[nums[i]]
            ls+=[nums[i+n]]
        return ls


# In[ ]:


Here's a step by step description of the code:

Initialize an empty list ls to store the shuffled elements.
Loop through nums by incrementing the index i from 0 to n-1.
For each iteration of the loop, add the element at index i in nums to ls using ls += [nums[i]].
After that, add the element at index i + n in nums to ls using ls += [nums[i + n]].
Repeat steps 3 and 4 for all iterations of the loop until i is equal to n-1.
After the loop, return the ls list as the shuffled result in the form [x1, y1, x2, y2, ..., xn, yn].

