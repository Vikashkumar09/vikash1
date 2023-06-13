#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Question 1**

Given a non-negative integer `x`, return *the square root of* `x` *rounded down to the nearest integer*. The returned integer should be **non-negative** as well.

You **must not use** any built-in exponent function or operator.

- For example, do not use `pow(x, 0.5)` in c++ or `x ** 0.5` in python.

**Example 1:**

Input: x = 4
Output: 2
Explanation: The square root of 4 is 2, so we return 2.


# In[ ]:


1. Normal Math Approach

class Solution:
    def mySqrt(self, x: int) -> int:
        number=1
        while number*number<=x:
            number+=1
        return number


# In[ ]:


2. Binary Search Approach

class Solution:
    def mySqrt(self, x: int) -> int:
        left,right=1,x
        while left<=right:
            mid=(left+right)//2
            if mid*mid==x:
                return mid
            if mid*mid>x:
                right=mid-1
            else:
                left=mid+1
        return right


# In[ ]:


**Question 2**

A peak element is an element that is strictly greater than its neighbors.

Given a **0-indexed** integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any of the peaks**.

You may imagine that `nums[-1] = nums[n] = -∞`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in `O(log n)` time.

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.


# In[ ]:


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid+1]:
                right = mid
            else:
                left = mid + 1
                
        return left


# In[ ]:


**Question 3**

****

Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return *the only number in the range that is missing from the array.*

**Example 1:**
Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.


# In[ ]:


class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums) + 1
        total = (n * (n-1)) // 2
        
        for num in nums:
            total -= num
        
        return total


# In[ ]:


**Question 4**

Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive.

There is only **one repeated number** in `nums`, return *this repeated number*.

You must solve the problem **without** modifying the array `nums` and uses only constant extra space.

**Example 1:**
    
Input: nums = [1,3,4,2,2]
Output: 2


# In[ ]:


class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        for i in nums:
            nums[abs(i)-1] *= -1
            if nums[abs(i)-1] > 0: return abs(i)


# In[ ]:


**Question 5**

Given two integer arrays `nums1` and `nums2`, return *an array of their intersection*. Each element in the result must be **unique** and you may return the result in **any order**.

**Example 1:**

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]


# In[ ]:


class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums1 to set
        nums1_set = set(nums1)
        # Create a set to store the intersection
        intersection_set = set()
        # Loop through nums2 and check if each element is in nums1_set
        for num in nums2:
            if num in nums1_set:
                intersection_set.add(num)
        # Convert the intersection set back to a list and return
        return list(intersection_set)


# In[ ]:


**Question 6**

Suppose an array of length `n` sorted in ascending order is **rotated** between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that **rotating** an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of **unique** elements, return *the minimum element of this array*.

You must write an algorithm that runs in `O(log n) time.`

**Example 1:**

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.


# In[ ]:


class Solution:
    def findMin(self, nums: List[int]) -> int:
        low,high=0,len(nums)-1
        n=len(nums)
        while low<=high:
            mid=low+(high-low)//2
            if nums[mid]<=nums[(mid-1+n)%n] and nums[mid]<=nums[(mid+1)%n]:
                return nums[mid]
            if nums[mid]>=nums[low]:
                if nums[high]>=nums[mid]:
                    high=mid-1
                else:
                    low=mid+1
            else:
                high=mid-1

        return -1


# In[ ]:


**Question 7**

Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

You must write an algorithm with `O(log n)` runtime complexity.

**Example 1:**

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4


# In[ ]:


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        def search(x):
            lo, hi = 0, len(nums)           
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < x:
                    lo = mid+1
                else:
                    hi = mid                    
            return lo
        
        lo = search(target)
        hi = search(target+1)-1
        
        if lo <= hi:
            return [lo, hi]
                
        return [-1, -1]


# In[ ]:


**Question 8**

Given two integer arrays `nums1` and `nums2`, return *an array of their intersection*. Each element in the result must appear as many times as it shows in both arrays and you may return the result in **any order**.


# In[ ]:


class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Convert nums1 to set
        nums1_set = set(nums1)
        # Create a set to store the intersection
        intersection_set = set()
        # Loop through nums2 and check if each element is in nums1_set
        for num in nums2:
            if num in nums1_set:
                intersection_set.add(num)
        # Convert the intersection set back to a list and return
        return list(intersection_set)

