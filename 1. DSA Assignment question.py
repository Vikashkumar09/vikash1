#!/usr/bin/env python
# coding: utf-8

# # 1. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# 
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# 
# You can return the answer in any order.
# 
# **Example:**
# Input: nums = [2,7,11,15], target = 9
# Output0 [0,1]
# 
# **Explanation:** Because nums[0] + nums[1] == 9, we return [0, 1][

# In[ ]:


Approach 1: Brute Force
    
    class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[j] == target - nums[i]:
                    return [i, j]
                    
Time complexity: O(n^2)

Space complexity: 0(1).
The space required does not depend on the size of the input array, so only constant space is used.


# In[ ]:


Approach 2: Two-pass Hash Table
Intuition

To improve our runtime complexity, we need a more efficient way to check if the complement exists in the array. If the complement exists, we need to get its index. What is the best way to maintain a mapping of each element in the array to its index? A hash table.

We can reduce the lookup time from O(n)O(n)O(n) to O(1)O(1)O(1) by trading space for speed. A hash table is well suited for this purpose because it supports fast lookup in near constant time. I say "near" because if a collision occurred, a lookup could degenerate to O(n)O(n)O(n) time. However, lookup in a hash table should be amortized O(1)O(1)O(1) time as long as the hash function was chosen carefully.

Algorithm

A simple implementation uses two iterations. In the first iteration, we add each element's value as a key and its index as a value to the hash table. Then, in the second iteration, we check if each element's complement (target−nums[i]target - nums[i]target−nums[i]) exists in the hash table. If it does exist, we return current element's index and its complement's index. Beware that the complement must not be nums[i]nums[i]nums[i] itself!


# In[ ]:


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            hashmap[nums[i]] = i
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap and hashmap[complement] != i:
                return [i, hashmap[complement]] 


# In[ ]:


Time complexity: O(n)
We traverse the list containing nnn elements exactly twice. Since the hash table reduces the lookup time to 0(1), the overall time complexity is O(n).

Space complexity: O(n).
The extra space required depends on the number of items stored in the hash table, which stores exactly n elements.



# In[ ]:


**Q2.** Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:

- Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
- Return k.

**Example :**
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_*,_*]
**Explanation:** Your function should return k = 2, with the first two elements of nums being 2. It does not matter what you leave beyond the returned k (hence they are underscores)[


# In[ ]:


class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        count = 0
        for i in range(len(nums)):
            if nums[i] == val:
                continue
            nums[count] = nums[i]
            count += 1
        
        return count


# In[ ]:


Time Complexity
We are scanning the array once, hence the time complexity will be O(n).

Space Complexity
Since we have no used any extra array, the space complexity will be O(1).


# In[ ]:


**Q3.** Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

**Example 1:**
Input: nums = [1,3,5,6], target = 5

Output: 2


# In[ ]:


Brute Force approach
The brute force approach is to linearly iterate over the array and find the index where the target can be inserted

Since the elements are sorted we can use binary search algorithm to find that correct index.


BINARY SEARCH APPROACH

- set start = 0 and end = N - 1.
- loop while (start <= end)
  - mid = (start + end)/2

  - if target > nums[mid]
    - start = mid + 1
  - else if target < nums[mid]
    - end = mid - 1
  - else
    - return mid

- return start


# In[ ]:


C++ Solution

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int start = 0;
        int end = nums.size()-1;

        while(start <= end){
            int mid = (start + end)/2;

            if(target > nums[mid]){
                start = mid + 1;
            }else if(target < nums[mid]){
                end = mid - 1;
            }else{
                return mid;
            }
        }

        return start;
    }
};


The solution is easy and quick to implement but it takes O(n) time.


# In[ ]:


**Q4.** You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.

**Example 1:**
Input: digits = [1,2,3]
Output: [1,2,4]

**Explanation:** The array represents the integer 123.

Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].


# In[ ]:


class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        digit_length = len(digits)
        
        i = digit_length - 1
        
        while digits[i] == 9 and i >= 0:
            i -= 1
        
        if i == -1:
            results = [0]*(digit_length + 1)
            results[0] = 1
            return results
        
        results = [0]*(digit_length)
        
        results[i] = digits[i] + 1
        
        for j in range(i-1, -1, -1):
            results[j] = digits[j]
        
        return results


# In[ ]:


Time Complexity
We are traversing an whole array (from right to left) until we find space, so time complexity will be O(n).

Space Complexity
Since we have used an extra array with +1 more space, the space complexity will be O(n+1).


# In[ ]:


**Q5.** You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

**Example 1:**
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

**Explanation:** The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.


# In[ ]:


class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        i = int(m - 1)
        j = int(n - 1)
        k = int(m + n - 1)
        
        while (k >= 0):
            if i>=0 and j>=0:
                if(nums2[j] > nums1[i]):
                    nums1[k] = nums2[j]
                    k -= 1
                    j -= 1
                else:
                    nums1[k] = nums1[i]
                    k -= 1
                    i -= 1
            elif j >= 0:
                nums1[k] = nums2[j]
                k -= 1
                j -= 1
            else:
                break


# In[ ]:


Time Complexity
Here, we are using while loop with (m+n), so time complexity will be O(m+n).

Space Complexity
Since we have not used any extra array, so, the space complexity will be O(1).


# In[ ]:


**Q6.** Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

**Example 1:**
Input: nums = [1,2,3,1]

Output: true


# In[ ]:


class Solution(object):
    def containsDuplicate(self, nums):
        hset = set()
        for idx in nums:
            if idx in hset:
                return True
            else:
                hset.add(idx)
                
# Time complexity: O(n)
# Space complexity: O(n)               


# In[ ]:


**Q7.** Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the nonzero elements.

Note that you must do this in-place without making a copy of the array.

**Example 1:**
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]


# In[ ]:


It uses two pointers technique.

class Solution:
    def moveZeroes(self, nums: list) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]

            # wait while we find a non-zero element to
            # swap with you
            if nums[slow] != 0:
                slow += 1
                
Time complexity: O(n). Our fast pointer does not visit the same spot twice.
Space complexity: O(1). All operations are made in-place.


# In[ ]:


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero_pos = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero_pos] = nums[zero_pos], nums[i]
                zero_pos += 1   


# In[ ]:


**Q8.** You have a set of integers s, which originally contains all the numbers from 1 to n. Unfortunately, due to some error, one of the numbers in s got duplicated to another number in the set, which results in repetition of one number and loss of another number.

You are given an integer array nums representing the data status of this set after the error.

Find the number that occurs twice and the number that is missing and return them in the form of an array.

**Example 1:**
Input: nums = [1,2,2,4]
Output: [2,3]


# In[ ]:


class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        s = n*(n+1)//2
        miss = s - sum(set(nums))
        duplicate = sum(nums) + miss - s
        return [duplicate, miss]


# In[ ]:


Time complexity: O(n)

