#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Question 1
Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2),..., (an, bn) such that the sum of min(ai, bi) for all i is maximized. Return the maximized sum.

**Example 1:**
Input: nums = [1,4,3,2]
Output: 4

**Explanation:** All possible pairings (ignoring the ordering of elements) are:

1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
So the maximum possible sum is 4


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


Total time complexity will be O(nlogn + n/2) = O(nlogn)
It only needs O(1) extra space.


# In[ ]:


# Question 2
Alice has n candies, where the ith candy is of type candyType[i]. Alice noticed that she started to gain weight, so she visited a doctor. 

The doctor advised Alice to only eat n / 2 of the candies she has (n is always even). Alice likes her candies very much, and she wants to eat the maximum number of different types of candies while still following the doctor's advice. 

Given the integer array candyType of length n, return the maximum number of different types of candies she can eat if she only eats n / 2 of them.

Example 1:
Input: candyType = [1,1,2,2,3,3]
Output: 3

Explanation: Alice can only eat 6 / 2 = 3 candies. Since there are only 3 types, she can eat one of each type.


# In[ ]:


Approach 1: Brute Force
Intuition and Algorithm

One way to find the number of unique candies is to traverse over each element in candyType, checking whether or not we've already seen a candy of this same type. We can do this check by iterating over all elements before the current element. If any of those are of the same type, then this is not a unique candy. We should keep track of the number of unique candies we find.


# In[ ]:


class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        # We need to count how many unique candies are in the array.
        unique_candies = 0
        # For each candy, we're going to check whether or not we've already
        # seen a candy identical to it.
        for i in range(len(candyType)):
            # Check if we've already seen a candy the same as candyType[i].
            for j in range(0, i):
                # If this candy is the same as previous one, we don't need to 
                # check further.
                if candyType[i] == candyType[j]:
                    break
            # Confused? An "else" after a "for" is an awesome Python feature.
            # The code in the "else" only runs if the "for" loop runs without a break.
            # In this case, we know that if we didn't "break" out of the loop, then 
            # candyType[i] is unique.
            # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
            else:
                unique_candies += 1
        # The answer is the minimum out of the number of unique candies, and 
        # half the length of the candyType array.
        return min(unique_candies, len(candyType) // 2)


# In[ ]:


Approach 2: Sorting
    
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        # We start by sorting candyType.
        candyType.sort()
        # The first candy is always unique.
        unique_candies = 1
        # For each candy, starting from the *second* candy...
        for i in range(1, len(candyType)):
            # This candy is unique if it is different to the one
            # immediately before it.
            if candyType[i] != candyType[i - 1]:
                unique_candies += 1
            # Optimization: We should terminate the loop if unique_candies
            # is now at the maxium she can eat.
            if unique_candies == len(candyType) // 2:
                break
        # Like before, the answer is the minimum out of the number of unique candies, and 
        # half the length of the candyType array.
        return min(unique_candies, len(candyType) // 2)


# In[ ]:


Approach 3: Using a Hash Set
    
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        # Count the number of unique candies by creating a set with
        # candyType, and then taking its length.
        unique_candies = len(set(candyType))
        # And find the answer in the same way as before.
        return min(unique_candies, len(candyType) // 2)


# In[ ]:


Time complexity : O(N)

Space complexity : O(N)


# In[ ]:


# Question 3

We define a harmonious array as an array where the difference between its maximum value
and its minimum value is exactly 1.

Given an integer array nums, return the length of its longest harmonious subsequence
among all its possible subsequences.

A subsequence of an array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

Example 1:
Input: nums = [1,3,2,2,5,2,3,7]
Output: 5

Explanation: The longest harmonious subsequence is [3,2,2,2,3].


# In[ ]:


class Solution:
    def findLHS(self, nums: List[int]) -> int:
        my_dict = defaultdict(int)
        # keep in dict the number of times each number appears:
        for num in nums:
            my_dict[num]+=1
            
        max_ = 0
        # for each number in dict check if it+its following number is more than previous max:
        for num in my_dict.keys():
            if my_dict.get(num+1):
                max_ = max(max_, my_dict[num] + my_dict.get(num+1))
        return max_


# In[ ]:


# Question 4

You have a long flowerbed in which some of the plots are planted, and some are not.
However, flowers cannot be planted in adjacent plots.
Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.

Example 1:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true


# In[ ]:


Approach for this Problem :
    
Loop through each position in the flowerbed.
If the current position is 0 and the adjacent positions (if any) are also 0, then plant a flower at that position and decrement n.
If n becomes 0, return true as all flowers have been planted.
If the loop finishes and n is still positive, return false as there are not enough free spaces in the flowerbed to plant all flowers.


# In[ ]:


class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        for i in range(len(flowerbed)):
            if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i+1] == 0):
                flowerbed[i] = 1
                n -= 1
        if n > 0:
            return False
        return True


# In[ ]:


Time complexity : O(n), where n is the length of the flowerbed. Space complexity : O(1), no extra space used.


# In[ ]:


# Question 5

Given an integer array nums, find three numbers whose product is maximum and return the maximum product.

Example 1:
Input: nums = [1,2,3]
Output: 6


# In[ ]:


Heap Concept

class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
      nums.sort()
      pos=heapq.nlargest(3,nums)
      neg=heapq.nsmallest(2,nums)
      return max(neg[0]*neg[1]*pos[0],pos[-1]*pos[-2]*pos[-3])


# In[ ]:


# Question 6

Given an array of integers nums which is sorted in ascending order, and an integer target,
write a function to search target in nums. If target exists, then return its index. Otherwise,
return -1.

You must write an algorithm with O(log n) runtime complexity.

Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4

Explanation: 9 exists in nums and its index is 4


# In[ ]:


Approach Binary Search

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums)-1

        while low <= high:
            mid = (low + high) // 2 

            if nums[mid] == target:
                return mid
            if target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        
        return -1


# In[ ]:


Time complexity: O(log(n)) Space complexity: O(1)


# In[ ]:


# Question 7

An array is monotonic if it is either monotone increasing or monotone decreasing.

An array nums is monotone increasing if for all i <= j, nums[i] <= nums[j]. An array nums is
monotone decreasing if for all i <= j, nums[i] >= nums[j].

Given an integer array nums, return true if the given array is monotonic, or false otherwise.

Example 1:
Input: nums = [1,2,2,3]
Output: true


# In[ ]:


# Check if given array is Monotonic
def isMonotonic(A):
    x, y = [], []
    x.extend(A)
    y.extend(A)
    x.sort()
    y.sort(reverse=True)
    if(x == A or y == A):
        return True
    return False
 
 
# Driver program
A = [6, 5, 4, 4]
 
# Print required result
print(isMonotonic(A))


# In[ ]:


Time Complexity: O(N*logN), where N is the length of the array. Auxiliary space: O(N), extra space is required for lists x and y.


# In[ ]:


# Question 8

You are given an integer array nums and an integer k.

In one operation, you can choose any index i where 0 <= i < nums.length and change nums[i] to nums[i] + x where x is an integer from the range [-k, k]. You can apply this operation at most once for each index i.

The score of nums is the difference between the maximum and minimum elements in nums.

Return the minimum score of nums after applying the mentioned operation at most once for each index in it.

Example 1:
Input: nums = [1], k = 0
Output: 0

Explanation: The score is max(nums) - min(nums) = 1 - 1 = 0.


# In[ ]:


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        left, right = [1] * (l := len(nums)), [1]*l
        for i in range(1, l):
            left[i] *= left[i-1]*nums[i-1]
            right[-i-1] *= right[-i]*nums[-i]
        return [lv*rv for lv, rv in zip(left, right)]


# In[ ]:


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = []
        
        acc = 1
        for n in nums:
            res.append(acc)
            acc *= n

        acc = 1
        for i in reversed(range(len(nums))):
            res[i] *= acc
            acc *= nums[i]
            
        return res


# In[ ]:


Time: O(n) for two passes over nums Space: O(1).

