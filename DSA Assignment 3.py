#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Question 1
Given an integer array nums of length n and an integer target, find three integers
in nums such that the sum is closest to the target.
Return the sum of the three integers.

You may assume that each input would have exactly one solution.

Example 1:
Input: nums = [-1,2,1,-4], target = 1
Output: 2

Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).


# In[ ]:


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        s=float('inf')
        t=len(nums)
        nums.sort()
        for x in range(t-2):
            i=x+1
            j=t-1
            while i<j:
                s1=nums[x]+nums[i]+nums[j]
                #print(x,i,j,'here')
                if s1==target:
                    return target
                if abs(target-s)>abs(target-s1):
                    #print(x,i,j,'-------',s1)
                    s=s1
                if s1<target:
                    i+=1
                else:
                    j-=1
        return s


# In[ ]:


Question 2
Given an array nums of n integers, return an array of all the unique quadruplets
[nums[a], nums[b], nums[c], nums[d]] such that:
           ● 0 <= a, b, c, d < n
           ● a, b, c, and d are distinct.
           ● nums[a] + nums[b] + nums[c] + nums[d] == target

You may return the answer in any order.

Example 1:
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]


# In[ ]:


# Store the pair of indices
class Pair:
    def __init__(self, x, y):
        self.index1 = x
        self.index2 = y
 
# Function to find the all the unique quadruplets
# with the elements at different indices
def GetQuadruplets(nums, target):
    # Store the sum mapped to a list of pair indices
    map = {}
 
    # Generate all possible pairs for the map
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            # Find the sum of pairs of elements
            sum = nums[i] + nums[j]
 
            # If the sum doesn't exist then update with the new pairs
            if sum not in map:
                map[sum] = [Pair(i, j)]
            # Otherwise, add the new pair of indices to the current sum
            else:
                map[sum].append(Pair(i, j))
 
    # Store all the Quadruplets
    ans = set()
 
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            lookUp = target - (nums[i] + nums[j])
 
            # If the sum with value (K - sum) exists
            if lookUp in map:
                # Get the pair of indices of sum
                temp = map[lookUp]
 
                for pair in temp:
                    # Check if i, j, k and l are distinct or not
                    if pair.index1 != i and pair.index1 != j and pair.index2 != i and pair.index2 != j:
                        l1 = [nums[pair.index1], nums[pair.index2], nums[i], nums[j]]
                         
                        # Sort the list to avoid duplicacy
                        l1.sort()
                         
                        # Update the set
                        ans.add(tuple(l1))
 
    # Print all the Quadruplets
    print(*reversed(list(ans)), sep = '\n')
 
# Driver Code
arr = [1, 0, -1, 0, -2, 2]
K = 0
GetQuadruplets(arr, K)


# In[ ]:


Time Complexity: O(N2 * log N)
Auxiliary Space: O(N2)


# In[ ]:


**Question 3**


A permutation of an array of integers is an arrangement of its members into a
sequence or linear order.

For example, for arr = [1,2,3], the following are all the permutations of arr:
[1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].

The next permutation of an array of integers is the next lexicographically greater
permutation of its integer. More formally, if all the permutations of the array are
sorted in one container according to their lexicographical order, then the next
permutation of that array is the permutation that follows it in the sorted container.

If such an arrangement is not possible, the array must be rearranged as the
lowest possible order (i.e., sorted in ascending order).

● For example, the next permutation of arr = [1,2,3] is [1,3,2].
● Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
● While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not
have a lexicographical larger rearrangement.

Given an array of integers nums, find the next permutation of nums.
The replacement must be in place and use only constant extra memory.

**Example 1:**
Input: nums = [1,2,3]
Output: [1,3,2]


# In[ ]:


class Solution(object):
    def nextPermutation(self, nums):
        n = len(nums)
        k, l = n - 2, n - 1
        while k >= 0 and nums[k] >= nums[k + 1]:
            k -= 1
        if k < 0:
            nums.reverse()
        else:
            while l > k and nums[l] <= nums[k]:
                l -= 1
            nums[k], nums[l] = nums[l], nums[k]
            nums[k + 1:n] = reversed(nums[k + 1:n])


# In[ ]:


Question 4

Given a sorted array of distinct integers and a target value, return the index if the
target is found. If not, return the index where it would be if it were inserted in
order.

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2


# In[ ]:


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        for i in range(len(nums)):
            if(nums[i]>=target):
                return i
        return len(nums)


# In[ ]:


**Question 5**

You are given a large integer represented as an integer array digits, where each
digits[i] is the ith digit of the integer. The digits are ordered from most significant
to least significant in left-to-right order. The large integer does not contain any
leading 0's.

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
        digits=[str(i) for i in digits]
        s=''.join(digits)
        a=int(s)+1
        l=list(str(a))
        l=[int(i) for i in l]
        return l


# In[ ]:


Question 6

Given a non-empty array of integers nums, every element appears twice except
for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only
constant extra space.

Example 1:
Input: nums = [2,2,1]
Output: 1


# In[ ]:


Using Hashing
    
function to find the once 
# appearing element in array
def findSingle( ar, n):
      
    res = ar[0]
      
    # Do XOR of all elements and return
    for i in range(1,n):
        res = res ^ ar[i]
      
    return res
  
# Driver code
ar = [2, 3, 5, 4, 5, 3, 4]
print "Element occurring once is", findSingle(ar, len(ar))


# In[ ]:


Time Complexity: O(n)
Auxiliary Space: O(1)


# In[ ]:


Question 7


You are given an inclusive range [lower, upper] and a sorted unique integer array
nums, where all elements are within the inclusive range.

A number x is considered missing if x is in the range [lower, upper] and x is not in
nums.

Return the shortest sorted list of ranges that exactly covers all the missing
numbers. That is, no element of nums is included in any of the ranges, and each
missing number is covered by one of the ranges.

Example 1:
Input: nums = [0,1,3,50,75], lower = 0, upper = 99
Output: [[2,2],[4,49],[51,74],[76,99]]

Explanation: The ranges are:
[2,2]
[4,49]
[51,74]
[76,99]


# In[ ]:


class Solution:
  def summaryRanges(self, nums: List[int]) -> List[str]:
    ans = []

    i = 0
    while i < len(nums):
      begin = nums[i]
      while i < len(nums) - 1 and nums[i] == nums[i + 1] - 1:
        i += 1
      end = nums[i]
      if begin == end:
        ans.append(str(begin))
      else:
        ans.append(str(begin) + "->" + str(end))
      i += 1

    return ans


# In[ ]:


Question 8

Given an array of meeting time intervals where intervals[i] = [starti, endi],
determine if a person could attend all meetings.

Example 1:
Input: intervals = [[0,30],[5,10],[15,20]]
Output: false


# In[ ]:


Sort the list by start time and iterate the sorted list. If the current start time is less than previous end time, then there is conflict and you can not attend all meeting.


# In[ ]:



def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        new_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(1,len(new_intervals)):
            if new_intervals[i-1][1] > new_intervals[i][0]:return False
        return True

