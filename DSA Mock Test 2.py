#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1. Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well. You must not use any built-in exponent function or operator. 

 Example 1:
Input: x = 4 Output: 2 Explanation: The square root of 4 is 2, so we return 2.


# In[ ]:


Binary Search Approach

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


You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.


Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4] Output: [7,0,8] Explanation: 342 + 465 = 807.


# In[ ]:


# Definition for singly-linked list.

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode() # for edge cases
        current, carry = dummy, 0

        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)

            carry, curr_value = divmod(val1 + val2 + carry, 10)
            current.next = ListNode(curr_value) # initially we add to dummy
            current = current.next

            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)

        return dummy.next

