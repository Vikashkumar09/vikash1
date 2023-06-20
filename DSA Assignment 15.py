#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Question 1**

Given an array **arr[ ]** of size **N** having elements, the task is to find the next greater element for each element of the array in order of their appearance in the array.Next greater element of an element in the array is the nearest element on the right which is greater than the current element.If there does not exist next greater of current element, then next greater element for current element is -1. For example, next greater of the last element is always -1.

**Example 1:**

Input:
N = 4, arr[] = [1 3 2 4]
Output:
3 4 4 -1
Explanation:
In the array, the next larger element
to 1 is 3 , 3 is 4 , 2 is 4 and for 4 ?
since it doesn't exist, it is -1.


# In[ ]:


def printNGE(arr):
 
    for i in range(0, len(arr), 1):
 
        next = -1
        for j in range(i+1, len(arr), 1):
            if arr[i] < arr[j]:
                next = arr[j]
                break
 
        print(str(arr[i]) + " -- " + str(next))
 
 
# Driver program to test above function
arr = [11, 13, 21, 3]
printNGE(arr)


# In[ ]:


**Question 2**

Given an array **a** of integers of length **n**, find the nearest smaller number for every element such that the smaller element is on left side.If no small element present on the left print -1.

**Example 1:**

Input: n = 3
a = {1, 6, 2}
Output: -1 1 1
Explaination: There is no number at the
left of 1. Smaller number than 6 and 2 is 1.


# In[ ]:


def printPrevSmaller(arr, n):
 
    # Always print empty or '_' for
    # first element
    print("_, ", end="")
 
    # Start from second element
    for i in range(1, n ):
     
        # look for smaller element
        # on left of 'i'
        for j in range(i-1 ,-2 ,-1):
         
            if (arr[j] < arr[i]):
             
                print(arr[j] ,", ",
                            end="")
                break
 
        # If there is no smaller
        # element on left of 'i'
        if (j == -1):
            print("_, ", end="")
 
# Driver program to test insertion
# sort
arr = [1, 3, 0, 2, 5]
n = len(arr)
printPrevSmaller(arr, n)
 
The time complexity of the above solution is O(n2).

Space Complexity: O(1)


# In[ ]:


3.  Implement a Stack using two queues **q1** and **q2**.

**Example 1:**
    
 Input:
push(2)
push(3)
pop()
push(4)
pop()
Output:3 4
Explanation:
push(2) the stack will be {2}
push(3) the stack will be {2 3}
pop()   poped element will be 3 the
        stack will be {2}
push(4) the stack will be {2 4}
pop()   poped element will be 4
   


# In[ ]:


class Stack:

    def __init__(self):
        self._queue = collections.deque()

    def push(self, x):
        q = self._queue
        q.append(x)
        for _ in range(len(q) - 1):
            q.append(q.popleft())
        
    def pop(self):
        return self._queue.popleft()

    def top(self):
        return self._queue[0]
    
    def empty(self):
        return not len(self._queue)


# In[ ]:


**Question 4**

You are given a stack **St**. You have to reverse the stack using recursion.

Input:St = {3,2,1,7,6}
Output:{6,7,1,2,3}


# In[ ]:


from collections import deque
 
 
# Recursive function to insert an item at the bottom of a given stack
def insertAtBottom(s, item):
 
    # base case: if the stack is empty, insert the given item at the bottom
    if not s:
        s.append(item)
        return
 
    # Pop all items from the stack and hold them in the call stack
    top = s.pop()
    insertAtBottom(s, item)
 
    # After the recursion unfolds, push each item in the call stack
    # at the top of the stack
    s.append(top)
 
 
# Recursive function to reverse a given stack
def reverseStack(s):
 
    # base case: stack is empty
    if not s:
        return
 
    # Pop all items from the stack and hold them in the call stack
    item = s.pop()
    reverseStack(s)
 
    # After the recursion unfolds, insert each item in the call stack
    # at the bottom of the stack
    insertAtBottom(s, item)
 
 
if __name__ == '__main__':
 
    s = deque(range(1, 6))
    print('Original stack is', s)
    reverseStack(s)
    print('Reversed stack is', s)


# In[ ]:


**Question 5**

You are given a string **S**, the task is to reverse the string using stack.

Example 1:
    
    Input: S="GeeksforGeeks"
Output: skeeGrofskeeG


# In[ ]:


def createStack():
    stack = []
    return stack
 
# Function to determine the size of the stack
 
 
def size(stack):
    return len(stack)
 
# Stack is empty if the size is 0
 
 
def isEmpty(stack):
    if size(stack) == 0:
        return true
 
# Function to add an item to stack .
# It increases size by 1
 
 
def push(stack, item):
    stack.append(item)
 
# Function to remove an item from stack.
# It decreases size by 1
 
 
def pop(stack):
    if isEmpty(stack):
        return
    return stack.pop()
 
# A stack based function to reverse a string
 
 
def reverse(string):
    n = len(string)
 
    # Create a empty stack
    stack = createStack()
 
    # Push all characters of string to stack
    for i in range(0, n, 1):
        push(stack, string[i])
 
    # Making the string empty since all
    # characters are saved in stack
    string = ""
 
    # Pop all characters of string and
    # put them back to string
    for i in range(0, n, 1):
        string += pop(stack)
 
    return string
 
 
# Driver program to test above functions
string = "GeeksQuiz"
string = reverse(string)
print("Reversed string is " + string)


# In[ ]:


**Question 6**

Given string **S** representing a postfix expression, the task is to evaluate the expression and find the final value. Operators will only include the basic arithmetic operators like ***, /, + and -**.

Input: S = "231*+9-"
Output: -4
Explanation:
After solving the given expression,
we have -4 as result.


# In[ ]:


class Evaluate:
 
    # Constructor to initialize the class variables
    def __init__(self, capacity):
        self.top = -1
        self.capacity = capacity
         
        # This array is used a stack
        self.array = []
 
    # Check if the stack is empty
    def isEmpty(self):
        return True if self.top == -1 else False
 
    # Return the value of the top of the stack
    def peek(self):
        return self.array[-1]
 
    # Pop the element from the stack
    def pop(self):
        if not self.isEmpty():
            self.top -= 1
            return self.array.pop()
        else:
            return "$"
 
    # Push the element to the stack
    def push(self, op):
        self.top += 1
        self.array.append(op)
 
    # The main function that converts given infix expression
    # to postfix expression
    def evaluatePostfix(self, exp):
 
        # Iterate over the expression for conversion
        for i in exp:
 
            # If the scanned character is an operand
            # (number here) push it to the stack
            if i.isdigit():
                self.push(i)
 
            # If the scanned character is an operator,
            # pop two elements from stack and apply it.
            else:
                val1 = self.pop()
                val2 = self.pop()
                self.push(str(eval(val2 + i + val1)))
 
        return int(self.pop())
 
 
 
# Driver code
if __name__ == '__main__':
    exp = "231*+9-"
    obj = Evaluate(len(exp))
     
    # Function call
    print("postfix evaluation: %d" % (obj.evaluatePostfix(exp)))


# In[ ]:


**Question 7**

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:

- `MinStack()` initializes the stack object.
- `void push(int val)` pushes the element `val` onto the stack.
- `void pop()` removes the element on the top of the stack.
- `int top()` gets the top element of the stack.
- `int getMin()` retrieves the minimum element in the stack.

You must implement a solution with `O(1)` time complexity for each function.

**Example 1:**

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2


# In[ ]:


class MinStack:
    def __init__(self):
        self.stack = []  # initialize main stack
        self.min_stack = []  # initialize minimum value stack

    def push(self, val: int) -> None:
        self.stack.append(val)  # push value onto main stack
        if not self.min_stack or val <= self.min_stack[-1]:  # if minimum stack is empty or the value is smaller or equal to current minimum
            self.min_stack.append(val)  # push value onto minimum stack

    def pop(self) -> None:
        if self.stack:  # check if main stack is not empty
            if self.stack[-1] == self.min_stack[-1]:  # if the element to pop is the current minimum
                self.min_stack.pop()  # pop from minimum stack
            self.stack.pop()  # always pop from main stack

    def top(self) -> int:
        if self.stack:  # check if main stack is not empty
            return self.stack[-1]  # return the top element

    def getMin(self) -> int:
        if self.min_stack:  # check if minimum stack is not empty
            return self.min_stack[-1]  # return the current minimum value


# In[ ]:


**Question 8**

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.


# In[ ]:


class Solution:
    def sumBackets(self, height: list[int], left, right):

        minHeightLeft = height[left]
        total = 0
        leftBacket = 0
        locationMinLeft = left

        while left < right:
            
            if height[left] < minHeightLeft:
                leftBacket += minHeightLeft - height[left]                
            else:
                minHeightLeft = height[left]
                total +=  leftBacket
                leftBacket = 0
                locationMinLeft = left            
            left += 1
            
        if minHeightLeft <= height[right]:
             return total + leftBacket, right
        else :      
            return total, locationMinLeft

    def sumBacketsReverce(self, height: list[int], left, right):

        minHeightRight = height[right]
        total = 0
        rightBacket = 0
        locationMinRight = right

        while left < right:
            
            if height[right] < minHeightRight:
                rightBacket += minHeightRight - height[right]                
            else :
                minHeightRight = height[right]
                total +=  rightBacket
                rightBacket = 0
                locationMinRight = right            
            right -= 1


        if minHeightRight <= height[left]:
            return total + rightBacket, left
        else :
            return total, locationMinRight
    
    def trap(self, height: List[int]) -> int:                      
        right = len(height)-1
        left =0
        totalSum =0


        while left < right-1:            
            if( height[left]< height[right]):
                total, left = self.sumBackets(height, left, right)    
            else:
                total, right = self.sumBacketsReverce(height, left, right)        
                
            totalSum += total       
             
        return totalSum

