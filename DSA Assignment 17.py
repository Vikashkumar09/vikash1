#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Question 1**

Given a string `s`, *find the first non-repeating character in it and return its index*. If it does not exist, return `-1`.

**Example 1:**

Input: s = "leetcode"
Output: 0


# In[ ]:


def firstNotRepeatingCharacter(s):
    '''
    input: A string of letters
    Output: the first letter that doesn't have a duplicate 
    
    Idea: We can loop through the string and add each letter 
    a set. Sets cannot have duplicates, so if the length of 
    the set is different that the string we return _. 
    That is the base case. Then we can check if each letter is in the set 
    if it isnt we return that letter.
    '''
    myset = set()
    for letter in s:
        myset.add(letter)
        if  not in myset:
            return letter
        return '_'
        # This solution does not work because we are looking for letters with no duplicates 
        
  def firstNotRepeatingCharacter(s):
    for c in s:
        if s.index(c) == s.rindex(c):
            return c
    return '_'
    # This solution returns the letter if the last occurence of it is the same 
    # index as the first occurence of it.


# In[ ]:


2. Given a **circular integer array** `nums` of length `n`, return *the maximum possible sum of a non-empty **subarray** of* `nums`.

A **circular array** means the end of the array connects to the beginning of the array. Formally, the next element of `nums[i]` is `nums[(i + 1) % n]` and the previous element of `nums[i]` is `nums[(i - 1 + n) % n]`.

A **subarray** may only include each element of the fixed buffer `nums` at most once. Formally, for a subarray `nums[i], nums[i + 1], ..., nums[j]`, there does not exist `i <= k1`, `k2 <= j` with `k1 % n == k2 % n`.

Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.


# In[ ]:


def maxCircularSum(a, n):
     
    # Corner Case
    if (n == 1):
        return a[0]
 
    # Initialize sum variable which
    # store total sum of the array.
    sum = 0
    for i in range(n):
        sum += a[i]
 
    # Initialize every variable
    # with first value of array.
    curr_max = a[0]
    max_so_far = a[0]
    curr_min = a[0]
    min_so_far = a[0]
 
    # Concept of Kadane's Algorithm
    for i in range(1, n):
       
        # Kadane's Algorithm to find Maximum subarray sum.
        curr_max = max(curr_max + a[i], a[i])
        max_so_far = max(max_so_far, curr_max)
 
        # Kadane's Algorithm to find Minimum subarray sum.
        curr_min = min(curr_min + a[i], a[i])
        min_so_far = min(min_so_far, curr_min)
    if (min_so_far == sum):
        return max_so_far
 
    # returning the maximum value
    return max(max_so_far, sum - min_so_far)
 
# Driver code
a = [11, 10, -20, 5, -3, -5, 8, -13, 10]
n = len(a)
print("Maximum circular sum is", maxCircularSum(a, n))


# In[ ]:


**Question 3**

The school cafeteria offers circular and square sandwiches at lunch break, referred to by numbers `0` and `1` respectively. All students stand in a queue. Each student either prefers square or circular sandwiches.

The number of sandwiches in the cafeteria is equal to the number of students. The sandwiches are placed in a **stack**. At each step:

- If the student at the front of the queue **prefers** the sandwich on the top of the stack, they will **take it** and leave the queue.
- Otherwise, they will **leave it** and go to the queue's end.

This continues until none of the queue students want to take the top sandwich and are thus unable to eat.

You are given two integer arrays `students` and `sandwiches` where `sandwiches[i]` is the type of the `ith` sandwich in the stack (`i = 0` is the top of the stack) and `students[j]` is the preference of the `jth` student in the initial queue (`j = 0` is the front of the queue). Return *the number of students that are unable to eat.*


# In[ ]:


class Solution:
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        for i in sandwiches:
            if i in students:
                students.remove(i)
            else:
                break
        return len(students)


# In[ ]:


**Question 4**

You have a `RecentCounter` class which counts the number of recent requests within a certain time frame.

Implement the `RecentCounter` class:

- `RecentCounter()` Initializes the counter with zero recent requests.
- `int ping(int t)` Adds a new request at time `t`, where `t` represents some time in milliseconds, and returns the number of requests that has happened in the past `3000` milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range `[t - 3000, t]`.

It is **guaranteed** that every call to `ping` uses a strictly larger value of `t` than the previous call.


# In[ ]:


from collections import deque
class RecentCounter:

    def __init__(self):
        self.q = deque()
        
    def ping(self, t: int) -> int:
        self.q.append(t)
        
        while t - self.q[0] > 3000:
            self.q.popleft()
            
        return len(self.q)
        
# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)


# In[ ]:


**Question 5**

There are `n` friends that are playing a game. The friends are sitting in a circle and are numbered from `1` to `n` in **clockwise order**. More formally, moving clockwise from the `ith` friend brings you to the `(i+1)th` friend for `1 <= i < n`, and moving clockwise from the `nth` friend brings you to the `1st` friend.

The rules of the game are as follows:

1. **Start** at the `1st` friend.
2. Count the next `k` friends in the clockwise direction **including** the friend you started at. The counting wraps around the circle and may count some friends more than once.
3. The last friend you counted leaves the circle and loses the game.
4. If there is still more than one friend in the circle, go back to step `2` **starting** from the friend **immediately clockwise** of the friend who just lost and repeat.
5. Else, the last friend in the circle wins the game.

Given the number of friends, `n`, and an integer `k`, return *the winner of the game*.


# In[ ]:


class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        return self.helper(n,k)+1

    def helper(self, n:int, k:int)-> int:
        if(n==1):
            return 0
        prevWinner = self.helper(n-1, k)
        return (prevWinner + k) % n


# In[ ]:


**Question 6**

You are given an integer array `deck`. There is a deck of cards where every card has a unique integer. The integer on the `ith` card is `deck[i]`.

You can order the deck in any order you want. Initially, all the cards start face down (unrevealed) in one deck.

You will do the following steps repeatedly until all cards are revealed:

1. Take the top card of the deck, reveal it, and take it out of the deck.
2. If there are still cards in the deck then put the next top card of the deck at the bottom of the deck.
3. If there are still unrevealed cards, go back to step 1. Otherwise, stop.

Return *an ordering of the deck that would reveal the cards in increasing order*.

**Note** that the first entry in the answer is considered to be the top of the deck.


# In[ ]:


class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        queue = deque()
        deck.sort(reverse=True)
        for card in deck:
            if queue:
                queue.appendleft(queue.pop())
            queue.appendleft(card)
        return queue


# In[ ]:


**Question 7**

Design a queue that supports `push` and `pop` operations in the front, middle, and back.

Implement the `FrontMiddleBack` class:

- `FrontMiddleBack()` Initializes the queue.
- `void pushFront(int val)` Adds `val` to the **front** of the queue.
- `void pushMiddle(int val)` Adds `val` to the **middle** of the queue.
- `void pushBack(int val)` Adds `val` to the **back** of the queue.
- `int popFront()` Removes the **front** element of the queue and returns it. If the queue is empty, return `1`.
- `int popMiddle()` Removes the **middle** element of the queue and returns it. If the queue is empty, return `1`.
- `int popBack()` Removes the **back** element of the queue and returns it. If the queue is empty, return `1`.

**Notice** that when there are **two** middle position choices, the operation is performed on the **frontmost** middle position choice. For example:

- Pushing `6` into the middle of `[1, 2, 3, 4, 5]` results in `[1, 2, 6, 3, 4, 5]`.
- Popping the middle from `[1, 2, 3, 4, 5, 6]` returns `3` and results in `[1, 2, 4, 5, 6]`.


# In[ ]:


class FrontMiddleBackQueue:

    def __init__(self):
        self.q=[]
        

    def pushFront(self, val: int) -> None:
        self.q.insert(0,val)
        

    def pushMiddle(self, val: int) -> None:
        n=len(self.q)
        if n%2==0:
            n=(n/2)
        else:
            n=int(n//2)
        i=0
        self.q.insert(int(n),val)
        print(self.q)
        

    def pushBack(self, val: int) -> None:
        self.q.append(val)

        

    def popFront(self) -> int:
        if len(self.q)==0:
            return -1
        l=self.q[0]
        del self.q[0]
        return l
        

    def popMiddle(self) -> int:
        if len(self.q)==0:
            return -1
        n=len(self.q)
        if n%2==0:
            n=(n/2)-1
        else:
            n=n//2
        p=self.q[int(n)]
        del self.q[int(n)]
        return p
        

    def popBack(self) -> int:
        if len(self.q)==0:
            return -1
        p=self.q[len(self.q)-1]
        del self.q[len(self.q)-1]
        return p

        


# Your FrontMiddleBackQueue object will be instantiated and called as such:
# obj = FrontMiddleBackQueue()
# obj.pushFront(val)
# obj.pushMiddle(val)
# obj.pushBack(val)
# param_4 = obj.popFront()
# param_5 = obj.popMiddle()
# param_6 = obj.popBack()


# In[ ]:


**Question 8**

For a stream of integers, implement a data structure that checks if the last `k` integers parsed in the stream are **equal** to `value`.

Implement the **DataStream** class:

- `DataStream(int value, int k)` Initializes the object with an empty integer stream and the two integers `value` and `k`.
- `boolean consec(int num)` Adds `num` to the stream of integers. Returns `true` if the last `k` integers are equal to `value`, and `false` otherwise. If there are less than `k` integers, the condition does not hold true, so returns `false`.


# In[ ]:


class newNode:
    def __init__(self, data):
        self.data = data
        self.left = self.right = None
        self.leftSize = 0
 
# Inserting a new Node.
def insert(root, data):
    if root is None:
        return newNode(data)
 
    # Updating size of left subtree.
    if data <= root.data:
        root.left = insert(root.left, data)
        root.leftSize += 1
    else:
        root.right = insert(root.right, data)
    return root
 
# Function to get Rank of a Node x.
def getRank(root, x):
     
    # Step 1.
    if root.data == x:
        return root.leftSize
 
    # Step 2.
    if x < root.data:
        if root.left is None:
            return -1
        else:
            return getRank(root.left, x)
 
    # Step 3.
    else:
        if root.right is None:
            return -1
        else:
            rightSize = getRank(root.right, x)
            if rightSize == -1:
                # x not found in right sub tree, i.e. not found in stream
                return -1
            else:
                return root.leftSize + 1 + rightSize
 
# Driver code
if __name__ == '__main__':
    arr = [5, 1, 4, 4, 5, 9, 7, 13, 3]
    n = len(arr)
    x = 4
 
    root = None
    for i in range(n):
        root = insert(root, arr[i])
 
    print("Rank of", x, "in stream is:",
                       getRank(root, x))
    x = 13
    print("Rank of", x, "in stream is:",
                       getRank(root, x))
    x = 8
    print("Rank of", x, "in stream is:",
                       getRank(root, x))

