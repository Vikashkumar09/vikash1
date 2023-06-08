#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**Question 1**

Given two strings s1 and s2, return *the lowest **ASCII** sum of deleted characters to make two strings equal*.

**Example 1:**

**Input:** s1 = "sea", s2 = "eat"

**Output:** 231

**Explanation:** Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.

Deleting "t" from "eat" adds 116 to the sum.

At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.


# In[ ]:


class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        dp = [[0 for j in range(len(s2)+1)] for i in range(len(s1)+1)]
        for i in range(len(s1)-1,-1,-1):
            for j in range(len(s2)-1,-1,-1):
                if s1[i] == s2[j]:
                    dp[i][j] = ord(s1[i]) + dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j+1])                    
        total = 0
        for c in s1:
            total += ord(c)
        for c in s2:
            total += ord(c)
        return total - dp[0][0]*2


# In[ ]:


**Question 2**

Given a string s containing only three types of characters: '(', ')' and '*', return true *if* s *is **valid***.

The following rules define a **valid** string:

- Any left parenthesis '(' must have a corresponding right parenthesis ')'.
- Any right parenthesis ')' must have a corresponding left parenthesis '('.
- Left parenthesis '(' must go before the corresponding right parenthesis ')'.
- '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".

**Example 1:**

**Input:** s = "()"

**Output:**

true


# In[ ]:


class Solution:
    def checkValidString(self, s: str) -> bool:
        
        # store the indices of '('
        stk = []
        
        # store the indices of '*'
        star = []
        
        
        for idx, char in enumerate(s):
            
            if char == '(':
                stk.append( idx )
                
            elif char == ')':
                
                if stk:
                    stk.pop()
                elif star:
                    star.pop()
                else:
                    return False
            
            else:
                star.append( idx )
        
        
        # cancel ( and * with valid positions, i.e., '(' must be on the left hand side of '*'
        while stk and star:
            if stk[-1] > star[-1]:
                return False
        
            stk.pop()
            star.pop()
        
        
        # Accept when stack is empty, which means all braces are paired
        # Reject, otherwise.
        return len(stk) == 0


# In[ ]:


**Question 3**

Given two strings word1 and word2, return *the minimum number of **steps** required to make* word1 *and* word2 *the same*.

In one **step**, you can delete exactly one character in either string.

**Example 1:**

**Input:** word1 = "sea", word2 = "eat"

**Output:** 2

**Explanation:** You need one step to make "sea" to "ea" and another step to make "eat" to "ea".


# In[ ]:


MAX = 500001
parent = [0] * MAX
Rank = [0] * MAX
 
# Function to find out
# parent of an alphabet
def find(x):
     
    if parent[x] == x:
        return x
    else:
        return find(parent[x])
 
# Function to merge two
# different alphabets
def merge(r1, r2):
 
    # Merge a and b using
    # rank compression
    if(r1 != r2):
        if(Rank[r1] > Rank[r2]):
            parent[r2] = r1
            Rank[r1] += Rank[r2]
 
        else:
            parent[r1] = r2
            Rank[r2] += Rank[r1]
 
# Function to find the minimum
# number of operations required
def minimumOperations(s1, s2):
 
    # Initializing parent to i
    # and rank(size) to 1
    for i in range(1, 26 + 1):
        parent[i] = i
        Rank[i] = 1
 
    # We will store our
    # answerin this list
    ans = []
 
    # Traversing strings
    for i in range(len(s1)):
        if(s1[i] != s2[i]):
 
            # If they have different parents
            if(find(ord(s1[i]) - 96) !=
               find(ord(s2[i]) - 96)):
 
                # Find their respective
                # parents and merge them
                x = find(ord(s1[i]) - 96)
                y = find(ord(s2[i]) - 96)
                merge(x, y)
 
                # Store this in
                # our Answer list
                ans.append([s1[i], s2[i]])
 
    # Number of operations
    print(len(ans))
    for i in ans:
        print(i[0], "->", i[1])
         
# Driver code
if __name__ == '__main__':
 
    # Two strings
    # S1 and S2
    s1 = "abb"
    s2 = "dad"
     
    # Function Call
    minimumOperations(s1, s2)


# In[ ]:


**Question 4**

You need to construct a binary tree from a string consisting of parenthesis and integers.

The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.
You always start to construct the **left** child node of the parent first if it exists.

get_ipython().system('[Screenshot 2023-05-29 010548.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bdbea2d1-34a4-4c4b-a450-ea6db7410c43/Screenshot_2023-05-29_010548.png)')

**Input:** s = "4(2(3)(1))(6(5))"

**Output:** [4,2,6,3,1,5]


# In[ ]:


def tree2str(self, root: Optional[TreeNode]) -> str:
    def solve(curNode):
        if not curNode:
            return ''
        
        subString = str(curNode.val)
		#If curNode has no children
        if not curNode.left and not curNode.right:
            return subString
		
        subString += '(' + solve(curNode.left) + ')'
        if curNode.right:
            subString += '(' + solve(curNode.right) + ')'
        
        return subString
   
    return solve(root)


Analysis:

Time complexity: O(n), where n is the total number of nodes in the tree
Space complexity: O(logn). In the worst case (skewed tree), O(n).


# In[ ]:


**Question 5**

Given an array of characters chars, compress it using the following algorithm:

Begin with an empty string s. For each group of **consecutive repeating characters** in chars:

- If the group's length is 1, append the character to s.
- Otherwise, append the character followed by the group's length.

The compressed string s **should not be returned separately**, but instead, be stored **in the input character array chars**. Note that group lengths that are 10 or longer will be split into multiple characters in chars.

After you are done **modifying the input array,** return *the new length of the array*.

You must write an algorithm that uses only constant extra space.

**Example 1:**

**Input:** chars = ["a","a","b","b","c","c","c"]

**Output:** Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]

**Explanation:**

The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".


# In[ ]:


class Solution:
    def compress(self, chars: List[str]) -> int:
        i=0
        count=1
        while i<len(chars)-1:
            if chars[i+1]==chars[i]:
                chars.pop(i+1)
                count+=1
            elif count>1:
                cc=[*str(count)]
                for j in range(len(cc)):
                    chars.insert(i+j+1,cc[j])
                count=1
                i+=len(cc)+1
            else:
                i+=1
        if count>1:
            chars+=[*str(count)]
        return len(chars)


# In[ ]:


**Question 6**

Given two strings s and p, return *an array of all the start indices of* p*'s anagrams in* s. You may return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Example 1:**

**Input:** s = "cbaebabacd", p = "abc"

**Output:** [0,6]

**Explanation:**

The substring with start index = 0 is "cba", which is an anagram of "abc".

The substring with start index = 6 is "bac", which is an anagram of "abc".


# In[ ]:


def findAnagrams(self, s: str, p: str) -> List[int]:
        hm, res, pL, sL = defaultdict(int), [], len(p), len(s)
        if pL > sL: return []

        # build hashmap
        for ch in p: hm[ch] += 1

        # initial full pass over the window
        for i in range(pL-1):
            if s[i] in hm: hm[s[i]] -= 1
            
        # slide the window with stride 1
        for i in range(-1, sL-pL+1):
            if i > -1 and s[i] in hm:
                hm[s[i]] += 1
            if i+pL < sL and s[i+pL] in hm: 
                hm[s[i+pL]] -= 1
                
            # check whether we encountered an anagram
            if all(v == 0 for v in hm.values()): 
                res.append(i+1)
                
        return res
    
    Time: O(n) - one pass over the p, on pass for s, and for every letter in s we iterate over values in hashmap (maximum 26)
Space: O(1) - hashmap with max 26 keys


# In[ ]:


**Question 7**

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].

The test cases are generated so that the length of the output will never exceed 105.

**Example 1:**

**Input:** s = "3[a]2[bc]"

**Output:** "aaabcbc"


# In[ ]:


class Solution:
    def decodeString(self, s: str) -> str:
        st = []
        num = 0
        res = ''

        for ch in s:
            if ch.isnumeric():
                num = num * 10 + int(ch)
            elif ch == '[':
                st.append(res)
                st.append(num)
                res = ''
                num = 0
            elif ch == ']':
                cnt = st.pop()
                prev = st.pop()
                res = prev + cnt * res
            else:
                res += ch
        return res


# In[ ]:


**Question 8**

Given two strings s and goal, return true *if you can swap two letters in* s *so the result is equal to* goal*, otherwise, return* false*.*

Swapping letters is defined as taking two indices i and j (0-indexed) such that i != j and swapping the characters at s[i] and s[j].

- For example, swapping at indices 0 and 2 in "abcd" results in "cbad".

**Example 1:**

**Input:** s = "ab", goal = "ba"

**Output:** true

**Explanation:** You can swap s[0] = 'a' and s[1] = 'b' to get "ba", which is equal to goal


# In[ ]:


class Solution:
    def buddyStrings(self, A: str, B: str) -> bool:
        # check same length
        if len(A) != len(B): return False
        
        # if strings are equal - check if there is a double to swap
        if A == B:
            return True if len(A) - len(set(A)) >= 1 else False
        
        # count differences between strings
        diff = []
        for i in range(len(A)):
            if A[i] != B[i]:
                diff.append(i)
                if len(diff) > 2: return False
                
        # not exactly two differences
        if len(diff) != 2: return False
        
        # check if can be swapped
        if A[diff[0]] == B[diff[1]] and A[diff[1]] == B[diff[0]]:
            return True
        
        return False

