{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a63ae106",
   "metadata": {},
   "source": [
    "# Question 1\n",
    "Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2),..., (an, bn) such that the sum of min(ai, bi) for all i is maximized. Return the maximized sum.\n",
    "\n",
    "**Example 1:**\n",
    "Input: nums = [1,4,3,2]\n",
    "Output: 4\n",
    "\n",
    "**Explanation:** All possible pairings (ignoring the ordering of elements) are:\n",
    "\n",
    "1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3\n",
    "2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3\n",
    "3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4\n",
    "So the maximum possible sum is 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9da4561e",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def arrayPairSum(self, nums):\n",
    "        \"\"\"\n",
    "        :type nums: List[int]\n",
    "        :rtype: int\n",
    "        \"\"\"\n",
    "\n",
    "        # approach: sort list, take smaller ones and sum them up\n",
    "\n",
    "        nums.sort()\n",
    "        return sum(nums[::2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5ad735f",
   "metadata": {},
   "outputs": [],
   "source": [
    "Total time complexity will be O(nlogn + n/2) = O(nlogn)\n",
    "It only needs O(1) extra space."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2650249",
   "metadata": {},
   "source": [
    "# Question 2\n",
    "Alice has n candies, where the ith candy is of type candyType[i]. Alice noticed that she started to gain weight, so she visited a doctor. \n",
    "\n",
    "The doctor advised Alice to only eat n / 2 of the candies she has (n is always even). Alice likes her candies very much, and she wants to eat the maximum number of different types of candies while still following the doctor's advice. \n",
    "\n",
    "Given the integer array candyType of length n, return the maximum number of different types of candies she can eat if she only eats n / 2 of them.\n",
    "\n",
    "Example 1:\n",
    "Input: candyType = [1,1,2,2,3,3]\n",
    "Output: 3\n",
    "\n",
    "Explanation: Alice can only eat 6 / 2 = 3 candies. Since there are only 3 types, she can eat one of each type."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd581fca",
   "metadata": {},
   "outputs": [],
   "source": [
    "Approach 1: Brute Force\n",
    "Intuition and Algorithm\n",
    "\n",
    "One way to find the number of unique candies is to traverse over each element in candyType, checking whether or not we've already seen a candy of this same type. We can do this check by iterating over all elements before the current element. If any of those are of the same type, then this is not a unique candy. We should keep track of the number of unique candies we find."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "557d351b",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def distributeCandies(self, candyType: List[int]) -> int:\n",
    "        # We need to count how many unique candies are in the array.\n",
    "        unique_candies = 0\n",
    "        # For each candy, we're going to check whether or not we've already\n",
    "        # seen a candy identical to it.\n",
    "        for i in range(len(candyType)):\n",
    "            # Check if we've already seen a candy the same as candyType[i].\n",
    "            for j in range(0, i):\n",
    "                # If this candy is the same as previous one, we don't need to \n",
    "                # check further.\n",
    "                if candyType[i] == candyType[j]:\n",
    "                    break\n",
    "            # Confused? An \"else\" after a \"for\" is an awesome Python feature.\n",
    "            # The code in the \"else\" only runs if the \"for\" loop runs without a break.\n",
    "            # In this case, we know that if we didn't \"break\" out of the loop, then \n",
    "            # candyType[i] is unique.\n",
    "            # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops\n",
    "            else:\n",
    "                unique_candies += 1\n",
    "        # The answer is the minimum out of the number of unique candies, and \n",
    "        # half the length of the candyType array.\n",
    "        return min(unique_candies, len(candyType) // 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef13e91b",
   "metadata": {},
   "outputs": [],
   "source": [
    "Approach 2: Sorting\n",
    "    \n",
    "class Solution:\n",
    "    def distributeCandies(self, candyType: List[int]) -> int:\n",
    "        # We start by sorting candyType.\n",
    "        candyType.sort()\n",
    "        # The first candy is always unique.\n",
    "        unique_candies = 1\n",
    "        # For each candy, starting from the *second* candy...\n",
    "        for i in range(1, len(candyType)):\n",
    "            # This candy is unique if it is different to the one\n",
    "            # immediately before it.\n",
    "            if candyType[i] != candyType[i - 1]:\n",
    "                unique_candies += 1\n",
    "            # Optimization: We should terminate the loop if unique_candies\n",
    "            # is now at the maxium she can eat.\n",
    "            if unique_candies == len(candyType) // 2:\n",
    "                break\n",
    "        # Like before, the answer is the minimum out of the number of unique candies, and \n",
    "        # half the length of the candyType array.\n",
    "        return min(unique_candies, len(candyType) // 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e12b432",
   "metadata": {},
   "outputs": [],
   "source": [
    "Approach 3: Using a Hash Set\n",
    "    \n",
    "class Solution:\n",
    "    def distributeCandies(self, candyType: List[int]) -> int:\n",
    "        # Count the number of unique candies by creating a set with\n",
    "        # candyType, and then taking its length.\n",
    "        unique_candies = len(set(candyType))\n",
    "        # And find the answer in the same way as before.\n",
    "        return min(unique_candies, len(candyType) // 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf2a154f",
   "metadata": {},
   "outputs": [],
   "source": [
    "Time complexity : O(N)\n",
    "    \n",
    "Space complexity : O(N)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "797a7251",
   "metadata": {},
   "source": [
    "# Question 3\n",
    "\n",
    "We define a harmonious array as an array where the difference between its maximum value\n",
    "and its minimum value is exactly 1.\n",
    "\n",
    "Given an integer array nums, return the length of its longest harmonious subsequence\n",
    "among all its possible subsequences.\n",
    "\n",
    "A subsequence of an array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.\n",
    "\n",
    "Example 1:\n",
    "Input: nums = [1,3,2,2,5,2,3,7]\n",
    "Output: 5\n",
    "\n",
    "Explanation: The longest harmonious subsequence is [3,2,2,2,3]."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "258fc8c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def findLHS(self, nums: List[int]) -> int:\n",
    "        my_dict = defaultdict(int)\n",
    "        # keep in dict the number of times each number appears:\n",
    "        for num in nums:\n",
    "            my_dict[num]+=1\n",
    "            \n",
    "        max_ = 0\n",
    "        # for each number in dict check if it+its following number is more than previous max:\n",
    "        for num in my_dict.keys():\n",
    "            if my_dict.get(num+1):\n",
    "                max_ = max(max_, my_dict[num] + my_dict.get(num+1))\n",
    "        return max_"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac346368",
   "metadata": {},
   "source": [
    "# Question 4\n",
    "\n",
    "You have a long flowerbed in which some of the plots are planted, and some are not.\n",
    "However, flowers cannot be planted in adjacent plots.\n",
    "Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.\n",
    "\n",
    "Example 1:\n",
    "Input: flowerbed = [1,0,0,0,1], n = 1\n",
    "Output: true"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4645cfa5",
   "metadata": {},
   "outputs": [],
   "source": [
    "Approach for this Problem :\n",
    "    \n",
    "Loop through each position in the flowerbed.\n",
    "If the current position is 0 and the adjacent positions (if any) are also 0, then plant a flower at that position and decrement n.\n",
    "If n becomes 0, return true as all flowers have been planted.\n",
    "If the loop finishes and n is still positive, return false as there are not enough free spaces in the flowerbed to plant all flowers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "326d9b5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:\n",
    "        for i in range(len(flowerbed)):\n",
    "            if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i+1] == 0):\n",
    "                flowerbed[i] = 1\n",
    "                n -= 1\n",
    "        if n > 0:\n",
    "            return False\n",
    "        return True\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f097c58e",
   "metadata": {},
   "outputs": [],
   "source": [
    "Time complexity : O(n), where n is the length of the flowerbed.\n",
    "Space complexity : O(1), no extra space used."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d5c02f2",
   "metadata": {},
   "source": [
    "# Question 5\n",
    "\n",
    "Given an integer array nums, find three numbers whose product is maximum and return the maximum product.\n",
    "\n",
    "Example 1:\n",
    "Input: nums = [1,2,3]\n",
    "Output: 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6951f80",
   "metadata": {},
   "outputs": [],
   "source": [
    "Heap Concept\n",
    "\n",
    "class Solution:\n",
    "    def maximumProduct(self, nums: List[int]) -> int:\n",
    "      nums.sort()\n",
    "      pos=heapq.nlargest(3,nums)\n",
    "      neg=heapq.nsmallest(2,nums)\n",
    "      return max(neg[0]*neg[1]*pos[0],pos[-1]*pos[-2]*pos[-3])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b532f5de",
   "metadata": {},
   "source": [
    "# Question 6\n",
    "\n",
    "Given an array of integers nums which is sorted in ascending order, and an integer target,\n",
    "write a function to search target in nums. If target exists, then return its index. Otherwise,\n",
    "return -1.\n",
    "\n",
    "You must write an algorithm with O(log n) runtime complexity.\n",
    "\n",
    "Input: nums = [-1,0,3,5,9,12], target = 9\n",
    "Output: 4\n",
    "\n",
    "Explanation: 9 exists in nums and its index is 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "498b04c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "Approach\n",
    "Binary Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca5ca2e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def search(self, nums: List[int], target: int) -> int:\n",
    "        low = 0\n",
    "        high = len(nums)-1\n",
    "\n",
    "        while low <= high:\n",
    "            mid = (low + high) // 2 \n",
    "\n",
    "            if nums[mid] == target:\n",
    "                return mid\n",
    "            if target > nums[mid]:\n",
    "                low = mid + 1\n",
    "            else:\n",
    "                high = mid - 1\n",
    "        \n",
    "        return -1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7004f311",
   "metadata": {},
   "outputs": [],
   "source": [
    "Time complexity: O(log(n))\n",
    "Space complexity: O(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42a9c5da",
   "metadata": {},
   "source": [
    "# Question 7\n",
    "\n",
    "An array is monotonic if it is either monotone increasing or monotone decreasing.\n",
    "\n",
    "An array nums is monotone increasing if for all i <= j, nums[i] <= nums[j]. An array nums is\n",
    "monotone decreasing if for all i <= j, nums[i] >= nums[j].\n",
    "\n",
    "Given an integer array nums, return true if the given array is monotonic, or false otherwise.\n",
    "\n",
    "Example 1:\n",
    "Input: nums = [1,2,2,3]\n",
    "Output: true"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7e935e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check if given array is Monotonic\n",
    "def isMonotonic(A):\n",
    "    x, y = [], []\n",
    "    x.extend(A)\n",
    "    y.extend(A)\n",
    "    x.sort()\n",
    "    y.sort(reverse=True)\n",
    "    if(x == A or y == A):\n",
    "        return True\n",
    "    return False\n",
    " \n",
    " \n",
    "# Driver program\n",
    "A = [6, 5, 4, 4]\n",
    " \n",
    "# Print required result\n",
    "print(isMonotonic(A))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "707eeedc",
   "metadata": {},
   "outputs": [],
   "source": [
    "Time Complexity: O(N*logN), where N is the length of the array.\n",
    "Auxiliary space: O(N), extra space is required for lists x and y."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a52547a",
   "metadata": {},
   "source": [
    "# Question 8\n",
    "\n",
    "You are given an integer array nums and an integer k.\n",
    "\n",
    "In one operation, you can choose any index i where 0 <= i < nums.length and change nums[i] to nums[i] + x where x is an integer from the range [-k, k]. You can apply this operation at most once for each index i.\n",
    "\n",
    "The score of nums is the difference between the maximum and minimum elements in nums.\n",
    "\n",
    "Return the minimum score of nums after applying the mentioned operation at most once for each index in it.\n",
    "\n",
    "Example 1:\n",
    "Input: nums = [1], k = 0\n",
    "Output: 0\n",
    "\n",
    "Explanation: The score is max(nums) - min(nums) = 1 - 1 = 0."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a50d610",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def productExceptSelf(self, nums: List[int]) -> List[int]:\n",
    "        left, right = [1] * (l := len(nums)), [1]*l\n",
    "        for i in range(1, l):\n",
    "            left[i] *= left[i-1]*nums[i-1]\n",
    "            right[-i-1] *= right[-i]*nums[-i]\n",
    "        return [lv*rv for lv, rv in zip(left, right)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "77a00791",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Solution:\n",
    "    def productExceptSelf(self, nums: List[int]) -> List[int]:\n",
    "        res = []\n",
    "        \n",
    "        acc = 1\n",
    "        for n in nums:\n",
    "            res.append(acc)\n",
    "            acc *= n\n",
    "\n",
    "        acc = 1\n",
    "        for i in reversed(range(len(nums))):\n",
    "            res[i] *= acc\n",
    "            acc *= nums[i]\n",
    "            \n",
    "        return res\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "122dc4ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "Time: O(n) for two passes over nums\n",
    "Space: O(1)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
