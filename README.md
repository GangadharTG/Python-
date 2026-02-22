# Python Data Structures & Algorithms (DSA) Guide

A comprehensive guide to understanding and implementing Data Structures and Algorithms in Python.

## Table of Contents

- [Introduction](#introduction)
- [Data Structures](#data-structures)
- [Algorithms](#algorithms)
- [Complexity Analysis](#complexity-analysis)
- [Common Patterns](#common-patterns)
- [Resources](#resources)

## Introduction

Data Structures and Algorithms are fundamental concepts in computer science that help us write efficient and scalable code. This guide covers the most important DSA concepts with Python implementations.

### Why DSA Matters
- **Efficiency**: Write faster, more optimized code
- **Problem Solving**: Develop logical thinking and problem-solving skills
- **Interviews**: Essential for technical interviews
- **Scalability**: Build systems that handle large datasets

---

## Data Structures

### 1. **Arrays / Lists**
- Ordered collection of elements
- Fixed-size (arrays) or dynamic-size (lists)
- Time: Access O(1), Insert/Delete O(n)

```python
# Python List
arr = [1, 2, 3, 4, 5]
arr.append(6)           # O(1)
arr.insert(0, 0)        # O(n)
arr.pop()               # O(1)
```

### 2. **Stacks**
- LIFO (Last In, First Out)
- Use for: Undo/Redo, Expression evaluation, DFS

```python
from collections import deque

class Stack:
	def __init__(self):
		self.items = deque()
    
	def push(self, item):
		self.items.append(item)
    
	def pop(self):
		return self.items.pop() if not self.is_empty() else None
    
	def is_empty(self):
		return len(self.items) == 0
```

### 3. **Queues**
- FIFO (First In, First Out)
- Use for: BFS, Task scheduling, Print jobs

```python
from collections import deque

class Queue:
	def __init__(self):
		self.items = deque()
    
	def enqueue(self, item):
		self.items.append(item)
    
	def dequeue(self):
		return self.items.popleft() if not self.is_empty() else None
    
	def is_empty(self):
		return len(self.items) == 0
```

### 4. **Linked Lists**
- Sequential access O(n), efficient insertion/deletion O(1)
- Singly, Doubly, Circular variants

```python
class ListNode:
	def __init__(self, val):
		self.val = val
		self.next = None

class LinkedList:
	def __init__(self):
		self.head = None
    
	def insert(self, val):
		new_node = ListNode(val)
		new_node.next = self.head
		self.head = new_node
    
	def display(self):
		current = self.head
		while current:
			print(current.val, end=" -> ")
			current = current.next
		print("None")
```

### 5. **Trees**
- Hierarchical structure with parent-child relationships
- Binary Trees, BST, AVL, Trie, Heap

```python
class TreeNode:
	def __init__(self, val):
		self.val = val
		self.left = None
		self.right = None

# Binary Search Tree Operations
class BST:
	def __init__(self):
		self.root = None
    
	def insert(self, val):
		if not self.root:
			self.root = TreeNode(val)
		else:
			self._insert_recursive(self.root, val)
    
	def _insert_recursive(self, node, val):
		if val < node.val:
			if node.left is None:
				node.left = TreeNode(val)
			else:
				self._insert_recursive(node.left, val)
		else:
			if node.right is None:
				node.right = TreeNode(val)
			else:
				self._insert_recursive(node.right, val)
```

### 6. **Graphs**
- Vertices (nodes) and edges
- Directed/Undirected, Weighted/Unweighted
- Representations: Adjacency List, Adjacency Matrix

```python
from collections import defaultdict, deque

class Graph:
	def __init__(self):
		self.graph = defaultdict(list)
    
	def add_edge(self, u, v):
		self.graph[u].append(v)
    
	def bfs(self, start):
		visited = set()
		queue = deque([start])
		visited.add(start)
        
		while queue:
			node = queue.popleft()
			print(node, end=" ")
			for neighbor in self.graph[node]:
				if neighbor not in visited:
					visited.add(neighbor)
					queue.append(neighbor)
```

### 7. **Hash Tables / Dictionaries**
- Key-value pairs
- Average O(1) for insert, delete, lookup

```python
# Python Dictionary
hash_table = {}
hash_table['key'] = 'value'     # O(1)
value = hash_table.get('key')   # O(1)
del hash_table['key']           # O(1)
```

### 8. **Heaps**
- Min-Heap / Max-Heap
- Use for: Priority Queues, Sorting (Heap Sort)

```python
import heapq

# Min-Heap
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)

min_val = heapq.heappop(heap)   # 3
```

---

## Algorithms

### Sorting Algorithms

| Algorithm | Time (Best) | Time (Avg) | Time (Worst) | Space |
|-----------|-------------|-----------|-------------|-------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |

```python
# Quick Sort
def quick_sort(arr):
	if len(arr) <= 1:
		return arr
	pivot = arr[len(arr) // 2]
	left = [x for x in arr if x < pivot]
	middle = [x for x in arr if x == pivot]
	right = [x for x in arr if x > pivot]
	return quick_sort(left) + middle + quick_sort(right)

# Merge Sort
def merge_sort(arr):
	if len(arr) <= 1:
		return arr
    
	mid = len(arr) // 2
	left = merge_sort(arr[:mid])
	right = merge_sort(arr[mid:])
    
	return merge(left, right)

def merge(left, right):
	result = []
	i = j = 0
	while i < len(left) and j < len(right):
		if left[i] <= right[j]:
			result.append(left[i])
			i += 1
		else:
			result.append(right[j])
			j += 1
	result.extend(left[i:])
	result.extend(right[j:])
	return result
```

### Searching Algorithms

```python
# Linear Search - O(n)
def linear_search(arr, target):
	for i in range(len(arr)):
		if arr[i] == target:
			return i
	return -1

# Binary Search - O(log n)
def binary_search(arr, target):
	left, right = 0, len(arr) - 1
	while left <= right:
		mid = (left + right) // 2
		if arr[mid] == target:
			return mid
		elif arr[mid] < target:
			left = mid + 1
		else:
			right = mid - 1
	return -1
```

### Traversal Algorithms

```python
# DFS (Depth-First Search) - Stack/Recursion
def dfs(graph, start, visited=None):
	if visited is None:
		visited = set()
	visited.add(start)
	print(start, end=" ")
    
	for neighbor in graph[start]:
		if neighbor not in visited:
			dfs(graph, neighbor, visited)

# BFS (Breadth-First Search) - Queue
from collections import deque

def bfs(graph, start):
	visited = set()
	queue = deque([start])
	visited.add(start)
    
	while queue:
		node = queue.popleft()
		print(node, end=" ")
		for neighbor in graph[node]:
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append(neighbor)
```

### Dynamic Programming

```python
# Fibonacci - Memoization
def fib_memo(n, memo={}):
	if n in memo:
		return memo[n]
	if n <= 1:
		return n
	memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
	return memo[n]

# Fibonacci - Tabulation
def fib_tab(n):
	if n <= 1:
		return n
	dp = [0] * (n + 1)
	dp[1] = 1
	for i in range(2, n + 1):
		dp[i] = dp[i - 1] + dp[i - 2]
	return dp[n]

# Longest Common Subsequence (LCS)
def lcs(text1, text2):
	m, n = len(text1), len(text2)
	dp = [[0] * (n + 1) for _ in range(m + 1)]
    
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if text1[i - 1] == text2[j - 1]:
				dp[i][j] = dp[i - 1][j - 1] + 1
			else:
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
	return dp[m][n]
```

---

## Complexity Analysis

### Time Complexity

Big O notation describes how the algorithm's performance scales with input size:

| Notation | Name | Example |
|----------|------|---------|
| O(1) | Constant | Array index access |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Simple loop |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Nested loops |
| O(n³) | Cubic | Three nested loops |
| O(2ⁿ) | Exponential | All subsets |
| O(n!) | Factorial | All permutations |

### Space Complexity

How much extra memory an algorithm uses:

```python
# O(1) - Constant space
def count_items(n):
	return n + 1

# O(n) - Linear space
def create_list(n):
	return [i for i in range(n)]

# O(n²) - Quadratic space
def create_matrix(n):
	return [[i + j for j in range(n)] for i in range(n)]
```

---

## Common Patterns

### 1. **Two Pointers**
```python
# Reverse a list
def reverse(arr):
	left, right = 0, len(arr) - 1
	while left < right:
		arr[left], arr[right] = arr[right], arr[left]
		left += 1
		right -= 1
```

### 2. **Sliding Window**
```python
# Maximum sum subarray of size k
def max_sum_subarray(arr, k):
	window_sum = sum(arr[:k])
	max_sum = window_sum
    
	for i in range(1, len(arr) - k + 1):
		window_sum = window_sum - arr[i - 1] + arr[i + k - 1]
		max_sum = max(max_sum, window_sum)
    
	return max_sum
```

### 3. **Fast & Slow Pointers**
```python
# Detect cycle in linked list
def has_cycle(head):
	slow = fast = head
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next
		if slow == fast:
			return True
	return False
```

### 4. **Prefix Sum**
```python
# Range sum query
class PrefixSum:
	def __init__(self, arr):
		self.prefix = [0]
		for num in arr:
			self.prefix.append(self.prefix[-1] + num)
    
	def range_sum(self, i, j):
		return self.prefix[j + 1] - self.prefix[i]
```

### 5. **Backtracking**
```python
# Generate all permutations
def permute(nums):
	result = []
    
	def backtrack(path, remaining):
		if not remaining:
			result.append(path)
			return
        
		for i in range(len(remaining)):
			backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:])
    
	backtrack([], nums)
	return result
```

---

## Resources

### Books
- Introduction to Algorithms (CLRS)
- Cracking the Coding Interview
- Algorithm Design Manual

### Online Platforms
- [LeetCode](https://leetcode.com)
- [HackerRank](https://www.hackerrank.com)
- [CodeSignal](https://codesignal.com)
- [GeeksforGeeks](https://www.geeksforgeeks.org)

### Key Takeaways
- Master fundamental data structures
- Understand time and space complexity
- Practice problem-solving regularly
- Learn multiple approaches to same problem
- Code clean and readable solutions

---

**Happy Learning! Keep practicing and solving problems to master DSA!**