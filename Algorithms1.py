# coding:utf-8
from typing import List, Dict
import math
import sys
import collections
import copy


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Node:
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def kmp_get_next(s):
    n = len(s)
    i, k = 0, -1
    next = [0 for _ in range(n)]
    next[0] = -1

    while i < n - 1:
        if k == -1 or s[i] == s[k]:
            i += 1
            k += 1
            if s[i] != s[k]:
                next[i] = k
            else:
                next[i] = next[k]
        else:
            k = next[k]

    return next


class HeapSort:

    def sort(self, arr):
        n = len(arr)
        # 构建大顶端 从后往前 第一个非叶子节点开始 n//2-1 是最后一个非叶子节点
        for i in range(n//2-1, -1, -1):
            self.adjustHeap(arr, i, n)

        # 将根节点(最大值)与最后一个元素交换位置，然后将前面的数据再次构建为一个大顶端树
        for i in range(n-1, -1, -1):
            temp = arr[0]
            arr[0] = arr[i]
            arr[i] = temp

            self.adjustHeap(arr, 0, i)

    def adjustHeap(self, arr, i, length):
        temp = arr[i]
        k = i * 2 + 1   # i * 2 + 1是左节点
        while k < length:
            # 如果左子结点小于右子结点，k指向右子结点
            if k+1 < length and arr[k] < arr[k+1]:
                k += 1
            # 子节点大于父节点 将子节点赋值给父节点
            if arr[k] > temp:
                arr[i] = arr[k]
                i = k
            else:
                break
            k = k * 2 + 1
        arr[i] = temp   # 将temp值放到最终的位置


# LeetCode-1 twoSum
def twoSum(nums: List[int], target: int) -> List[int]:
    dic: Dict[int, int] = {}
    for index, value in enumerate(nums):
        key = target - value
        if key in dic.keys():
            return [dic[key], index]
        dic[value] = index
    return None


# LeetCode-2 Add Two Numbers
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    p1, p2 = l1, l2
    result = ListNode(-1)
    last_node = result
    carry = 0
    while p1 or p2:
        a = p1.val if p1 else 0
        b = p2.val if p2 else 0
        val = a + b + carry
        carry = val // 10

        node = ListNode(val % 10)
        last_node.next = node
        last_node = node

        p1 = p1.next if p1 and p1.next else None
        p2 = p2.next if p2 and p2.next else None

    if carry > 0:
        last_node.next = ListNode(carry)
    return result.next


# LeetCode-3 最长的子字符串，不重复字符
def lengthOfLongestSubstring(s: str) -> int:
    max_len = 0
    start = 0
    char_map: Dict[str, int] = {}
    for i, ch in enumerate(s):
        if ch not in char_map or char_map[ch] < start:
            # 在当前区间未重复
            max_len = max(i - start + 1, max_len)
        else:
            start = char_map[ch] + 1
        char_map[ch] = i

    return max_len


# LeetCode-4 两个有序数组找中位数
def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    """
    根据中间值的特性，左边值小于右边，然后用二分的方式去找到一个index满足上诉条件
    因为有序，所以确定一个index，两个子数组分别有前几个元素在中间值左边是固定的
    """
    m, n = len(nums1), len(nums2)
    if m > n:
        m, n, nums1, nums2 = n, m, nums2, nums1
    imin, imax, half_len = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2
        j = half_len - i
        if i < m and nums2[j - 1] > nums1[i]:
            # 选值太小
            imin = i + 1
        elif i > 0 and nums1[i - 1] > nums2[j]:
            # 选值太大
            imax = i - 1
        else:
            # 选值正好
            if i == 0:
                max_of_left = nums2[j - 1]
            elif j == 0:
                max_of_left = nums1[i - 1]
            else:
                max_of_left = max(nums1[i - 1], nums2[j - 1])

            # 总长度是奇数
            if (m + n) % 2 == 1:
                return max_of_left

            if i == m:
                min_of_right = nums2[j]
            elif j == n:
                min_of_right = nums1[i]
            else:
                min_of_right = min(nums1[i], nums2[j])

            return (max_of_left + min_of_right) / 2


# LeetCode-5 最长的回文子串
def longestPalindrome(s: str) -> str:
    """分别以每个字符为回文中心点来进行判断"""
    i, j, n = 0, 0, len(s)
    max_start, max_end = 0, 0
    while i < n:
        # 确定中心点  为避免连续相同的情况 所以向右搜索相同的字符
        left = right = i
        while right + 1 < n and s[right + 1] == s[right]:
            right += 1

        # 直接跳过重复字符串 避免无用搜索
        i = right + 1

        # 然后以left right为边界搜索
        while left > 0 and right < n - 1 and s[left - 1] == s[right + 1]:
            left -= 1
            right += 1

        if (right - left + 1) > (max_end - max_start):
            max_start, max_end = left, right

    return s[max_start:max_end + 1]


# LeetCode-6 ZigZag Conversion
def convert(s: str, numRows: int) -> str:
    if numRows == 1:
        return s
    rows: List[str] = [""] * numRows
    i = 0
    go = 1
    for index, val in enumerate(s):
        rows[i] += val

        if i == numRows - 1:
            go = -1
        elif i == 0:
            go = 1
        i += go

    result = ""
    for m in rows:
        result += m
    return result


# LeetCode-7 Reverse Integer
def reverse(x: int) -> int:
    symbol = 1
    result = 0
    if x < 0:
        symbol = -1
        x = x * symbol
    while x != 0:
        val = x % 10
        result = result * 10 + val
        x = x // 10
    result = result * symbol
    if result < (-2) ** 31 or result > 2 ** 31 - 1:
        return 0

    return result


# LeetCode-8 String to Integer
def myAtoi(s: str) -> int:
    result = 0
    symbol = 1
    s = s.lstrip()
    for i, c in enumerate(s):
        if c == "+" and i == 0:
            symbol = 1
        elif c == "-" and i == 0:
            symbol = -1
        elif '\u0030' <= c <= '\u0039':
            result = result * 10 + int(c)
        else:
            break
    result = result * symbol
    if result < (-2) ** 31:
        return (-2) ** 31
    elif result > 2 ** 31 - 1:
        return 2 ** 31 - 1
    return result


# LeetCode-9 Palindrome Number
def isPalindrome(x: int) -> bool:
    if x < 0:  # 负数都不是回文
        return False
    result = 0
    temp = x
    while temp != 0:
        val = temp % 10
        result = result * 10 + val
        temp = temp // 10
    if result == x:
        return True
    else:
        return False


# LeetCode-10 Regular Expression Matching 字符串正则匹配
def isMatch(s: str, p: str) -> bool:
    if not p:
        return not s

    first_match = bool(s) and p[0] in {s[0], '.'}
    if len(p) >= 2 and p[1] == '*':
        return isMatch(s, p[2:]) or (first_match and isMatch(s[1:], p))
    return first_match and isMatch(s[1:], p[1:])


# 动态规划
def isMatch_dp(text: str, pattern: str) -> bool:
    """动态规划 拆分成子问题 text[i:]是否和pattern[j:]匹配 """
    dp = [[False] * (len(pattern) + 1) for _ in range(len(text) + 1)]

    dp[-1][-1] = True
    for i in range(len(text), -1, -1):
        for j in range(len(pattern) - 1, -1, -1):
            first_match = i < len(text) and pattern[j] in {text[i], '.'}
            if j + 1 < len(pattern) and pattern[j + 1] == '*':
                dp[i][j] = dp[i][j + 2] or (first_match and dp[i + 1][j])
            else:
                dp[i][j] = first_match and dp[i + 1][j + 1]

    return dp[0][0]


# LeetCode-11  Container With Most Water
def maxArea(height: List[int]) -> int:
    """暴力解法，可以使用双重循环遍历所有组合
    双指针解法，让两个指针分别指向头和尾，由于面积由最小的边决定，移动大的边只会让面积更小，所以每次都让最小的一边进行移动
    """
    max_area, l, r = 0, 0, len(height) - 1
    while l < r:
        max_area = max(max_area, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1

    return max_area


# LeetCode-14 Longest Common Prefix
def longestCommonPrefix(strs):
    if len(strs) == 0:
        return ""
    prefix = strs[0]
    for i in range(1, len(strs)):
        while not strs[i].startswith(prefix):
            prefix = prefix[0:-2]
            if len(prefix) == 0:
                return ""
    return prefix


# LeetCode-15 电话号码的字母组合
def letterCombinations(digits):
    phone = {'2': ['a', 'b', 'c'],
             '3': ['d', 'e', 'f'],
             '4': ['g', 'h', 'i'],
             '5': ['j', 'k', 'l'],
             '6': ['m', 'n', 'o'],
             '7': ['p', 'q', 'r', 's'],
             '8': ['t', 'u', 'v'],
             '9': ['w', 'x', 'y', 'z']}
    result = []

    def helpCombine(current, leftoverDigits):
        if not leftoverDigits:
            result.append(current)
            return
        else:
            for char in phone[leftoverDigits[0]]:
                helpCombine(current + char, leftoverDigits[1:])

    if not digits:
        return []
    else:
        helpCombine("", digits)
        return result


# LeetCode-19 Remove Nth Node From End of List
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    """
    1. 暴力解法，通过一次循环计算出链表总长度，然后在一次循环找到相应位置
    2. 使用双指针，是他们的间隔为n，那么当一个指针指向结尾的时候，另一个指针就指向目标值
    """
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy

    # 此时first与second间隔为n
    for i in range(1, n + 2):
        first = first.next

    while first:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next


# LeetCode-20  有效的括号
def isValid(s: str) -> bool:
    match = {"{": "}", "[": "]", "(": ")"}
    queue = []
    if len(s) % 2 != 0:
        return False

    for ch in s:
        if ch == "{" or ch == "[" or ch == "(":
            queue.append(ch)
        else:
            if not queue or ch != match[queue[-1]]:
                return False
            else:
                queue.pop()
    return not queue


# LeetCode-21. Merge Two Sorted Lists
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    result = ListNode(0)
    cur = result
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next

    while l1:
        cur.next = l1
        l1 = l1.next
        cur = cur.next

    while l2:
        cur.next = l2
        l2 = l2.next
        cur = cur.next

    return result.next


# LeetCode-24. 成对交换节点
def swapPairs(head: ListNode) -> ListNode:
    def swap(left, right):
        if left and right:
            temp = left.val
            left.val = right.val
            right.val = temp
            if right.next and right.next.next:
                swap(right.next, right.next.next)

    if head and head.next:
        swap(head, head.next)
    return head


# LeetCode-25. Reverse Nodes in k-Group
def reverseKGroup(head: ListNode, k: int) -> ListNode:
    start = temp = head
    stack = []
    count, back = 0, 0
    while temp:
        if back == 1:
            temp.val = stack.pop()
            temp = temp.next
            if not stack:
                back = 0
                start = temp
        else:
            stack.append(temp.val)
            count += 1
            if count == k:
                back = 1
                count = 0
                temp = start
            else:
                temp = temp.next

    if count != 0:
        temp = start
        while temp:
            temp.val = stack.pop()
            temp = temp.next

    return head


# LeetCode-25. Remove Duplicates from Sorted Array
def removeDuplicates(nums: List[int]) -> int:
    i, j = 0, 1
    n = len(nums)
    while j < n:
        if nums[i] == nums[j]:
            nums.pop(j)
            n -= 1
        else:
            j += 1
            i += 1

    return n


# LeetCode-32  最长的有效括号
def longestValidParentheses(s: str) -> int:
    max_len = 0
    stack = [-1]
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len


# LeetCode-33  在旋转排序数组中搜索
def search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    mid = left + ((right - left) // 2)
    while left <= right:
        mid = left + ((right - left) // 2)
        if nums[mid] == target:
            return mid

        # 中间值的两边至少有一个子序列是上升序列, 很容易判断目标值在不在这个区间, 从而找到目标区间
        if nums[mid] < nums[right]:
            # 这里说明右边是一个完整的上升序列
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
        else:
            # 这里说明左边是一个完整的上升序列
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
    return -1


# LeetCode-35. Search Insert Position
def searchInsert(nums: List[int], target: int) -> int:
    for i, v in enumerate(nums):
        if target == v or target < v:
            return i
    return len(nums)


# LeetCode-39. Combination Sum
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    result = []

    def combin(candidated, temp, subCombin):
        if temp == 0:
            result.append(subCombin)
            return
        for ix, num in enumerate(candidated):
            if num > temp:
                # if we don't sort the candidates , this must be continue,but it is slow
                break
            combin(candidated[ix:], temp - num, subCombin + [num])
        return

    combin(sorted(candidates), target, [])
    return result


# LeetCode-40. Combination Sum II(每个数字只能使用一次)
def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    result = []

    def combin(candidated, temp, subCombin):
        if temp == 0 and subCombin not in result:
            result.append(subCombin)
            return
        for ix, num in enumerate(candidated):
            if num > temp:
                # if we don't sort the candidates , this must be continue,but it is slow
                break
            combin(candidated[ix + 1:], temp - num, subCombin + [num])
        return

    combin(sorted(candidates), target, [])
    return result


# LeetCode-41. First Missing Positive
def firstMissingPositive(nums: List[int]) -> int:
    N = len(nums)
    for i, num in enumerate(nums):
        if num <= 0 or num >= N + 1 or num == i + 1:
            continue
        while num != i + 1 and nums[num - 1] != num:
            nums[i], nums[num - 1] = nums[num - 1], nums[i]
            num = nums[i]
            if not (1 <= num <= N):
                break
    for i, num in enumerate(nums):
        if num != i + 1:
            return i + 1
    return N + 1


# LeetCode-42. Trapping Rain Water
def trap(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    result = 0
    left_max, right_max = 0, 0
    while left < right:
        if height[left] < height[right]:
            if height[left] < left_max:
                result += (left_max - height[left])
            else:
                left_max = height[left]
            left += 1
        else:
            if height[right] < right_max:
                result += (right_max - height[right])
            else:
                right_max = height[right]
            right -= 1
    return result


# LeetCode-55. Jump Game I
def jump(nums: List[int]) -> int:
    if len(nums) <= 1:
        return True
    end_range = 0
    farthest = 0
    for i, n in enumerate(nums[:-1]):
        if i + n > farthest:
            farthest = i + n
        if farthest >= len(nums) - 1:
            return True
        if i == end_range:
            end_range = farthest
            if end_range == i:
                return False
    return False


# LeetCode-45. Jump Game II
def jump2(nums: List[int]) -> int:
    jump_nb = 0
    end_range = 0
    farthest = 0
    for i, n in enumerate(nums[:-1]):
        if i + n > farthest:
            farthest = i + n
        if farthest >= len(nums) - 1:
            return jump_nb + 1
        if i == end_range:
            jump_nb += 1
            end_range = farthest
    return jump_nb


# LeetCode-46. Permutations
def permute(nums):
    res = []
    if len(nums) == 0:
        return [res]
    for i in range(len(nums)):
        temp = nums[:i] + nums[i + 1:]
        child_permutation = permute(temp)
        for lst in child_permutation:
            res.append([nums[i]] + lst)
    return res


def permuteUnique(nums):
    res = []
    first = []
    if len(nums) == 0:
        return [res]
    for i in range(len(nums)):
        if nums[i] in first:
            continue
        first.append(nums[i])
        temp = nums[:i] + nums[i + 1:]
        child_permutation = permuteUnique(temp)
        for lst in child_permutation:
            res.append([nums[i]] + lst)
    return res


# LeetCode-48. Rotate Image
def rotate(matrix: List[List[int]]) -> None:
    n = len(matrix[0])
    for i in range(n):
        for j in range(i, n - i - 1):  # n-i-1 is the end point, it is important
            temp = matrix[i][j]

            matrix[i][j] = matrix[n - j - 1][i]
            matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1]
            matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1]
            matrix[j][n - i - 1] = temp


# LeetCode-50. Pow(x, n)
def myPow(x: float, n: int) -> float:
    neg = n < 0
    if neg:
        n = -1 * n
    res = 1
    while n:
        if n & 1:
            res *= x
        x = x * x
        n >>= 1

    if neg:
        res = 1 / res
    return res


# LeetCode-51. N-Queens 递归
def solveNQueens(n: int) -> List[List[str]]:
    res = []
    queenPos = [-1] * n

    # 这里代表0 - k-1行都已经摆好 开始摆第k行
    def NQueen(k):
        if n == k:
            s = "." * n
            temp = []
            for i in queenPos:
                temp += [s[:i] + 'Q' + s[i + 1:]]
            res.append(temp)
            return
        # 遍历所有位置 看是否满足条件
        for i in range(n):
            for j in range(k):
                if queenPos[j] == i or abs(queenPos[j] - i) == abs(k - j):
                    break
            else:
                # 循环结束（不包含break） 代表找到了合适的位置
                queenPos[k] = i
                NQueen(k + 1)

    NQueen(0)
    return res


# LeetCode-53  Maximum Subarray
def maxSubArray(nums: List[int]) -> int:
    cur_sum = 0
    max_sum = nums[0]
    for i in nums:
        cur_sum += i
        if cur_sum > max_sum:
            max_sum = cur_sum
        if cur_sum < 0:
            cur_sum = 0
    return max_sum


# LeetCode-56. Merge Intervals
def merge(intervals):
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


# LeetCode-60. Permutation Sequence
def getPermutation(n: int, k: int) -> str:
    """将问题转化为 在子序列strs中第i项是什么
    因为是有序的，所以可以计算出当前序列第一项元素是什么 然后取出第一个元素，剩下的序列进行子问题的递归调用
    """
    def func(strs, i):
        if len(strs) <= 1 or i <= 0:
            return strs
        temp = math.factorial(len(strs) - 1)
        index = i // temp
        first = strs[index]
        strs = strs[:index] + strs[index + 1:]
        return first + func(strs, i % temp)

    nums = [str(i + 1) for i in range(n)]
    return func("".join(nums), k - 1)


# LeetCode-62. Unique Paths
def uniquePaths(m: int, n: int) -> int:
    """典型的递推问题
    由于dp[i][j] 只依赖于左边的元素和上面的元素 所以 dp[i][j] = dp[i][j - 1] + dp[j - 1][j]
    另外由于dp[i][j]只依赖前面的值，不会影响后面的值，所以使用一维数组，动态修改存储当前列走过的可能
    """
    dp = [0 for _ in range(m)]
    dp[0] = 1
    for i in range(n):
        for j in range(1, m):
            dp[j] = dp[j] + dp[j - 1]

    return dp[m - 1]


# LeetCode-63. Unique Paths II
def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
    """中间有障碍 所以需要判断"""
    n = len(obstacleGrid)
    m = len(obstacleGrid[0])
    dp = [0 for _ in range(m)]
    dp[0] = 1 if obstacleGrid[0][0] == 0 else 0
    for i in range(n):
        for j in range(0, m):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
            else:
                if j != 0:
                    dp[j] = dp[j] + dp[j - 1]

    return dp[m - 1]


# LeetCode-64. Minimum Path Sum
def minPathSum(grid: List[List[int]]) -> int:
    n = len(grid)
    m = len(grid[0])
    # 二维数组 代表从 i，j到终点的最短路径长度
    dp = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if i == n - 1 and j == m - 1:
                dp[i][j] = grid[i][j]
            elif i == n - 1:
                dp[i][j] = grid[i][j] + dp[i][j + 1]
            elif j == m - 1:
                dp[i][j] = grid[i][j] + dp[i + 1][j]
            else:
                dp[i][j] = grid[i][j] + min(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def minPathSum2(grid: List[List[int]]) -> int:
    n = len(grid)
    m = len(grid[0])
    total_len = 0
    min_len = sys.maxsize
    # 二维数组 代表i, j有没有走过
    # dp = [[0 for _ in range(m)] for _ in range(n)]

    # 从i，j到终点的最小长度
    def dfs(i, j):
        nonlocal min_len, total_len
        if i == n - 1 and j == m - 1:
            min_len = min(min_len, total_len)
            return
        if total_len > min_len:
            return

        if j != m - 1:
            total_len += grid[i][j + 1]
            dfs(i, j + 1)
            total_len -= grid[i][j + 1]

        if i != n - 1:
            total_len += grid[i + 1][j]
            dfs(i + 1, j)
            total_len -= grid[i + 1][j]

    total_len += grid[0][0]
    dfs(0, 0)
    return min_len


# LeetCode-66. Plus One
def plusOne(digits: List[int]) -> List[int]:
    n = len(digits)
    carry = 1
    for i in range(n - 1, -1, -1):
        val = digits[i] + carry
        if val >= 10:
            digits[i] = val % 10
            carry = 1
        else:
            digits[i] = val
            carry = 0
            break
    if carry == 1:
        digits.insert(0, 1)

    return digits


# LeetCode-67. Add Binary
def addBinary(a: str, b: str) -> str:
    i, j = len(a) - 1, len(b) - 1
    res = 0
    carry = 0
    while i >= 0 or j >= 0:
        val = carry
        if i >= 0:
            val += int(a[i])
        if j >= 0:
            val += int(b[j])
        if val >= 2:
            res += str(val % 2)
            carry = 1
        else:
            res += str(val)
            carry = 0
        i -= 1
        j -= 1

    if carry == 1:
        res += "1"

    return res[::-1]


# LeetCode-70. Climbing Stairs
def climbStairs(n: int) -> int:
    if n == 1:
        return 1
    # dp 代表n个台阶的走法 dp[n] = dp[n - 1] + dp[n - 2]
    dp = [0 for _ in range(n + 1)]
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# LeetCode-83. Remove Duplicates from Sorted List II
def deleteDuplicates2(head: ListNode) -> ListNode:
    if not head:
        return None
    res = cur = ListNode(-1)
    temp = None
    while head:
        if not temp:
            temp = ListNode(head.val)
        else:
            if temp.val != head.val:
                cur.next = temp
                cur = cur.next
                temp = ListNode(head.val)

        head = head.next

    cur.next = temp

    return res.next


# LeetCode-84. Largest Rectangle in Histogram
def largestRectangleArea(heights: List[int]) -> int:
    stack = []
    best = 0
    heights.append(-1)
    for c, h in enumerate(heights):
        c_ = c
        while stack and h < stack[-1][0]:
            h_, c_ = stack.pop()
            best = max(best, h_ * (c - c_))
        if not stack or h > stack[-1][0]:
            stack.append((h, c_))
    return best


# LeetCode-86. Partition List
def partition(head: ListNode, x: int) -> ListNode:
    left = lp = ListNode(-1)
    right = rp = ListNode(-1)
    while head:
        if head.val < x:
            lp.next = head
            lp = lp.next
        else:
            rp.next = head
            rp = rp.next
        head = head.next

    rp.next = None
    lp.next = right.next
    return left.next


def subsetsWithDup(nums):
    res = [[]]
    count = collections.Counter(nums)
    for n in count:
        tmp = []
        for i in range(count[n]):
            tmp += [r + [n]*(i+1) for r in res]
        res += tmp
    return res


# 中序遍历
def inorderTraversal(root: TreeNode) -> List[int]:
    res = []

    def traversal(t: TreeNode):
        if not t:
            return

        traversal(t.left)
        res.append(t.val)
        traversal(t.right)

    traversal(root)
    return res


def generateTrees(n):
    if n == 0:
        return []

    def gen(start, end):
        if start > end:
            return [None]
        if start == end:
            return [None]

        res = []
        for i in range(start, end):
            root = TreeNode(i)
            for l in gen(start, i):
                for r in gen(i + 1, end):
                    root.left, root.right = l, r
                    res.append(copy.deepcopy(root))

        return res

    return gen(1, n + 1)

# 99. Recover Binary Search Tree
def recoverTree(root: TreeNode) -> None:
    if not root:
        return
    prev_node = first_node = last_node = None

    def inorder(t):
        nonlocal prev_node, first_node, last_node
        if not t:
            return
        inorder(t.left)
        if prev_node:
            if prev_node.val > t.val:
                if not first_node:
                    first_node = prev_node
                last_node = t
        prev_node = t
        inorder(t.right)

    inorder(root)
    if first_node and last_node:
        temp = first_node.val
        first_node.val = last_node.val
        last_node.val = temp


# 100. Same Tree
def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    if not p and not q:
        return True
    elif p and q and p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    else:
        return False


# 101. Symmetric Tree
def isSymmetric(root: TreeNode) -> bool:
    if not root:
        return True

    def isMirror(t1, t2):
        if not t1 and not t2:
            return True
        elif not t1 or not t2:
            return False
        else:
            return t1.val == t2.val and isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)

    return isMirror(root.left, root.right)


# 102. Binary Tree Level Order Traversal
def levelOrder(root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    res = []
    queue = [root]
    while queue:
        temp_queue = []
        level = []
        while queue:
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                temp_queue.append(node.left)
            if node.right:
                temp_queue.append(node.right)
        res.append(level)
        queue = temp_queue
    return res


# 103. Binary Tree Zigzag Level Order Traversal
def zigzagLevelOrder(root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    res = []
    queue = [root]
    d = 0  # 0 从左到右 1 从右到左
    while queue:
        temp_queue = []
        level = []
        while queue:
            node = queue.pop(0)
            if d == 0:
                level.append(node.val)
            else:
                level.insert(0, node.val)

            if node.left:
                temp_queue.append(node.left)
            if node.right:
                temp_queue.append(node.right)
        d = 1 if d == 0 else 0
        res.append(level)
        queue = temp_queue
    return res


# 104. Maximum Depth of Binary Tree
def maxDepth(root: TreeNode) -> int:
    if not root:
        return 0
    res = 0
    queue = [root]
    while queue:
        temp_queue = []
        while queue:
            node = queue.pop(0)
            if node.left:
                temp_queue.append(node.left)
            if node.right:
                temp_queue.append(node.right)
        res += 1
        queue = temp_queue
    return res


# 递归获取树高 效率比较低
def maxDepth(root: TreeNode) -> int:
    if not root:
        return 0
    return max(1 + self.maxDepth(root.left), 1 + self.maxDepth(root.right))


# 111. Minimum Depth of Binary Tree
def minDepth(root: TreeNode) -> List[List[int]]:
    if not root:
        return 0
    res = 0
    queue = [root]
    while queue:
        temp_queue = []
        while queue:
            node = queue.pop(0)
            if not node.left and not node.right:
                return res + 1
            if node.left:
                temp_queue.append(node.left)
            if node.right:
                temp_queue.append(node.right)
        res += 1
        queue = temp_queue
    return res


# 105. Construct Binary Tree from Preorder and Inorder Traversal
def buildTree(preorder: List[int], inorder: List[int]) -> TreeNode:
    """
    通过前序遍历，中序遍历构建树
    通过前序遍历可以获取根节点，通过根节点可以将中序遍历分成左右子树 然后就是递归问题
    """
    if not preorder or not inorder:
        return None
    # if len(preorder) == 1:
    #     return TreeNode(preorder[0])
    root = preorder.pop(0)
    index = inorder.index(root)

    rootNode = TreeNode(root)
    rootNode.left = buildTree(preorder, inorder[0:index])
    rootNode.right = buildTree(preorder, inorder[index + 1:])
    return rootNode


# 106. Construct Binary Tree from Inorder and Postorder Traversal
def buildTree2(inorder: List[int], postorder: List[int]) -> TreeNode:
    """
    通过后序遍历，中序遍历构建树
    与上面类似 后序遍历最后一个为根节点
    """
    if not postorder or not inorder:
        return None

    root = postorder.pop()
    index = inorder.index(root)

    rootNode = TreeNode(root)
    rootNode.left = buildTree2(inorder[0:index], postorder[0:index])
    rootNode.right = buildTree2(inorder[index + 1:], postorder[index:])
    return rootNode


# 108. Convert Sorted Array to Binary Search Tree
def sortedArrayToBST(nums: List[int]) -> TreeNode:
    """中序遍历转为平衡树 每次都将数组平分"""
    if not nums:
        return None
    n = len(nums)
    index = n // 2
    rootNode = TreeNode(nums[index])
    rootNode.left = sortedArrayToBST(nums[0:index])
    rootNode.right = sortedArrayToBST(nums[index + 1:])

    return rootNode


def isBalanced(root: TreeNode) -> bool:
    res = True

    def maxDepth(root: TreeNode) -> int:
        nonlocal res
        if not res:
            return -1
        if not root:
            return 0
        left = maxDepth(root.left)
        right = maxDepth(root.right)
        if abs(left - right) > 1:
            res = False
            return -1
        return max(left, right) + 1

    maxDepth(root)
    return res


# 112. Path Sum
def hasPathSum(root: TreeNode, sum: int) -> bool:
    if not root:
        return False
    res = sum - root.val
    if not root.left and not root.right and res == 0:
        return True
    return hasPathSum(root.left, res) or hasPathSum(root.right, res)


# 117. Populating Next Right Pointers in Each Node
def connect(root: 'Node') -> 'Node':
    if not root:
        return None
    queue = [root]
    while queue:
        temp_queue = []
        while queue:
            node = queue.pop(0)
            if queue:
                node.next = queue[0]
            else:
                node.next = None

            if node.left:
                temp_queue.append(node.left)
            if node.right:
                temp_queue.append(node.right)
        queue = temp_queue
    return root


# 118. Pascal's Triangle
def generate_triangle(numRows: int) -> List[List[int]]:
    res = []
    for i in range(numRows):
        if i == 0:
            res.append([1])
        else:
            temp = res[-1]
            temp = [temp[j] + temp[j + 1] for j in range(len(temp) - 1)]
            res.append([1] + temp + [1])
    return res


def generate_triangle2(numRows: int) -> List[int]:
    L = []
    for i in range(numRows + 1):
        if i == 0:
            L = [1]
        else:
            temp = [L[j] + L[j + 1] for j in range(len(L) - 1)]
            L = [1] + temp + [1]
    return L


def maxPathSum(root: TreeNode) -> int:
    if not root:
        return 0
    max_path = root.val

    def maxSum(t: TreeNode):
        nonlocal max_path
        if not t:
            return -10000000

        left = maxSum(t.left)
        right = maxSum(t.right)
        mxSinglePath = max(t.val, t.val + max(left, right))
        max_path = max(left + right + t.val, mxSinglePath, max_path)
        return mxSinglePath

    maxSum(root)
    return max_path


if __name__ == '__main__':
    # print(twoSum([2, 7, 11, 15], 9))
    # print(lengthOfLongestSubstring("aab"))
    # print(findMedianSortedArrays([1, 1], [2, 2]))
    # print(longestPalindrome("bananas"))
    # print(convert("abc", 1))
    # print(reverse(-1003))
    # print(myAtoi("   -42"))
    # print(isMatch_dp("aa", "a*"))
    # print(isValid("(())"))

    l1 = ListNode(3)
    l2 = ListNode(2)
    l3 = ListNode(5)
    l4 = ListNode(4)
    l5 = ListNode(5)
    l1.next = l2
    l2.next = l3
    l3.next = l4
    l4.next = l5

    # print(reverseKGroup(l1, 3))
    # print(longestValidParentheses("()"))

    # print(combinationSum2([10,1,2,7,6,1,5], 8))
    # print(jump([2,3,1,1,4]))
    # print(myPow(2.0000, -2))
    # print(myPow(2.0000, 4))
    # print(solveNQueens(4))

    # print(jump2([3, 2, 1, 0, 4]))
    # print(getPermutation(4, 9))
    # print(uniquePaths(7, 3))
    # print(uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]))
    # print(minPathSum2([[1,3,1],[1,5,1],[4,2,1]]))
    # print(addBinary("11", "1"))
    # print(largestRectangleArea([2,1,5,6,2,3]))

    # print(partition(l1, 3))
    # a = [1, 2, 3, 4]
    # print(a[:])

    # print(subsetsWithDup([1, 2, 2]))

    # print(generateTrees(3))

    # root = TreeNode(1)
    # root.left = TreeNode(3)
    # root.left.right = TreeNode(2)
    #
    # print(recoverTree(root))

    # print(buildTree2([9,3,15,20,7], [9,15,7,20,3]))
    print(kmp_get_next("asdasc"))
