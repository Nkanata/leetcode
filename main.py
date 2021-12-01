# This is a sample Python script.
import collections
import itertools
from math import floor
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from paranoid import List

from FirstLast import Solution
from SuperPalindrome import SuperPalindrome

SuperPalindrome


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def searchRange(nums, target):
    low = 0
    high = len(nums) - 1
    while low <= high:
        mid = floor((low + high) / 2)
        # print(mid)
        guess = nums[mid]
        # print(guess)
        if guess == target:
            print(mid)
        # return [mid]
        if guess > target:
            high = mid - 1
        else:
            low = mid + 1
    return print([-1, -1])


# def powerfulIntegers(x, y, bound):
def runningSum(nums):
    num2 = []
    for x in range(len(nums)):
        if x == 0:
            num2.append(nums[0])
            print("num2", num2)
        else:
            y = x + 1
            nu = nums[0:y]
            print(x, nu)
            num2.append(sum(nu))
            print("sum", sum(nu))
    print(num2)
    return num2
    # print(x)


def runningSum2(self, nums):
    for i in range(1, len(nums)):
        nums[i] = nums[i - 1] + nums[i]
    return nums


"""
:type nums: List[int]
:rtype: List[int]
"""


def runningSum3(self, nums):
    out = []
    sum = 0
    for x in nums:
        sum = sum + x
        out.append(sum)

    return out


def checkPossibility(nums):
    x = sorted(nums, reverse=True)
    y = sorted(nums)
    if x == nums:
        print(x)
        print(False)
        return False
    elif nums[0] == y[-1]:
        print(nums[0])
        print(y[-1])
        return True
    # elif x != y:
    #   print(y)
    #  return False
    elif nums[-1] != nums.index(max(nums)):
        print(True)
        return True
    else:
        a = nums.index(max(nums))
        print(a)
        print(True)
        return True


def minDistance(word1, word2):
    x = len(word1)
    y = len(word2)
    same = ''
    if x == y:
        for i in range(x):
            if word1[i] in word2:
                same += word1[i]
                print(same)
        a = word1.replace(same, '')
        b = word2.replace(same, '')
        print(len(a), len(b))
        print(len(a) + len(b))
        return len(a) + len(b)
    elif x > y:
        if word2 in word1:
            same += word2
            print(same)
            print(x, y)
            a = word1.replace(same, '')
            b = word2.replace(same, '')
            print(len(a), len(b))
            print(len(a) + len(b))
            return len(a) + len(b)
        else:
            for i in range(y):
                if word2[i] in word1:
                    same += word2[i]
            a = word1.replace(same, '')
            b = word2.replace(same, '')
            print(a, b)

            print(len(a), len(b))
            print(len(a) + len(b))
            return len(a) + len(b)
    else:
        if word1 in word2:
            same += word1
            print(same)
        print(x, y)
        a = word1.replace(same, '')
        b = word2.replace(same, '')
        print(len(a), len(b))
        print(len(a) + len(b))
        return len(a) + len(b)


def printCombinations(input, index, output, outlength):
    if len(input) == index:
        output[outlength] = '\0'
        print('\0')


def ambiguousCoordinates(S):
    def make(frag):
        N = len(frag)
        for d in range(1, N + 1):
            left = frag[:d]
            right = frag[d:]
            if ((not left.startswith('0') or left == '0')
                    and (not right.endswith('0'))):
                yield left + ('.' if d != N else '') + right

    S = S[1:-1]
    return ["({}, {})".format(*cand)
            for i in range(1, len(S))
            for cand in itertools.product(make(S[:i]),
                                          make(S[i:]))]


def longestStrChain(words):
    print(words)
    l = len(words)
    for i in range(1, l):
        temp = words[i]
        j = i - 1
        while j >= 0 and len(temp) < len(words[j]):
            words[j + 1] = words[j]
            j -= 1

        words[j + 1] = temp

    print(words)


def minMoves2(nums):
    std = 2
    moves = 0
    if len(nums) == 1:
        return 0
    for i in range(len(nums)):
        if nums[i] > std:
            dif = nums[i] - std
            moves = moves + dif
        elif nums[i] < std:
            dif = std - nums[i]
            moves = moves + dif
    # print(moves)
    return moves


s = [1, 10, 2, 9]


def minmoves2(nums):
    mn = min(nums)
    nums = [x - mn for x in nums]
    print(nums)
    nums.sort()
    N = len(nums)

    left = 0
    previous = 0
    right = sum(nums)
    print("right", right)
    best = 10 ** 10
    print("range", range(N))
    n = [-2, -5, -7, 2, 1]
    print(min(n))

    for index in range(N):
        delta = nums[index] - previous
        print(index)

        left_count = index
        left += left_count * delta
        right_count = N - index - 1
        right -= (right_count + 1) * delta

        print(f"{delta=} {left_count=} {left=} {right_count=} {right=}")
        total = left + right
        best = min(best, total)
        previous = nums[index]
    return best


def levelorder(root):
    arr = []
    count = 0
    previous = []
    for i in range(len(root)):
        if i == 0:
            newArr = [root[i]]
            arr.append(newArr)
        if (count < 2) & (i != 0):
            previous.append(root[i])
            count += 1
        if count == 2:
            count = 0
            arr.append(previous)
            previous = []
    return arr


def tolowercase(s):
    n = ""
    for i in s:
        asci = ord(i)
        if 65 <= asci <= 91:
            print(i, asci)
            j = asci - 65
            j = 97 + j
            print(chr(j))
            n += chr(j)
        else:
            n += i
    return n


def evalRPN(tokens: list[str]) -> int:
    stack = []
    for token in tokens:
        try:
            stack.append(int(token))
        except:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))
            else:
                print("not supposed to happen")

    return stack[-1]


def maxProduct(self, words: list[str]) -> int:
    N = len(words)
    mask = []

    def calculateMask(word):
        mask = 0

        for c in word:
            offset = ord(c) - ord('a')

            mask |= (1 << offset)

        return mask

    for word in words:
        mask.append(calculateMask(word))

    mx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (mask[i] & mask[j]) == 0:
                mx = max(mx, len(words[i]) * len(words[j]))
    return mx


def customSortString(order: str, str1: str) -> str:
    print(order, str1)
    newStr = order
    for c in str1:
        if c not in order:
            print(c)
            newStr += c
    print(newStr)
    return newStr


def pushDominoes(dominoes: str) -> str:
    N = len(dominoes)
    # print(N)
    ans = ''
    for i in range(N):
        if dominoes[i] == '.':
            if i - 1 >= 0 and dominoes[i - 1] == 'R':
                if i + 1 < N and dominoes[i + 1] != 'L':
                    ans += 'R'
                else:
                    ans += '.'

            elif i - 1 >= 0 and dominoes[i - 1] == 'L':
                if i + 1 < N and dominoes[i + 1] != 'L':
                    ans += '.'
                if i + 1 < N and dominoes[i + 1] == 'L':
                    ans += 'L'
            elif i + 1 < N and dominoes[i + 1] == 'L':
                ans += 'L'
            else:
                ans += '.'

        elif dominoes[i] == 'R':
            ans += 'R'

        elif dominoes[i] == 'L':
            ans += 'L'

    return ans


def solution(N):
    # write your code in Python 3.6
    binary = bin(N).replace("0b", "")
    print(binary)
    count = []
    sbinary = str(binary)
    zerocount = 0
    for i in range(len(sbinary)):
        if sbinary[i] == 0:
            zerocount += 1
        else:
            count.append(zerocount)
            zerocount = 0
    return max(count)


def largestIsland(grid: list[list[int]]) -> int:
    rows = len(grid)
    cols = len(grid[0])

    colors = [[-1] * cols for _ in range(rows)]
    # print(colors)
    # cl = [[-1] * cols for _ in range(rows)]
    # print(cl)
    colorCount = collections.Counter()

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def dfs(x, y, color):
        print(f'x={x},y={y}')
        if colors[x][y] == color:
            return
        if grid[x][y] == 0:
            return

        print(f'colors[x][y]={colors[x][y]}')
        colors[x][y] = color
        print(f'colors[x][y]={colors[x][y]}')
        print(f'colorCount={colorCount}')
        colorCount[color] += 1
        print(f'colorCount={colorCount}')

        for dx, dy in directions:
            print(f'dx={dx}, dy={dy}')
            nx, ny = x + dx, y + dy
            print(f'nx={nx}, ny={ny}')

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                dfs(nx, ny, color)

    currentColor = 1
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1 and colors[x][y] == -1:
                dfs(x, y, currentColor)
                currentColor += 1

    print(f'colorCount.values()={colorCount.values()}')
    print(f'list(colorCount.values())={list(colorCount.values()) + [0]}')
    best = max(list(colorCount.values()) + [0])
    print(f'best={best}')
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 0:
                used = set()
                print(f'used={used}')
                count = 1

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1 and colors[nx][ny] not in used:
                        used.add(colors[nx][ny])
                        print(f'used={used}')

                        count += colorCount[colors[nx][ny]]

                    best = max(best, count)

    return best


def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
    nums.sort()
    N = len(nums)
    subsets = []

    for mask in range(1 << N):
        current = []

        for index in range(N):
            if (mask & (1 << index)) > 0:
                current.append(nums[index])

        subsets.append(current)

    subsets.sort()
    subsetsWithoutDupe = [subsets[0]]

    for index in range(1, len(subsets)):
        if subsets[index] != subsets[index - 1]:
            subsetsWithoutDupe.append(subsets[index])

    return subsetsWithoutDupe


def minCut(s: str) -> int:
    N = len(s)
    p = 0

    for i in range(1, N):
        s1 = s[:i]
        s2 = s[i:]
        if s1 == reverse(s1) and s2 == reverse(s2):
            p += 1
        print(s1, s2)

    return p


def reverse(string):
    string = string[::-1]
    return string


def addbinary(a: str, b: str) -> str:
    int()


# arr = [1, 0, 2, 3, 4, 0, 5, 7]
def duplicateZeros(arr: list[int]) -> list:
    N = len(arr)
    temp = -1
    for i in range(1, N):
        prev = arr[i - 1]
        print(i, temp)
        curr = arr[i]
        if temp != -1:
            arr[i] = temp
            temp = curr
            continue
        # print(temp, curr)
        if prev == 0:
            temp = arr[i]
            # print(temp) 
            arr[i] = 0
    return arr


def maxCount(m: int, n: int, ops: list[list[int]]) -> int:
    if len(ops) == 0:
        return m * n

    matrix = [[0] * m for _ in range(n)]

    for coor in ops:
        for x in range(coor[0]):
            for y in range(coor[1]):
                matrix[x][y] += 1

    mx = 0
    count = {}
    for x in range(m):
        for y in range(n):
            mx = max(mx, matrix[x][y])
            if matrix[x][y] in count.keys():
                count[matrix[x][y]] += 1
            else:
                count[matrix[x][y]] = 1

                # print(mx)
    return count[mx]


def reverse(text):
    N = len(text)
    s = text

    def swap(i, j, t):
        t = list(t)
        temp = t[i]
        t[i] = t[j]
        t[j] = temp
        return "".join(t)

    j = N - 1
    for i in range(N):
        if 97 <= ord(s[i]) <= 122 or 65 <= ord(s[i]) <= 90:
            while not (97 <= ord(s[j]) <= 122 or 65 <= ord(s[j]) <= 90):
                j -= 1
            if i < j:
                print(s[i], s[j], s)
                s = swap(i, j, s)
                j -= 1
            else:
                break
        else:
            continue

    return s


def reverseOnlyLetters(self, s: str) -> str:
    l, r = 0, len(s) - 1

    res = list(s)

    while l < r:
        if not res[l].isalpha():
            l += 1
        elif not res[r].isalpha():
            r -= 1
        else:
            res[l], res[r] = res[r], res[l]

            l += 1
            r -= 1

    return ''.join(res)


def jobScheduling(startTime: list[int], endTime: list[int], profit: list[int]) -> int:

    if max(startTime) == min(startTime):
        return max(profit)

    N = len(startTime)
    start = min(startTime)
    end = max(endTime)
    dp = [0] * N
    for i in range(N):
        curr_profit = profit[i]
        curr_start = startTime[i]
        curr_end = endTime[i]
        mp = 0
        while start < end:
            mp = max(mp, curr_profit + 0)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    num = [5, 7, 7, 8, 8, 10]
    nus = [1, 2, 3, 4]
    nms = [-1, 4, 2, 3]
    # Output: [1, 3, 6, 10]
    target = 8
    word = "sea"
    wor2 = "eat"
    wrd2 = "leetcode"
    wod1 = "etco"
    word2 = "mart"
    word1 = "karma"
    inpu = [1, 2, 3, 4]
    words = ["xbc", "pcxbcf", "xb", "cxbc", "pcxbc"]
    nums = [1, 2, 3]
    # s = [1, 10, 2, 9]

    order = "cba"
    str1 = "abcd"
    grid = [[1, 0], [0, 1]]
    # largestIsland(grid)
    s = 'aab'
    arr = [1, 0, 2, 3, 4, 0, 5, 7]
    # print(duplicateZeros(arr))
    m = 39999
    n = 39999
    ops = [[19999, 19999]]
    # print(maxCount(m, n, ops))
    s5 = "a+bcd-fgh[jk][jjkl"
    # print(reverse(s5))
    startTime = [1, 2, 3, 3]
    endTime = [3, 4, 5, 6]
    profit = [50, 10, 40, 70]
    jobScheduling(startTime, endTime, profit)
    # print(minCut(s))
    # for i in range(5, -1, -1):
    #   print(i)

    # customSortString(order, str1)
    # print(solution(1041))
    # minmoves2(s)
    # root = [3, 9, 20,null, null, 15, 7]
    # levelorder(root)
    # tolowercase("LOVELY")
    # print(tolowercase("lovely"))

    # tokens = ["2", "1", "+", "3", "*"]
    # print(evalRPN(tokens))

    # longestStrChain(words)
    # printCombinations(inpu, 4, [], 0)
    # minDistance(word1, word2)

    # checkPossibility(nums)
    # runningSum(nums)
    # searchRange(nums, target)
    # solution = Solution()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
