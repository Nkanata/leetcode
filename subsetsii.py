class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:

        nums.sort()

        ans = []
        self.backtrack([], nums, 0, ans)
        return ans

    def backtrack(self, curr, nums, idx, ans):
        ans.append(curr[:])
        if idx >= len(nums):
            return
        for i in range(idx, len(nums)):
            if i > idx and nums[i] == nums[i - 1]:
                continue
            curr.append(nums[i])
            self.backtrack(curr, nums, i + 1, ans)
            curr.pop()
