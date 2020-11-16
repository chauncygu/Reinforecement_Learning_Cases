'''
给一定数量的信封，带有整数对 (w, h) 分别代表信封宽度和高度。一个信封的宽高均大于另一个信封时可以放下另一个信封。求最大的信封嵌套层数。
样例
给一些信封 [[5,4],[6,4],[6,7],[2,3]] ，最大的信封嵌套层数是 3([2,3] => [5,4] => [6,7])。
'''

from typing import List

class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        size = len(envelopes)
        #print(envelopes)
        if size < 2:  # if the size of envelopes is smaller than 2, return
            return size
        envelopes.sort(key=lambda x: x[0])  # sort the envelopes according the width
        dp = [1 for _ in range(size)]  #record the number of envelope layers
        #print(dp)
        for i in range(1, size):
            for j in range(i):
                # compare the width and length of envelopes
                if envelopes[j][0] < envelopes[i][0] and envelopes[j][1] < envelopes[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

if __name__ == '__main__':
    envelopes = [[5,4],[6,4],[6,7],[2,3],[99,100]]  #input the matrix of envelopes
    solution = Solution()
    res = solution.maxEnvelopes(envelopes)
    print("The maximum envelope nesting layer is %d" %(res))
