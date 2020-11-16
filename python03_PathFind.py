
'''
#蜜蜂采蜂蜜题目
pathsum = []
def array(coord, begin, end):
    global pathsum
    l = len(coord)
    if begin >= end:  # 排序结束
        print(coord)
        path = int(pow((coord[0][0] ** 2 + coord[0][1] ** 2), 1 / 2) + pow((coord[l - 1][0] ** 2 + coord[l - 1][1] ** 2), 1 / 2))  # 一首一尾到原点距离固定
        for j in range(l - 1):
            path = path + int(pow(((coord[j + 1][0] - coord[j][0]) ** 2 + (coord[j + 1][1] - coord[j][1]) ** 2), 1 / 2))
        print(path)
        pathsum.append(path)
    else:
        for x in range(begin, end):
            coord[x], coord[begin] = coord[begin], coord[x]  # 把begin开始的元素和某个元素调换了
            array(coord, begin + 1, end)  # 排序从begin+1剩下的元素
            coord[x], coord[begin] = coord[begin], coord[x]  # 把排列复原

coord = [(50, 0), (50, 5), (50, 20), (50, 30), (50, 50)]
array(coord, 0, len(coord))
print(pathsum)
print(min(pathsum))
print(len(coord))
'''

'''
#全排列解决办法1: 回溯法
class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def backtrack(position, end):
            """
            Find possible results using backtrack.
            :param position:
            :param end:
            :return:
            """

            if position == end:
                res.append(nums[:])
                return

            for index in range(position, end):
                nums[index], nums[position] = nums[position], nums[index]
                backtrack(position + 1, end)
                nums[index], nums[position] = nums[position], nums[index]

        res = []
        backtrack(0, len(nums))
        return res
'''

'''
#全排列:（方案2：深度优先搜索
class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        visit = [True for _ in range(len(nums))]
        tmp = nums[:]

        def dfs(position):
            if position == len(nums):
                res.append(tmp[:])
                return

            for index in range(0, len(nums)):
                if visit[index]:
                    tmp[position] = nums[index]
                    visit[index] = False
                    dfs(position + 1)
                    visit[index] = True

        res = []
        dfs(0)
        return res
'''
