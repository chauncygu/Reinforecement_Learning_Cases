
'''
#机器人运动范围
class Solution:
    # 注意和矩阵路径的区别，不一定到头，只要满足条件即可，可以返回（回溯法的特点）
    # 且路径节点可重复，无步数限制
    def __init__(self):  # 机器人可以倒回来，但不能重复计数。
        self.count = 0

    def movingCount(self, threshold, rows, cols):
        # write code here
        flag = [[1 for i in range(cols)] for j in range(rows)]
        self.findWay(flag, 0, 0, threshold)  # 从（0，0）开始走
        return self.count

    def findWay(self, flag, i, j, k):
        if i >= 0 and j >= 0 and i < len(flag) and j < len(flag[0]) and sum(list(map(int, str(i)))) + sum(
                list(map(int, str(j)))) <= k and flag[i][j] == 1:
            flag[i][j] = 0
            self.count += 1
            self.findWay(flag, i - 1, j, k)
            self.findWay(flag, i + 1, j, k)
            self.findWay(flag, i, j - 1, k)
            self.findWay(flag, i, j + 1, k)

model = Solution()
model.movingCount(5,6,6)
print(model.count)


    def __init__(self, A, B):  # 机器人可以倒回来，但不能重复计数。
        self.A = A
        self.B = B


#a=(i+2 for i in  range(3))
#print(a)
'''
