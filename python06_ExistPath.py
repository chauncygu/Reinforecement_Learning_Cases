
'''
#路径存在判断
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        for i in range(rows):
            for j in range(cols):
                if matrix[i * cols + j] == path[0]:
                    if self.findPath(list(matrix), rows, cols, path[1:], i, j):
                        return True
        return False

    def findPath(self, matrix, rows, cols, path, m, n):
        # 结束条件
        if not path:
            return True
        if m > cols or n > rows or m < 0 or n < 0:
            return False
        # 避免重复遍历
        matrix[m * cols + n] = '0'

        if n + 1 < rows and matrix[m * cols + n + 1] == path[0]:
            return self.findPath(matrix, rows, cols, path[1:], m, n + 1)
        if n - 1 >= 0 and matrix[m * cols + n - 1] == path[0]:
            return self.findPath(matrix, rows, cols, path[1:], m, n - 1)
        if m + 1 < cols and matrix[(m + 1) * cols + n] == path[0]:
            return self.findPath(matrix, rows, cols, path[1:], m + 1, n)
        if m - 1 > 0 and matrix[(m - 1) * cols + n] == path[0]:
            return self.findPath(matrix, rows, cols, path[1:], m - 1, n)

model = Solution()
#a = [ [a, b, c, e], [s, f, c, s], [a, d, e, e] ]
aa=model.hasPath('dsdffderergt', 3, 4, 'ds')
print(aa)
'''

# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        res = []
        self.level(pRoot, 0, res)
        return res

    def level(self, root, level, res):
        if not root:
            return
        if level == len(res):
            res.append([])
        res[level].append(root.val)
        if root.left:
            self.level(root.left, level + 1, res)
        if root.right:
            self.level(root.right, level + 1, res)
a = [1,2,3,4]
tree = TreeNode(a)
model = Solution()
print(model.Print(tree))
