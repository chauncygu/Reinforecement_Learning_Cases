


class Node(object):
    """单链表的结点"""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None
'''
class SingleLinkList(object):
    """单链表"""
    def __init__(self):
        self._head = None


if __name__ == '__main__':
    # 创建链表
    link_list = SingleLinkList()
    # 创建结点
    node1 = Node(1)
    node2 = Node(2)

    # 将结点添加到链表
    link_list._head = node1
    # 将第一个结点的next指针指向下一结点
    node1.next = node2

    # 访问链表
    print(link_list._head.item)  # 访问第一个结点数据
    print(link_list._head.next.item)  # 访问第二个结点数据
'''

'''
class SingleLinkList(object):
    """单链表"""
    def __init__(self):
        self._head = None
    def is_empty(self):
        """判断链表是否为空"""
        return self._head is None
    def length(self):
        """链表长度"""
        # 初始指针指向head
        cur = self._head
        count = 0
        # 指针指向None 表示到达尾部
        while cur is not None:
            count += 1
            # 指针下移
            cur = cur.next
        return count
    def items(self):
        """遍历链表"""
        # 获取head指针
        cur = self._head
        # 循环遍历
        while cur is not None:
            # 返回生成器
            yield cur.item
            # 指针下移
            cur = cur.next
    def add(self, item):
        """向链表头部添加元素"""
        node = Node(item)
        # 新结点指针指向原头部结点
        node.next = self._head
        # 头部结点指针修改为新结点
        self._head = node
    def append(self, item):
        """尾部添加元素"""
        node = Node(item)
        # 先判断是否为空链表
        if self.is_empty():
            # 空链表，_head 指向新结点
            self._head = node
        else:
            # 不是空链表，则找到尾部，将尾部next结点指向新结点
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node
    def insert(self, index, item):
        """指定位置插入元素"""
        # 指定位置在第一个元素之前，在头部插入
        if index <= 0:
            self.add(item)
        # 指定位置超过尾部，在尾部插入
        elif index > (self.length() - 1):
            self.append(item)
        else:
            # 创建元素结点
            node = Node(item)
            cur = self._head
            # 循环到需要插入的位置
            for i in range(index - 1):
                cur = cur.next
            node.next = cur.next
            cur.next = node
    def remove(self, item):
        """删除节点"""
        cur = self._head
        pre = None
        while cur is not None:
            # 找到指定元素
            if cur.item == item:
                # 如果第一个就是删除的节点
                if not pre:
                    # 将头指针指向头节点的后一个节点
                    self._head = cur.next
                else:
                    # 将删除位置前一个节点的next指向删除位置的后一个节点
                    pre.next = cur.next
                return True
            else:
                # 继续按链表后移节点
                pre = cur
                cur = cur.next
    def find(self, item):
        """查找元素是否存在"""
        return item in self.items()
'''


'''
#旋转n阶方阵
if __name__ == '__main__':
    a=[1,2,3]
    b=[4,5,6]
    c=[7,8,9]
    zipped=zip(a,b,c)
    for i in zipped:
        print(i)
    print('加*号相当于解压')
    for i in zip(*zip(a,b,c)):
        print(i)
        print(i[1])
'''

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

'''
import numpy as np

"""
编辑距离：用于比较两个字符串的相似程度
StrA: 字符串A
StrB：字符串B
函数包含处理过程
包含bug：处理不同长度的字符串的时候会出现报错
"""


def min_edit_dist(StrA, StrB):
    # 获取字符串长度
    a_length = len(StrA)
    b_length = len(StrB)
    # 创建矩阵
    matrix = np.zeros((b_length + 1, a_length + 1))
    # 初始化矩阵
    for i in range(1, a_length + 1):
        matrix[0][i] = i
    for j in range(1, b_length + 1):
        matrix[j][0] = j
    # 开始进行动态规划
    cost = 0  # 代价值
    for i in range(1, b_length + 1):
        for j in range(1, a_length + 1):
            if StrA[j - 1] == StrB[i - 1]:
                cost = 0
            else:
                cost = 1
            # 三种字符串操作方式增加 删除 替换
            edit_exchange_dis = matrix[j - 1][i - 1] + cost  # 替换
            edit_add_dis = matrix[j - 1][i] + 1  # 添加
            edit_del_dis = matrix[j][i - 1] + 1  # 删除
            matrix[j][i] = min(edit_exchange_dis, edit_add_dis, edit_del_dis)
            # print(matrix[j][i])                          #最小编辑距离更改记录
            print(matrix)  # 打印算法过程
            print('______________________')  # 分割
    # print(i)    遍历完整性
    print('相似度为：', 1 - 1 / max(a_length, b_length))  # 1-（字符串更改最少次数/字符串最长距离）
# print(matrix)
# print(StrA[a_length-1])
# print(StrB[b_length-1])
# print(a_length)


#min_edit_dist('jsi2nz', 'jsionz')  # 测试

'''

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

'''
#滑动窗口
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # 如果数组 num 不存在，则返回 []
        if not num:
            return []
        # 如果滑动窗口的大小大于数组的大小，或者 size 小于 0，则返回 []
        if size > len(num) or size <1:
            return []
        # 如果滑动窗口的大小为 1 ，则直接返回原始数组
        if size == 1:
            return num
        # 存放最大值，次大值的数组，和存放输出结果数组的初始化
        temp = [0]
        res = []
        # 对于数组中每一个元素进行判断
        for i in range(len(num)):
            # 判断第 i 个元素是否可以加入 temp 中
            # 如果比当前最大的元素还要大，清空 temp 并把该元素放入数组
            # 首先判断当前最大的元素是否过期
            #print(i)
            if i -temp[0] > size-1:
                print(temp[0])
                temp.pop(0)
            # 将第 i 个元素与 temp 中的值比较，将小于 i 的值都弹出
            while (len(temp)>0 and num[i] >= num[temp [-1]]):
                temp.pop()
            # 如果现在 temp 的长度还没有达到最大规模，将元素 i 压入
            if len(temp)< size-1:
                temp.append(i)
            # 只有经过一个完整的窗口才保存当前的最大值
            if i >=size-1:
                res.append(num[temp [0]])
        return res

model = Solution()
a = model.maxInWindows([2,3,4,2,6,2,5,1],3)
print(a)
'''

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





