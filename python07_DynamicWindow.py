
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
