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
