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
