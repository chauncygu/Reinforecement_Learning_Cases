#A为一个十进制数（以整数为例），k位，k<100。求B使得B为大于A的最小整数，且A各位的和等于B各位的和。

class Solution:
    # Find the sum of the bits of each number
    def sumAB(self, a):
        n = abs(int(a))
        sum = 0
        while n != 0:
            sum += n % 10
            n //= 10
        return sum

    # Find the B value
    def findB(self,A):
        for i in range(A,(10**100)-1):
            if self.sumAB(A) == self.sumAB(i) and i > A:
                return i


if __name__ == '__main__':
    solution = Solution()
    A = int(input())  # input A value
    print("The value of B is %d" %(solution.findB(A)))

