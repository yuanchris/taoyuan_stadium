# # def triangles():
# #     mylist = [1]
# #     while True:
# #         yield mylist
# #         newlist = [1]
# #         for x in range (1, len(mylist)):
# #             newlist.append(mylist[x] + mylist[x - 1])
# #         newlist.append(1)
# #         mylist = newlist
# #     return 'done'
# def triangles():

#     L=[1]  

#     while True:

#         yield L

#         L.append(0)
#         L = [ L[i]+ L[i-1] for i in range(len(L))]
    

# # 期待輸出:
# # [1]
# # [1, 1]
# # [1, 2, 1]
# # [1, 3, 3, 1]
# # [1, 4, 6, 4, 1]
# # [1, 5, 10, 10, 5, 1]
# # [1, 6, 15, 20, 15, 6, 1]
# # [1, 7, 21, 35, 35, 21, 7, 1]
# # [1, 8, 28, 56, 70, 56, 28, 8, 1]
# # [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
# n = 0
# results = []
# for t in triangles():
#     results.append(t)
#     n = n + 1
#     if n == 10:
#         break

# for t in results:
#     print(t)

# if results == [
#     [1],
#     [1, 1],
#     [1, 2, 1],
#     [1, 3, 3, 1],
#     [1, 4, 6, 4, 1],
#     [1, 5, 10, 10, 5, 1],
#     [1, 6, 15, 20, 15, 6, 1],
#     [1, 7, 21, 35, 35, 21, 7, 1],
#     [1, 8, 28, 56, 70, 56, 28, 8, 1],
#     [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
# ]:
#     print('測試通過!')
# else:
#     print('測試失敗!')


def triangles():  
    m = [1]
    while True:
        print('Before yield:',id(m))  # By Ross: 这里使用id查看变量的物理地址 
        yield m[:]
        m.append(0)
        m = [m[i]+m[i-1] for i in range(len(m))]  # By Ross: 这里通过=重新给变量m赋值,物理地址更改,id(m)的值改变
        print('After:',id(m))

n = 0
results = []
for t in triangles():
    results.append(t)
    print('results.append(t):', id(results[-1]))  # By Ross: 这里的t是个列表,和上面triangle()中yield m中的m为同一个值(指向的是内存中的同一处物理地址)
    n = n + 1
    if n == 1:
        break

for t in results:
    print(t)
