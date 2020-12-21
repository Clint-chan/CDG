---
title: 算法导论总复习
author: Clint_chan
categories: [ 陈定钢]
image: assets/images/flag1.jpg
tags: [算法导论复习,向高分前进！]
---
  
----------
### 目录
* [一.排序算法](#一排序算法)
	 * [1.选择排序](#1选择排序)
	     * [1.1原理&图示](#11原理及图示)
	     * [1.2代码部分](#12代码部分)
	 * [2.插入排序](#2插入排序)
	     * [2.1原理&图示](#21原理及图示)
	     * [2.2代码部分](#22代码部分)
	 * [3.归并排序](#3归并排序)
	     * [3.1原理&图示](#31原理及图示)
	     * [3.2代码部分](#32代码部分)
	 * [4.快速排序](#4快速排序)
	     * [4.1原理&图示](#41原理及图示)
	     * [4.2代码部分](#42代码部分)
	 * [5.堆排序](#5堆排序)
	     * [5.1原理&图示](#51原理及图示)
	     * [5.2代码部分](#52代码部分)
	 * [6.计数排序](#6计数排序)
	     * [6.1原理&图示](#61原理及图示)
	     * [6.2代码部分](#62代码部分)
	 * [7.桶排序](#7桶排序)
	     * [7.1原理&图示](#71原理及图示)
	     * [7.2代码部分](#72代码部分)
	 * [8.基数排序](#8基数排序)
	     * [8.1原理&图示](#81原理及图示)
	     * [8.2代码部分](#82代码部分)
	 * [9.希尔排序](#9希尔排序)
	     * [9.1原理&图示](#91原理及图示)
	     * [9.2代码部分](#92代码部分)
	 * [10.排序算法比较](#10排序算法比较)
*  [二.红黑树](#二红黑树)
*  [三.动态规划](#三动态规划)
	  * [1.钢条问题](#1钢条切割问题)
	  * [2.矩阵链问题](#2矩阵链问题)
*  [四.贪心算法](#四贪心算法)
*  [考试资料](#考试资料)
*  [教案](#教案)
*  [参考文献](#参考文献)
	




 
 
#### 一.排序算法

##### 1.选择排序
###### 1.1原理及图示
![选择排序](https://clint-chan.github.io/CDG/assets/images/selectionSort.gif)
###### 1.2代码部分
``` 
#Python代码
def selectionSort(arr):
    for i in range(len(arr) - 1):
        # 记录最小数的索引
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        # i 不是最小数时，将 i 和最小数进行交换
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr
```

##### 2.插入排序
###### 2.1原理及图示
![插入排序](https://www.runoob.com/wp-content/uploads/2019/03/insertionSort.gif)
###### 2.2代码部分
``` 
# Python代码
def insertionSort(arr):
    for i in range(len(arr)):
        preIndex = i-1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:
            arr[preIndex+1] = arr[preIndex]
            preIndex-=1
        arr[preIndex+1] = current
    return arr
```

##### 3.归并排序
###### 3.1原理及图示
![归并排序](https://www.runoob.com/wp-content/uploads/2019/03/mergeSort.gif)
###### 3.2代码部分
``` 
# Python代码
def mergeSort(arr):
    import math
    if(len(arr)<2):
        return arr
    middle = math.floor(len(arr)/2)
    left, right = arr[0:middle], arr[middle:]
    return merge(mergeSort(left), mergeSort(right))

def merge(left,right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0));
    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0));
    return result
```

##### 4.快速排序<i class="fas fa-highlighter"></i>
###### 4.1原理及图示
![快速排序](https://www.runoob.com/wp-content/uploads/2019/03/quickSort.gif)
###### 4.2代码部分
``` 
#快速排序Python代码
def quickSort(arr, left=None, right=None):
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(arr)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partition(arr, left, right)
        quickSort(arr, left, partitionIndex-1)
        quickSort(arr, partitionIndex+1, right)
    return arr

def partition(arr, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if arr[i] < arr[pivot]:
            swap(arr, i, index)
            index+=1
        i+=1
    swap(arr,pivot,index-1)
    return index-1

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
```

##### 5.堆排序<i class="fas fa-highlighter"></i>
###### 5.1原理及图示
![堆排序](https://www.runoob.com/wp-content/uploads/2019/03/heapSort.gif)

> 考试如无特别说明，高度深度从0开始记。
> >树的高度为根节点的高度；某节点的高度等于该节点到叶子节点的最长路径（边数）；节点的深度是根节点到这个节点所经历的边的个数。

###### 5.2代码部分
``` 
#堆排序 Python代码
def buildMaxHeap(arr):
    import math
    for i in range(math.floor(len(arr)/2),-1,-1):
        heapify(arr,i)

def heapify(arr, i):
    left = 2*i+1
    right = 2*i+2
    largest = i
    if left < arrLen and arr[left] > arr[largest]:
        largest = left
    if right < arrLen and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        swap(arr, i, largest)
        heapify(arr, largest)

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heapSort(arr):
    global arrLen
    arrLen = len(arr)
    buildMaxHeap(arr)
    for i in range(len(arr)-1,0,-1):
        swap(arr,0,i)
        arrLen -=1
        heapify(arr, 0)
    return arr
```

##### 6.计数排序
###### 6.1原理及图示
![计数排序](https://www.runoob.com/wp-content/uploads/2019/03/countingSort.gif)
###### 6.2代码部分
``` 
#计数排序Python代码
def countingSort(arr, maxValue):
    bucketLen = maxValue+1
    bucket = [0]*bucketLen
    sortedIndex =0
    arrLen = len(arr)
    for i in range(arrLen):
        if not bucket[arr[i]]:
            bucket[arr[i]]=0
        bucket[arr[i]]+=1
    for j in range(bucketLen):
        while bucket[j]>0:
            arr[sortedIndex] = j
            sortedIndex+=1
            bucket[j]-=1
    return arr
```

##### 7.桶排序
###### 7.1原理及图示
![桶排序](https://clint-chan.github.io/CDG/assets/images/bucket-sort.png)
###### 7.2代码部分
``` 
#桶排序Python代码
def bucketSort(nums):
    #选择一个最大的数
    max_num = max(nums)
    # 创建一个元素全是0的列表, 当做桶
    bucket = [0]*(max_num+1)
    # 把所有元素放入桶中, 即把对应元素个数加一
    for i in nums:
        print(f"{bucket=}")
 
        bucket[i] += 1
        # 存储排序好的元素
    sort_nums = []
    print(f"{bucket=}")
    for j in range(len(bucket)):
       n = bucket[j]
       if n != 0:
           for _ in range(n):
               print(f"{sort_nums=}{j=}")
               sort_nums.append(j)
 
 
    return sort_nums
 
nums = [5,6,3,2,1,65,2,0,8,0,9]
print("测试结果:")
print(bucketSort(nums))
```

##### 8.基数排序
###### 8.1原理及图示
![基数排序](https://www.runoob.com/wp-content/uploads/2019/03/radixSort.gif)
###### 8.2代码部分
``` 
#基数排序Python代码
def radix_sort(s):
    """基数排序"""
    i = 0 # 记录当前正在排拿一位，最低位为1
    max_num = max(s)  # 最大值
    j = len(str(max_num))  # 记录最大值的位数
    while i < j:
        bucket_list =[[] for _ in range(10)] #初始化桶数组
        for x in s:
            bucket_list[int(x / (10**i)) % 10].append(x) # 找到位置放入桶数组
        print(bucket_list)
        s.clear()
        for x in bucket_list:   # 放回原序列
            for y in x:
                s.append(y)
        i += 1

if __name__ == '__main__':
    a = [334,5,67,345,7,345345,99,4,23,78,45,1,3453,23424]
    radix_sort(a)
    print(a)
```

##### 9.希尔排序
###### 9.1原理及图示
![希尔排序](https://www.runoob.com/wp-content/uploads/2019/03/Sorting_shellsort_anim.gif)
###### 9.2代码部分
``` 
#Python代码
def shellSort(arr):
    import math
    gap=1
    while(gap < len(arr)/3):
        gap = gap*3+1
    while gap > 0:
        for i in range(gap,len(arr)):
            temp = arr[i]
            j = i-gap
            while j >=0 and arr[j] > temp:
                arr[j+gap]=arr[j]
                j-=gap
            arr[j+gap] = temp
        gap = math.floor(gap/3)
    return arr
```
##### 10.排序算法比较
Congretulations! 恭喜你已经复习完了排序算法，下面咱们来比较这些算法吧！
<i class="fab fa-angellist fa-3x"></i>

![排序总结](https://clint-chan.github.io/CDG/assets/images/比较.jpg)

对上述算法做到知其然并知其所以然。

----------


#### 二.红黑树
 [点击这里回到目录](#目录)
#### 三.动态规划 <i class="fas fa-highlighter"></i>
	动态规划三要素：
					（1）问题的阶段
					（2）每个阶段的状态
					（3）相邻两个阶段之间的递推关系
	特点：
					（1）最优化原理
					（2）无后效性
					（3）有重叠子问题
	


----------
##### 1.钢条切割问题
- ①自顶向下（非多项式级复杂度）
- ②带备忘录的自顶向下（平方阶）
- ③自底向上（平方阶）
  
`①自顶向下`
``` 
#自顶向下
1 def CutRod(p, n):  # 函数返回：切割长度为 n 的钢条所得的最大收益
2    if n == 0:
3        return 0
4    q = -1
5    for i in range(1, n+1):
6        q = max(q, p[i] + CutRod(p, n-i)) 
7    return q
```
`Core part:Line 5,6.Specific codes go looking up pdf. `

`②带备忘录的自顶向下`
``` 
#带备忘录的自顶向下
1 def MemorizedCutRod(p, n):
2    r=[-1]*(n+1)                          #  数组初始化
3    def MemorizedCutRodAux(p, n, r):
4        if r[n] >= 0:
5            return r[n]
6        q = -1
7        if n == 0:
8            q = 0
9        else:
10            for i in range(1, n + 1):
11                q = max(q, p[i] + MemorizedCutRodAux(p, n - i, r))
12        r[n] = q
13        return q
14    return MemorizedCutRodAux(p, n, r),r
```
`Core part:Line 3,4,5.Specific codes go looking up pdf. `

`③自底向上`
``` 
1 def BottomUpCutRod(p, n):
2   r = [0]*(n+1)
3    for i in range(1, n+1):
4        if n == 0:
5            return 0
6        q =0
7        for j in range(1, i+1):
8            q = max(q, p[j]+r[i-j])
9            r[i] = q
10    return r[n],r
```
`Core part:Line All.Specific codes go looking up pdf.`
##### 2.矩阵链问题
 [点击这里回到目录](#目录)
#### 四.贪心算法
 [点击这里回到目录](#目录)
 
 -----------------
#### 考试资料
- [1] [排序算法选择题](https://clint-chan.github.io/CDG/assets/file/%E7%AE%97%E6%B3%95%E5%AF%BC%E8%AE%BA%E9%80%89%E6%8B%A9%E9%A2%98%E7%BB%83%E4%B9%A0.docx?_blank)
- [2] [（李芸老师）算法导论第1次作业](https://clint-chan.github.io/CDG/assets/file/（李芸老师）算法导论第1次作业.docx)
- [3] [（李芸老师）算法导论第2次作业](https://clint-chan.github.io/CDG/assets/file/（李芸老师）算法导论第2次作业.docx)
- [4] [（李芸老师）算法导论第3次作业](https://clint-chan.github.io/CDG/assets/file/（李芸老师）算法导论第3次作业.docx)
- [5] [（李芸老师）2020秋期中测试题](https://clint-chan.github.io/CDG/assets/file/（李芸老师）2020秋期中测试题.doc)
#### 教案
 - [1] [算法基础知识](https://clint-chan.github.io/CDG/assets/file/pdf/(1)算法基础知识.pdf)
 - [2] [函数的增长](https://clint-chan.github.io/CDG/assets/file/pdf/(2)%20函数的增长.pdf)
 - [3] [概率分析与随机算法](https://clint-chan.github.io/CDG/assets/file/pdf/(3)概率分析与随机算法.pdf)
 - [4] [分治策略](https://clint-chan.github.io/CDG/assets/file/pdf/(4)%20分治策略.pdf)
 - [5] [堆排序](https://clint-chan.github.io/CDG/assets/file/pdf/(5)%20堆排序.pdf)
 - [6] [快速排序](https://clint-chan.github.io/CDG/assets/file/pdf/(6)%20快速排序.pdf)
 - [7] [线性时间排序、统计量](https://clint-chan.github.io/CDG/assets/file/pdf/(7)%20线性时间排序、统计量.pdf)
 - [8] [基本数据结构](https://clint-chan.github.io/CDG/assets/file/pdf/(8)%20基本数据结构.pdf)
 - [9] [二叉搜索树](https://clint-chan.github.io/CDG/assets/file/pdf/(9)%20二叉搜索树.pdf)
 - [10] [红黑树](https://clint-chan.github.io/CDG/assets/file/pdf/(10)%20红黑树.pdf)
 - [11] [动态规划](https://clint-chan.github.io/CDG/assets/file/pdf/(11)%20动态规划.pdf)
 - [12] [贪心算法](https://clint-chan.github.io/CDG/assets/file/pdf/(12)%20贪心算法.pdf)
 - [13] [基本的图算法](https://clint-chan.github.io/CDG/assets/file/pdf/(13)%20基本的图算法.pdf)
 - [14] [粒子群优化算法](https://clint-chan.github.io/CDG/assets/file/pdf/(14)%20粒子群优化算法.pdf)
 - [15] [遗传算法和模拟退火算法](https://clint-chan.github.io/CDG/assets/file/pdf/(15)%20遗传算法和模拟退火算法.pdf)
#### 参考文献
- [1] [快速排序算法——C/C++](https://blog.csdn.net/weixin_42109012/article/details/91645051)
- [2] [Python实现二叉树遍历的递归和非递归算法](https://blog.csdn.net/bluesliuf/article/details/89321695)
- [3] [求递归算法时间复杂度：递归树](https://www.cnblogs.com/wu8685/archive/2010/12/21/1912347.html)
- [4] [主方法求解递归式](https://blog.csdn.net/qq_40512922/article/details/96932368)
- [5] [30张图带你彻底理解红黑树](https://www.jianshu.com/p/e136ec79235c)
- [6] [常见的八大排序算法的比较和选择依据](https://www.cnblogs.com/bjwu/articles/10006419.html)
- [7] [各种排序算法总结和比较](https://www.cnblogs.com/zhaoshuai1215/p/3448154.html?_blank)

`温馨提示：用Ctrl+鼠标左键可以创建新窗口打开链接以便流畅体验。`

[点击这里回到目录](#目录)
