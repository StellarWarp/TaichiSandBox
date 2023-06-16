# 线性层次包围盒
## 莫顿码 Morton Code

Morton Code将x,y,z坐标编码为一个整数，使得相邻的坐标在编码后也是相邻的。这样，我们就可以将三维空间的点映射到一维空间，从而可以使用一维空间的数据结构来存储三维空间的点。

具体的编码方式为：将各坐标轴的二进制数位交错排列，如对于三维的三位坐标
$$
\begin{array}{l}
x = x_1x_2x_3\\
y = y_1y_2y_3\\
z = z_1z_2z_3
\end{array}
$$
编码后的莫顿码为
$$
mort(x,y,z) = x_1y_1z_1x_2y_2z_2x_3y_3z_3
$$
对于二维的二位坐标
$$
\begin{array}{l}
x = x_1x_2\\
y = y_1y_2
\end{array}
$$
编码后的莫顿码为
$$
mort(x,y) = x_1y_1x_2y_2
$$


Games401 课程中的二维Morton Code示意图如下

![Games401 课程中的二维Morton Code示意图](C:\Users\Estelle\source\repos\TaichiSandBox\BVH.assets\image-20230517225043437.png)

## 有序二叉压缩前缀树(Ordered binary radix tree)


### 层次划分
![image-20230517225857731](C:\Users\Estelle\source\repos\TaichiSandBox\BVH.assets\image-20230517225857731.png)

整个空间为根节点，其包含所有物体，交替对x,y,z坐标进行以下操作：
从坐标轴的高位开始，将空间划分为两个子空间，左子空间包含坐标轴的0，右子空间包含坐标轴的1

### 有序二叉压缩前缀树

![image-20230517230723407](C:\Users\Estelle\source\repos\TaichiSandBox\BVH.assets\image-20230517230723407.png)

二叉树的特性
$$
n_{internal} = n_{leaf} - 1
$$

#### 前缀编码

$\delta_{ij}$ 代表叶子节点 $i$ 与叶子节点 $j$ 的最长公共前缀
$$
\begin{align*}
\forall i',j'\in [i,j]\\
\delta_{ij} \leq \delta_{i'j'}
\end{align*}
$$
即在叶子节点 $i$ 与叶子节点 $j$ 之间的任意两个叶子节点的最长公共前缀均小于等于叶子节点 $i$ 与叶子节点 $j$ 的最长公共前缀

**性质**
- $\delta_{ij}$对于的节点是叶子节点i与叶子节点j的共同父节点
- $\delta_{ij}$的长度是共同父节点的深度

### 并行构建内部节点

1. 计算内部节点的索引范围
2. 计算左右节点的分裂点
3. 构建AABB


### 并行基数排序

- 将数字从低位到高位依次进行排序
- 每一个数字对于一个线程

1. 建立当前位映射表
   1. 0映射表
   2. 1映射表
   3. 合并映射表
2. 将数按照位置放入新的数组中

```c
one[i] = bit[current](input[i])
zero[i] = ~ one[i]
zero_map = prefix_sum(zero)
total_zero = zero_map[-1] + zero[-1]
one_map[i] = i + total_zero - zero_map[i] // being - sum of zero
map[i] = one[i] ? one_map[i] : zero_map[i]
output[map[i]] = input[i]

```

#### Reduce and Scan Prefix Sum
前缀和的计算对GPU并不友好，但是有可以加速的方法

Reduce-then-scan方法：复杂度(WorkComplexity)为$O(n)$
该算法整体分为两个阶段，第一个阶段是reduce，向上遍历到根，第二个阶段是scan，从根回归到叶
```py
# reduce
for i in range(1, n):
    if i % 2 == 0:
        input[i] = input[i // 2]
    else:
        input[i] = input[i // 2] + input[i - 1]

# scan
for i in range(n - 1, 0, -1):
    if i % 2 == 0:
        input[i] = input[i // 2 - 1] + input[i]
    else:
        input[i] = input[i // 2]

```


