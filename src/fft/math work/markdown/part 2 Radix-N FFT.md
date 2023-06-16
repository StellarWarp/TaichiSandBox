## Radix-N FFT

一般的 FFT 算法是将 DFT 分解成两个 DFT，然后再将这两个 DFT 分解成两个 DFT，以此类推，直到分解成两个点的 DFT，然后再将这两个点的 DFT 合并成一个两点的 DFT，以此类推，直到合并成一个 N 点的 DFT。

$$
\begin{align*}
X[k] &= \sum_{n=0}^{N-1}x[n]\omega_N^{nk}\\
&= \sum_{n=0}^{N/2-1}x[2n]\omega_N^{2nk} + \sum_{n=0}^{N/2-1}x[2n+1]\omega_N^{(2n+1)k}\\
&= \sum_{n=0}^{N/2-1}x[2n]\omega_{N/2}^{nk} + \omega_N^k\sum_{n=0}^{N/2-1}x[2n+1]\omega_{N/2}^{nk}\\
&= X_e[k] + \omega_N^kX_o[k]\\
\end{align*}
$$

为了减少FFT中内存读写的次数，我们将其分成4组 (4点DTF)

$$
\begin{align*}
X[k] &= \sum_{n=0}^{N-1}x[n]\omega_N^{nk}\\
&= \sum_{n=0}^{N/4-1}x[4n]\omega_N^{4nk} + \sum_{n=0}^{N/4-1}x[4n+1]\omega_N^{(4n+1)k} + \sum_{n=0}^{N/4-1}x[4n+2]\omega_N^{(4n+2)k} + \sum_{n=0}^{N/4-1}x[4n+3]\omega_N^{(4n+3)k}\\
&=\sum_{n=0}^{N/4-1}x[4n]\omega_{N/4}^{nk} + \omega_N^k\sum_{n=0}^{N/4-1}x[4n+1]\omega_{N/4}^{nk} + \omega_N^{2k}\sum_{n=0}^{N/4-1}x[4n+2]\omega_{N/4}^{nk} + \omega_N^{3k}\sum_{n=0}^{N/4-1}x[4n+3]\omega_{N/4}^{nk}\\
&= X_{N/4}[4k] + \omega_N^kX_{N/4}[4k+1] + \omega_N^{2k}X_{N/4}[4k+2] + \omega_N^{3k}X_{N/4}[4k+3]\\
\end{align*}
$$



另一种方法是将 Radix-2 公式继续展开，这样可以得到乘法次数更少的公式。

$$
\begin{align*}
X[k] 
&= \sum_{n=0}^{N/2-1}x[2n]\omega_{N/2}^{nk} + \omega_N^k\sum_{n=0}^{N/2-1}x[2n+1]\omega_{N/2}^{nk}\\
&= \sum_{n=0}^{N/4-1}x[4n]\omega_{N/4}^{nk} + \omega_N^k\sum_{n=0}^{N/4-1}x[4n+2]\omega_{N/4}^{nk} + \omega_N^k\left( \sum_{n=0}^{N/4-1}x[4n+1]\omega_{N/4}^{nk} + \omega_N^k\sum_{n=0}^{N/4-1}x[4n+3]\omega_{N/4}^{nk} \right)\\
&= \sum_{n=0}^{N/4-1}x[4n]\omega_{N/4}^{nk} + \omega_N^k\left( \sum_{n=0}^{N/4-1}x[4n+1]\omega_{N/4}^{nk} + \sum_{n=0}^{N/4-1}x[4n+2]\omega_{N/4}^{nk} \right) + \omega_N^{2k}\sum_{n=0}^{N/4-1}x[4n+3]\omega_{N/4}^{nk}\\
&= X_{N/4}[4k] + \omega_N^k\left( X_{N/4}[4k+1] + X_{N/4}[4k+2] \right) + \omega_N^{2k}X_{N/4}[4k+3]\\
\end{align*}
$$

这样我们就得到 Radix-4 的 FFT 算法，在最大并行度下，算法的时间复杂度为 $O(N\log_4N)$，比一般的 FFT 算法的时间复杂度 $O(N\log_2N)$ 要小。

现在我们来推导 Radix-4 FFT 的映射公式

记函数 
$$
f^{k}_N(x_0, x_1, \cdots, x_{R-1}) = \sum_{r=0}^{R-1}x_n\omega_N^{rk}
$$

第 $L$ 层的第 $k$ 个点的映射公式为

$$
\begin{align*}
X_{N}[k] &= f_{N}(X_{N/4}[4k], X_{N/4}[4k+1], X_{N/4}[4k+2], X_{N/4}[4k+3])\\
\end{align*}
$$

将 $k$ 换元为 $4k$、$4k+1$、$4k+2$、$4k+3$，得到

$$
\begin{align*}
X_{N/4}[4k] &=   f^{k}_{N/4}(X_{N/16}[16k],   &X_{N/16}[16k+4], &X_{N/16}[16k+8],  &X_{N/16}[16k+12])\\
X_{N/4}[4k+1] &= f^{k}_{N/4}(X_{N/16}[16k+1], &X_{N/16}[16k+5], &X_{N/16}[16k+9],  &X_{N/16}[16k+13])\\
X_{N/4}[4k+2] &= f^{k}_{N/4}(X_{N/16}[16k+2], &X_{N/16}[16k+6], &X_{N/16}[16k+10], &X_{N/16}[16k+14])\\
X_{N/4}[4k+3] &= f^{k}_{N/4}(X_{N/16}[16k+3], &X_{N/16}[16k+7], &X_{N/16}[16k+11], &X_{N/16}[16k+15])\\
\end{align*}
$$

与Radix-2 FFT推导类似
写成分段函数的形式
$$
X_{N/4}[k] = \begin{cases}
f^{k}_{N/4}(X_{N/16}[4k],   &X_{N/16}[4k+4], &X_{N/16}[4k+8],  &X_{N/16}[4k+12]) & k \text{ mod } 4 = 0\\
f^{k}_{N/4}(X_{N/16}[4k-3], &X_{N/16}[4k+1], &X_{N/16}[4k+5],  &X_{N/16}[4k+9]) & k \text{ mod } 4 = 1\\
f^{k}_{N/4}(X_{N/16}[4k-6], &X_{N/16}[4k-2], &X_{N/16}[4k+2],  &X_{N/16}[4k+6]) & k \text{ mod } 4 = 2\\
f^{k}_{N/4}(X_{N/16}[4k-9], &X_{N/16}[4k-5], &X_{N/16}[4k-1],  &X_{N/16}[4k+3]) & k \text{ mod } 4 = 3\\
\end{cases}
$$



(这里犯了过一个经验主义错误，不进行分段函数的展开就直接按照Radix-2的规律写，导致花费了大量的Debug时间才定位到公式推导的错误)

$$
X_{N/4}[k] = f^{(k - k \text{ mod } 4)/4}_{N/4}
\begin{pmatrix}
X_{N/16}[4k +0- 3(k \text{ mod } 4)],\\
X_{N/16}[4k +4- 3(k \text{ mod } 4)],\\
X_{N/16}[4k +8- 3(k \text{ mod } 4)],\\
X_{N/16}[4k +12- 3(k \text{ mod } 4)]\\
\end{pmatrix}
$$

(因为参数过长所以竖着写)


继续推导

$$
16\begin{cases}
X_{N/16}[16k] &=  f^{k}_{N/16}(X_{N/64}[64k],   &X_{N/64}[64k+16], &X_{N/64}[64k+32],  &X_{N/64}[64k+48])\\
\cdots\\
X_{N/16}[16k+4] &=f^{k}_{N/16}(X_{N/64}[64k+4], &X_{N/64}[64k+20], &X_{N/64}[64k+36],  &X_{N/64}[64k+52])\\
\cdots\\
X_{N/16}[16k+8] &=f^{k}_{N/16}(X_{N/64}[64k+8], &X_{N/64}[64k+24], &X_{N/64}[64k+40],  &X_{N/64}[64k+56])\\
\cdots\\
X_{N/16}[16k+12]&=f^{k}_{N/16}(X_{N/64}[64k+12],&X_{N/64}[64k+28],&X_{N/64}[64k+44], &X_{N/64}[64k+60])\\
\cdots\\
\end{cases}
$$
写成分段函数的形式

$$
X_{N/16}[k] = \begin{cases}
f^{k}_{N/16}(X_{N/64}[4 k],&X_{N/64}[4 k + 16],&X_{N/64}[4 k + 32],&X_{N/64}[4 k + 48]) & k \text{ mod } 16 = 0\\
\cdots\\
f^{k}_{N/16}(X_{N/64}[4 k - 12],&X_{N/64}[4 k + 4],&X_{N/64}[4 k + 20],&X_{N/64}[4 k + 36]) & k \text{ mod } 16 = 0\\
\cdots\\
f^{k}_{N/16}(X_{N/64}[4 k - 24],&X_{N/64}[4 k - 8],&X_{N/64}[4 k + 8],&X_{N/64}[4 k + 24]) & k \text{ mod } 16 = 0\\
\cdots\\
f^{k}_{N/16}(X_{N/64}[4 k - 36],&X_{N/64}[4 k - 20],&X_{N/64}[4 k - 4],&X_{N/64}[4 k + 12]) & k \text{ mod } 16 = 0\\
\cdots\\
\end{cases}
$$

$$
X_{N/16}[k] = f^{(k-k \text{ mod } 16)/16}_{N/16}
\begin{pmatrix}
X_{N/64}[4k +0- 3(k \text{ mod } 16)],\\
X_{N/64}[4k +16- 3(k \text{ mod } 16)],\\
X_{N/64}[4k +32- 3(k \text{ mod } 16)],\\
X_{N/64}[4k +48- 3(k \text{ mod } 16)]\\
\end{pmatrix}
$$

总结得到 Radix-4 的映射公式

记 $\overline{l} = L- l$ , $\overline{l} =\{ L, L-1,\cdots,0\}$
$$
X_{N/4^{\overline{l}}}[k] = f^{(k-k \text{ mod } 4^{\overline{l}})/4^{\overline{l}}}_{N/4^{\overline{l}}}
\begin{pmatrix}
X_{N/4^{\overline{l}+1}}[4k                        - 3(k \text{ mod } 4^{\overline{l}})],\\
X_{N/4^{\overline{l}+1}}[4k+4^{\overline{l}}       - 3(k \text{ mod } 4^{\overline{l}})],\\
X_{N/4^{\overline{l}+1}}[4k+2\cdot4^{\overline{l}} - 3(k \text{ mod } 4^{\overline{l}})],\\
X_{N/4^{\overline{l}+1}}[4k+3\cdot4^{\overline{l}} - 3(k \text{ mod } 4^{\overline{l}})]
\end{pmatrix}
$$

我们可以继续推导 Radix-8、Radix-16 …… 的映射公式
但是 Radix-4 以上的公式太长了，不方便显示，所以我把
分段函数的推导代码放这里了
```py
import sympy as sp
k = sp.Symbol('k', integer=True)
R = 8
S = 1
R_S = R**S
gap = 1
print_latex = False
if print_latex: print(f'X_{{N/{R_S}}}[k] = \\begin{{cases}}')
for r in range(0,R_S,gap):
    a = R_S*R*(k-r)/R_S
    b = [a + r + R_S*i for i in range(R)]
    for i in range(R):
        b[i] = sp.latex(sp.simplify(b[i]))
    if print_latex : 
        
        print(f'f^{{k}}_{{N/{R_S}}}(')
        for i in range(R):
            print(f'&X_{{N/{R_S*R}}}[{b[i]}]',end='')
        print(f') & k \\text{{ mod }} {R_S} = 0 \\\\ ')
        if gap != 1: 
            print('\\cdots')
    else:
        print(b)
if print_latex: print('\\end{cases}')
```

根据以上可以再总结出 Radix-R 的映射公式 (N已经使用了)

$$
\begin{align*}
X_{N/R^{\overline{l}}}[k] = f^{(k-k \text{ mod } R^{\overline{l}})/R^{\overline{l}}}_{N/R^{\overline{l}}}
\begin{pmatrix}
X_{N/R^{\overline{l}+1}}[Rk                             - (R-1)(k \text{ mod } R^{\overline{l}})],\\
X_{N/R^{\overline{l}+1}}[Rk+R^{\overline{l}}            - (R-1)(k \text{ mod } R^{\overline{l}})],\\
\cdots\\
X_{N/R^{\overline{l}+1}}[Rk+(R-1)\cdot R^{\overline{l}} - (R-1)(k \text{ mod } R^{\overline{l}})]
\end{pmatrix}
\end{align*}
$$

将 $X_{N/R^{\overline{l}}}$ 记为 $X_l$ 表示在第 $l$ 次 FFT 运算中的写入对象，并记 $r$ 为第 $r$ 个参数，$r \in \{0,1,\cdots,R-1\}$，简记函数为

$$
X_l[k] = f^{k-k \text{ mod } R^{\overline{l}}}_{N}
\begin{pmatrix}
X_{l-1}[Rk+r\cdot R^{\overline{l}} - (R-1)\cdot(k \text{ mod } R^{\overline{l}})]
\end{pmatrix}
$$

取模后得到

$$
X_l[k] = f^{k-k \text{ mod } R^{\overline{l}}}_{N}
\begin{pmatrix}
X_{l-1}[(&Rk+r\cdot R^{\overline{l}} - (R-1)\cdot(k \text{ mod } R^{\overline{l}})&)\text{ mod } N]
\end{pmatrix}
$$

其中 

$$
\begin{align*}
L &= \log_R N\\
l &= \{0,1,\cdots,L\}\\
\overline{l} &= L- l\\
\overline{l} &=\{ L, L-1,\cdots,0\}\\ 
\end{align*}
$$

利用 $m \text{ mod } n = m - n \lfloor \frac{m}{n} \rfloor$转化

$$
\begin{align*}
Rk - (R-1)\cdot(k \text{ mod } R^{\overline{l}})
&= Rk - (R-1)\cdot(k - R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor)\\
&= R\cdot R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor - R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor + k\\
k-k \text{ mod } R^{\overline{l}}
&= R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor\\
\end{align*}
$$

$$
\begin{align*}
X_l[k] 
&= f^{R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor}_{N}
\begin{pmatrix}
X_{l-1}[(&R\cdot R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor - R^{\overline{l}} \lfloor \frac{k}{R^{\overline{l}}} \rfloor + k+m\cdot R^{\overline{l}}&)\text{ mod } N]
\end{pmatrix}\\
\end{align*}
$$


现在考虑如何快速计算 $\log_RN$ 与 $R^{\overline{l}}$

因为 $R$ $N$ 都是2的整数次幂，所以 $\log_2R$ 为整数、$\log_2N$ 为整数

$$
\begin{align*}
\log_RN &= \frac{\log_2N}{\log_2R}\\
\end{align*}
$$

$$
\begin{align*}
R^{\overline{l}} &= (2^{\log_2R})^{\overline{l}} \\
&= 2^{\log_2R\cdot\overline{l}}\\
&= 2^{\log_2R\cdot(L-l)}\\
\end{align*}
$$

注意到 $\log_RN$ 不一定为整数，这意味这在进行 FFT 时往往不可能只使用一种 Radix，而是需要多种 Radix 的组合






举例来说，当 $N = 128 = 2^7$时 需要 $\log_2N=7$ 次 Radix-2 运算，1次 Radix-4 运算相当于2次 Radix-2 运算，所以可以使用3次 Radix-4 运算与1次 Radix-2 运算来代替，而一次 Radix-16 运算相当于2次 Radix-4 运算，所以也可以使用1次 Radix-16 运算与1次 Radix-4 运算来代替3次 Radix-4，最终使用的 Radix 组合为 $\{1\times2,1\times4,1\times16\}$

用公式来表述为
$$
\begin{align*}
\log_{2}128 &= \log_{2}2^7\\
&= 7 \cdot \log_{2}2\\
&= \left \lfloor 7/2 \right \rfloor \cdot \log_{2}4 + 7 \text{ mod } 2 \cdot \log_{2}2\\
&= 3 \cdot \log_{2}4 + 1 \cdot \log_{2}2\\
&= \left \lfloor 3/2 \right \rfloor \cdot \log_{2}16 + 3 \text{ mod } 2 \cdot \log_{2}4 + 1 \cdot \log_{2}2\\
&= 1 \cdot \log_{2}16 + 1 \cdot \log_{2}4 + 1 \cdot \log_{2}2\\
\end{align*}
$$

$N = 128$ 也可以用一次 Radix-64 与 一次 Radix-2 来代替

Radix与等价的Radix代替次数如下

| Radix | 2 | 4 | 8 | 16 | 32 | 64 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | 2 | 3 | 4 | 5 | 6 |
| 4 |   | 1 |   |  2 |   | 3  |
| 8 |   |   | 1 |    |   |  2  |
| 16 |   |   |   | 1 |   |    |
| 32 |   |   |   |    | 1 |    |
| 64 |   |   |   |    |    | 1  |

使用两个Radix的组合时，有对应的$log_2N$为
| Radix | 2 | 4 | 8 | 16 | 32 | 64 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 2 | 3 | 4 | 5 | 6 | 7 |
| 4 |   | 4 | 5 | 6 | 7 | 8 |
| 8 |   |   | 6 | 7 | 8 | 9 |
| 16 |   |   |   | 8 | 9  |  10  |
| 32 |   |   |   |    | 10 |  11  |
| 64 |   |   |   |    |    | 12  |

可见同一 $N$ 即使是在相同的运算次数下，也有不同的组合方式

我们采用贪心策略来选择 Radix 组合，即每次选择最大的 Radix 来代替，这样可以保证使用的 Radix 数量最少



采用 Radix $R_{max}$ 的次数为 

$$
T = \lfloor \frac{\log_2N}{\log_2R_{max}}\rfloor
$$

最后一次

Radix $R = 2^{log_2N \text{ mod } \log_2R_{max}} $ 的操作以完成FFT


