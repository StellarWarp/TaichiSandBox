## 蝶形公式

直接将两个Raidx-2 DFT 替换成 Radix-4 DFT 或更高阶的 DFT 会增加运算量，(在GPU中)反而会使得 FFT 变慢，现在要做的是把 Radix-N 在内部重新分解成更小的 Raidx 进行计算，这样才能在减少读写次数的同时不增加运算量

再让我们回到 Radix-2 的 FFT

$$
\begin{align*}
X[k] &= X_{N/2}[2k] + \omega^k_NX_{N/2}[2k+1]\\
\end{align*}
$$

这个公式只表示了 FFT 的分解，却没有FFT的精髓

FFT之所以能够快速，是因为它利用了重叠子问题的性质，即 $X_{N/2}[2k]$ 与 $X_{N/2}[2k+1]$ 被重复使用了多次

因为$X_{N/2}$是循环数组，有

$$
\begin{align*}
X_{N/2}[2k] &= X_{N/2}[2k+N] \\&= X_{N/2}[2(k+N/2)]\\
X_{N/2}[2k+1] &= X_{N/2}[2k+1+N]\\&= X_{N/2}[2(k+N/2)+1]\\
\end{align*}
$$

$$
\begin{align*}
&X_{N/2}[2(k+N/2)] + \omega^{k+N/2}_NX_{N/2}[2(k+N/2)+1]\\
=&X_{N/2}[2(k+N/2)] + \omega^{N/2}_N\omega^{k}_NX_{N/2}[2(k+N/2)+1]\\
=&X_{N/2}[2(k+N/2)] + \omega^{1}_{2}\omega^{k}_NX_{N/2}[2(k+N/2)+1]\\
=&X_{N/2}[2(k+N/2)] -\omega^{k}_NX_{N/2}[2(k+N/2)+1]\\
=&X[k+N/2]
\end{align*}
$$

这就是蝶形公式

$$
\begin{align*}
X[k] &= X_{N/2}[2k] + \omega^k_NX_{N/2}[2k+1]\\
X[k+N/2] &= X_{N/2}[2k] - \omega^{k}_NX_{N/2}[2k+1]\\
\end{align*}
$$

Radix-R 的一次分解公式为

$$
\begin{align*}
X[k] &= \sum_{r=0}^{R-1}\omega^{rk}_{N}X_{N/R}[Rk+r]\\
\end{align*}
$$

正向推导

$$
\begin{align*}
X[k+t\frac{N}{R}] &= \sum_{r=0}^{R-1}\omega^{r(k+t N/R)}_{N}X_{N/R}[R(k+t\frac{N}{R})+r]\\
&= \sum_{r=0}^{R-1}\omega^{rt N/R}_{N}\omega^{rk}_{N}X_{N/R}[Rk+r+tN]\\
&= \sum_{r=0}^{R-1}\omega^{rt}_{R}\omega^{rk}_{N}X_{N/R}[Rk+r]\\
\end{align*}
$$

得到 Radix-R 的一次分解的蝶形公式

$$
X[k+t\frac{N}{R}]= \sum_{r=0}^{R-1}\omega^{rt}_{R}\omega^{rk}_{N}X_{N/R}[Rk+r]\\
$$

$$
t \in \{0, 1, 2, \cdots, R-1\}
$$

在前有映射公式

$$
X_{N/P}[k] = \ _{R_n}f^{b}_{N}
\begin{pmatrix}
X_{N/(P\cdot R_n)}[(k + b\cdot(R_n-1) + P \cdot r_n)\text{ mod } N]
\end{pmatrix}\\\\
X_{N/P}[k] = \sum_{r=0}^{R_n-1}\omega_N^{r_nb}
X_{N/(P\cdot R_n)}[(k + b\cdot(R_n-1) + P \cdot r_n)\text{ mod } N]
\\\\
b = P \lfloor \frac{k}{P} \rfloor\\\\
P = \prod_{i=1}^{n-1}R_i\\\\
\{R_1,R_1,R_2\cdots R_n\}
$$

在之前推导的映射公式中，同一组Radix-N蝴蝶操作并不是被排列在一起，而是被分布开了，可以通过重新映射使其排列在一起

$k'$的Raidx操作序列
$$
u = k' \text{ mod } (N/P)\\
U = \lfloor\frac{k'}{N/P} \rfloor \\
k =P u + U
$$



但是这样做是十分低效的，此外在单点映射公式中，没有利用单位根的对称性质，实际上，利用蝶形公式和单位根的对称性，复数乘法的计算量可以减少一半

我们需要再重新推导一下 Radix-N 的蝶形公式

$N$ 含因子 $R_1, R_2, R_3, \cdots, R_n$

FFT 的分解公式为

$$
X_{N}[k] = \sum_{r_1=0}^{R_1-1}\omega^{r_1k}_{N}X_{N/R_1}[R_1k+r_1+s]\\
s \in \{0\}\\
$$

$$
X_{N/R_1}[R_1 k + s]= \sum_{r_2=0}^{R_2-1}\omega^{r_2k}_{N/R_1} X_{N/(R_1\cdot R_2)}[R_1R_2k+R_1r_2 + s]\\
s \in \{0,1,2,\cdots,R_1-1\}
$$

$$
X_{N/(R_1\cdot R_2)}[R_1R_2k + s]= \sum_{r_3=0}^{R_3-1}\omega^{r_3k}_{N/(R_1\cdot R_2)} X_{N/(R_1\cdot R_2\cdot R_3)}[R_1R_2R_3k+R_1R_2r_3 + s]\\
s \in \{0,1,2,\cdots,R_1R_2-1\}
$$

将其换为蝶形公式有

$$
X_{N}[k+t\frac{N}{R_1}] = \sum_{r_1=0}^{R_1-1}\omega^{r_1t}_{R_1}\omega^{r_1k}_{N}X_{N/R_1}[R_1k+r_1+s]\\
s \in \{0\}\\
k \in \{0,1,2,\cdots,\frac{N}{R_1}-1\}
$$

$$
\begin{align*}
X_{N/R_1}[R_1 k +t\frac{N}{R_2}+ s]&= X_{N/R_1}[R_1 (k +t\frac{N}{R_1 R_2})+ s]\\
&= \sum_{r_2=0}^{R_2-1}\omega^{r_2(k +t\frac{N}{R_1 R_2})}_{N/R_1} X_{N/(R_1\cdot R_2)}[R_1R_2(k +t\frac{N}{R_1 R_2})+R_1r_2 + s]\\ 
&= \sum_{r_2=0}^{R_2-1}\omega^{r_2(k +t\frac{N}{R_1 R_2})}_{N/R_1} X_{N/(R_1\cdot R_2)}[R_1R_2k +R_1r_2 + s+tN]\\ 
&= \sum_{r_2=0}^{R_2-1}\omega^{r_2t}_{R_2}\omega^{r_2k}_{N/R_1} X_{N/(R_1\cdot R_2)}[R_1R_2k +R_1r_2 + s]\\
\end{align*}\\
$$

$$
s \in \{0,1,2,\cdots,R_1-1\}\\
k \in \{0,1,2,\cdots,\frac{N}{R_1\cdot R_2}-1\}
$$

同理

$$
\begin{align*}
X_{N/(R_1\cdot R_2)}[R_1R_2 k +t\frac{N}{R_3}+ s]&= \sum_{r_3=0}^{R_3-1}\omega^{r_3t}_{R_3}\omega^{r_3k}_{N/(R_1\cdot R_2)} X_{N/(R_1\cdot R_2\cdot R_3)}[R_1R_2R_3k +R_1R_2r_3 + s]\\
\end{align*}\\
$$

$$
s \in \{0,1,2,\cdots,R_1R_2-1\}\\
k \in \{0,1,2,\cdots,\frac{N}{R_1\cdot R_2\cdot R_3}-1\}
$$

$$
\begin{align*}
X_{N/P}[P k +t\frac{N}{R_n}+ s]&= \sum_{r_n=0}^{R_n-1}\omega^{r_nt}_{R_n}\omega^{r_nk}_{N/P} X_{N/(P\cdot R_n)}[PR_nk +Pr_n + s]\\
\end{align*}\\
$$

$$
s \in \{0,1,2,\cdots,P-1\}\\
k \in \{0,1,2,\cdots,\frac{N}{P\cdot R_n}-1\}
$$

这里不需要再继续换元，我们已经得到了Radix-N蝶形公式

我们的目的是将 $N$ 分解为几个较大的Radix，再将这些Radix在内部分解为更小的Radix

接下来要开始套娃了，我们将对 $R_n$ 因子的分解再进行细分

其蝶形分解公式为

$$
X_{N/P}[P k +t\frac{N}{R_n}+ s]= \sum_{r_n=0}^{R_n-1}\omega^{r_nt}_{R_n}\omega^{r_nk}_{N/P} X_{N/(P\cdot R_n)}[PR_nk +Pr_n + s]\\
$$

$R_n$ 的质因子序列为 $\{R_n^1,R_n^2,R_n^3,\cdots,R_n^m\}$

在内部分解中 $k$ 与 $s$ 被视为常量，按 $t$ 与 $r_n$ 抽取，左右均被映射到 $[0,R_n-1]$ 得到

$$
X_{R_n}[t]= \sum_{r_n=0}^{R_n-1}\omega^{r_nk}_{N/P}\omega^{r_nt}_{R_n} X_{R_n/R_n}[r_n]\\
$$

这条公式与 DFT 公式 $X[k] = \sum_{n=0}^{N-1}\omega_N^{nk}x[n]$ 很像，但是它并不是 DTF，这里多出了 $\omega^{r_nk}_{N/P}$ 项

现在将得到的公式重新整理一下，记
$$
Y = X\\
N = R_n\\
n = r_n\\
k = t\\
C^n = \omega^{r_nk}_{N/P}\\
\{R_1,R_2,R_3,\cdots,R_m\} = \{R_n^1,R_n^2,R_n^3,\cdots,R_n^m\}\\
$$
转换后得到
$$
Y_{N}[k]= \sum_{n=0}^{N-1}C^n\omega^{nk}_{N} Y_{N/N}[n]\\
$$

写成类似于 DTF 形式

$$
Y[k]= \sum_{n=0}^{N-1}C^n\omega^{nk}_{N} y[n]\\
$$

再按照 FFT 公式推理的步骤，再推导一遍此公式的蝶形形式

$$
\begin{align*}
Y[k] &= \sum_{n=0}^{N/2-1}C^{2n} \omega^{2nk}_{N} y[2n] + \sum_{n=0}^{N/2-1}C^{2n+1} \omega^{(2n+1)k}_{N} y[2n+1]\\
&= \sum_{n=0}^{N/2-1}C^{2n} \omega^{2nk}_{N} y[2n] + C\omega^{k}_{N} \sum_{n=0}^{N/2-1}C^{2n} \omega^{2nk}_{N} y[2n+1]\\
&= \sum_{n=0}^{N/2-1}C^{n}_{1/2} \omega^{nk}_{N/2} y[2n] + C\omega^{k}_{N} \sum_{n=0}^{N/2-1}C^{n}_{1/2} \omega^{nk}_{N/2} y[2n+1]\\
\end{align*}
$$

接下来展开为四项

$$
\begin{align*}
&\sum_{n=0}^{N/2-1}C^{n}_{1/2} \omega^{nk}_{N/2} y[2n] + C\omega^{k}_{N} \sum_{n=0}^{N/2-1}C^{n}_{1/2} \omega^{nk}_{N/2} y[2n+1]\\
=& \sum_{n=0}^{N/4-1} C^{n}_{1/4} \omega^{nk}_{N/4} y[4n] + C_{1/2}\omega^{k}_{N/2} \sum_{n=0}^{N/4-1} C^{n}_{1/4} \omega^{nk}_{N/4} y[4n+2]\\ 
&+C\omega^{k}_{N}\begin{pmatrix}
\sum_{n=0}^{N/4-1} C^{n}_{1/4} \omega^{nk}_{N/4} y[4n+1] + C_{1/2}\omega^{k}_{N/2} \sum_{n=0}^{N/4-1} C^{n}_{1/4} \omega^{nk}_{N/4} y[4n+3]
\end{pmatrix}
\end{align*}
$$

写成循环数组形式

$$
Y_N[k] = Y_{N/2}[2k] + C^{1}_{1}\omega^{k}_{N} Y_{N/2}[2k+1]\\
$$

对于 Radix-R 有

$$
Y_{N}[k] = \sum_{r=0}^{R-1}C^{r}_{1} \omega^{rk}_{N} Y_{N/R}[Rk+r]\\
$$

记
$$
\ _R h^k_{N/P}(y_0,y_1,y_2,\cdots,y_{R-1}) = \sum_{r=0}^{R-1}C^{r}_{1/P} \omega^{rk}_{N/P} y_r\\
$$

有
$$
\begin{align*}
Y_{N/P}[k] &= \ _R h^k_{N/P}
\begin{pmatrix}
Y_{N/(P\cdot R)}[Rk+0],Y_{N/(P\cdot R)}[Rk+1],Y_{N/(P\cdot R)}[Rk+2],\cdots,Y_{N/(P\cdot R)}[Rk+R-1]
\end{pmatrix}\\
&= \ _R h^k_{N/P}\begin{pmatrix}Y_{N/(P\cdot R)}[Rk+r]\end{pmatrix}\\
\end{align*}
$$

对比之下，FFT 的递推式为

$$
\begin{align*}
X_{N/P}[k] 
&= \ _R f^k_{N/P}\begin{pmatrix}
X_{N/(P\cdot R)}[Rk+r]\end{pmatrix}\\
\end{align*}
$$

至此为止，此公式的递推形式以及与 FFT 公式的递推形式完全一致，所以无需向下推导，我们可以直接使用换元法得到此公式的蝶形形式。

根据 Radix-N FFT 的蝶形形式
$$
\begin{align*}
X_{N/P}[P k +t\frac{N}{R_n}+ s]&= \sum_{r_n=0}^{R_n-1}\omega^{r_nt}_{R_n}\omega^{r_nk}_{N/P} X_{N/(P\cdot R_n)}[PR_nk +Pr_n + s]\\
\end{align*}\\
$$

$$
s \in \{0,1,2,\cdots,P-1\}\\
k \in \{0,1,2,\cdots,\frac{N}{P\cdot R_n}-1\}
$$

将 $X$ 替换为 $Y$ , $\omega^{r_nk}_{N/P}$ 替换为 $C^{r_n}_{1/P} \omega^{r_nk}_{N/P}$ 得到

$$
\begin{align*}
Y_{N/P}[P k +t\frac{N}{R_n}+ s]&= \sum_{r_n=0}^{R_n-1}\omega^{r_nt}_{R_n}C^{r_n}_{1/P} \omega^{r_nk}_{N/P} Y_{N/(P\cdot R_n)}[PR_nk +Pr_n + s]\\
\end{align*}\\
N = R'\\
C^{r_n} = \omega^{r_nk'}_{N'/P'}\\
$$

$k'$ $N'$ $P'$ $R'$为外层中的参数


# 其它

除2以外的质数的蝶形公式中的共轭对称

$$
\begin{align*}
\omega^{rt}_{R} &= \ ^*\omega^{-rt}_{R}\\
\omega^{rk}_{N} &= \ ^*\omega^{-rk}_{N}\\
\end{align*}
$$