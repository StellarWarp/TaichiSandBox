## 蝶形公式

直接将两个Raidx-2 DFT 替换成 Radix-4 DFT 或更高阶的 DFT 会增加运算量，反而会使得 FFT 变慢，现在要做的是把 Radix-N 在内部重新分解成更小的 Raidx 进行计算，简单来说就是把 FFT 的完整展开硬编码到代码中，这样就可以减少运算量，提高速度

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

这就是蝶形公式（这其实可以看作把循环数组表示的展开式取模）

$$
\begin{align*}
X[k] &= X_{N/2}[2k] + \omega^k_NX_{N/2}[2k+1]\\
X[k+N/2] &= X_{N/2}[2k] - \omega^{k}_NX_{N/2}[2k+1]\\
\end{align*}
$$

Radix-R 的分解公式为

$$
\begin{align*}
X[k] &= \sum_{r=0}^{R-1}\omega^{rk}_{N}X_{N/R}[Rk+r]\\
\end{align*}
$$

这次正向推导

$$
\begin{align*}
X[k+t\frac{N}{R}] &= \sum_{r=0}^{R-1}\omega^{r(k+t N/R)}_{N}X_{N/R}[R(k+t\frac{N}{R})+r]\\
&= \sum_{r=0}^{R-1}\omega^{rt N/R}_{N}\omega^{rk}_{N}X_{N/R}[Rk+r+tN]\\
&= \sum_{r=0}^{R-1}\omega^{rt}_{R}\omega^{rk}_{N}X_{N/R}[Rk+r]\\
\end{align*}
$$

即 Radix-R 的蝶形公式为

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

在之前推导的映射公式中，同一组Radix-N蝴蝶操作并不是被排列在一起，而是被分布开了

$$
u = k' \text{ mod } (N/P)\\
U = \lfloor\frac{k'}{N/P} \rfloor \\
k =P u + U
$$


$k'$的Raidx操作序列就被排在了一起

或者

我们需要再重新推导一下蝶形公式

令 $s = k \text{ mod } P$
$\text{Radix } R_1$ 的分解公式为


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


除2以外的质数的蝶形公式中的共轭对称

$$
\begin{align*}
\omega^{rt}_{R} &= \ ^*\omega^{-rt}_{R}\\
\omega^{rk}_{N} &= \ ^*\omega^{-rk}_{N}\\
\end{align*}
$$