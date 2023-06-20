# FFT 的数据单点映射

我们将 FFT 的递推公式展开，可以得到

$$
\begin{align*}
X[k] 
&= \sum_{n=0}^{N/2-1}x[2n]\omega_{N/2}^{nk} + \omega_N^k\sum_{n=0}^{N/2-1}x[2n+1]\omega_{N/2}^{nk}\\
&= \sum_{n=0}^{N/4-1}x[4n]\omega_{N/4}^{nk} + \omega_N^k\sum_{n=0}^{N/4-1}x[4n+2]\omega_{N/4}^{nk} \\&+ \omega_N^k\left( \sum_{n=0}^{N/4-1}x[4n+1]\omega_{N/4}^{nk} + \omega_N^k\sum_{n=0}^{N/4-1}x[4n+3]\omega_{N/4}^{nk} \right)\\
\end{align*}
$$

进行一次 FFT 共需要 $L=\log_2N$ 层并行运算，每一层都是将 $N$ 个点分成两组，然后将这两组点合并成一个 $N/2$ 点的DFT

将第 $l$ 层的 $N$ 长度的向量表示为 $X_{N/2^{L-l}}$，那么 $L$ 层为 $X_{N}$ ，第 $L-1$ 层为 $X_{N/2}$, ……，第 $0$ 层为 $X_{N/N} = x$

这里的 $X_{N/2}$ 并不是长度为 $N/2$ 的向量，$X_{N/2}$ 的长度依然为 $N$ 

这里引入循环数组 $X[k+N]=X[k]$ 因为在傅里叶变换中，数值是周期延拓的，只要在最后进行取模运算即可

根据上面展开的公式，可以得到
在第$\log_2N$ 层的映射关系
$$
X_N[k]= X_{N/2}[2k] + \omega_N^kX_{N/2}[2k+1]
$$

根据 FFT 的递推公式
$$
X[k] = \sum_{n=0}^{N/2-1}x[2n]\omega_{N/2}^{nk} + \omega_N^k\sum_{n=0}^{N/2-1}x[2n+1]\omega_{N/2}^{nk}\\
$$

将 $2k$ 中的 $k$ 换元为 $2k$ 与 $2k + 1$ 

得到在第$L-1$ 层的映射关系

$$
\begin{align*}
X_{N/2}[2k] &= X_{N/4}[4k] + \omega_{N/2}^{k}X_{N/4}[4k+2]\\
X_{N/2}[2k+1] &= X_{N/4}[4k+1] + \omega_{N/2}^{k}X_{N/4}[4k+3]\\
\end{align*}
$$

将这两个公式翻译一下即：给定在数组 $X_{N/2}$ 中的编号 $k$ , 欲求得 $X_{N/2}[2k]$ ，则需寻 $X_{N/4}$ 中 $4k$ 与 $4k+2$ 的值 ……

再将两条公式合并得到一条公式
先将公式写为分段函数的形式，这里需要将原式中的 $k$ 均换元为 $k'$ 然后分别令 $2k = k'$ $2k+1=k'$ 得到
$$
X_{N/2}[k] = \begin{cases}
X_{N/4}[2k]   + \omega^{k/2}_{N/2}X_{N/4}[2k+2] & k \text{ mod } 2 = 0\\
X_{N/4}[2k-1] + \omega^{(k-1)/2}_{N/2}X_{N/4}[2k+1] & k \text{ mod } 2 = 1\\
\end{cases}
$$

再进行合并

$$
X_{N/2}[k] = 
X_{N/4}[2k - k \text{ mod } 2] 
+\omega^{k-k \text{ mod } 2}_{N/2}
X_{N/4}[2k + 2 - k \text{ mod } 2]
$$

继续递推

$$
\begin{align*}
X_{N/4}[4k]   &= X_{N/8}[8k]   &+ \omega^{k}_{N/4}X_{N/8}[8k+4]\\
X_{N/4}[4k+1] &= X_{N/8}[8k+1] &+ \omega^{k}_{N/4}X_{N/8}[8k+5]\\
X_{N/4}[4k+2] &= X_{N/8}[8k+2] &+ \omega^{k}_{N/4}X_{N/8}[8k+6]\\
X_{N/4}[4k+3] &= X_{N/8}[8k+3] &+ \omega^{k}_{N/4}X_{N/8}[8k+7]\\
\end{align*}
$$

写为分段函数

$$
X_{N/4}[k] = \begin{cases}
X_{N/8}[2k]   + \omega^{(k  )/4}_{N/4}X_{N/8}[2k+4] & k \text{ mod } 4 = 0\\
X_{N/8}[2k-1] + \omega^{(k-1)/4}_{N/4}X_{N/8}[2k+3] & k \text{ mod } 4 = 1\\
X_{N/8}[2k-2] + \omega^{(k-2)/4}_{N/4}X_{N/8}[2k+2] & k \text{ mod } 4 = 2\\
X_{N/8}[2k-3] + \omega^{(k-3)/4}_{N/4}X_{N/8}[2k+1] & k \text{ mod } 4 = 3\\
\end{cases}
$$

合并

$$
X_{N/4}[k] = 
X_{N/8}[2k - k \text{ mod } 4] + 
\omega^{(k-k \text{ mod } 4)/4}_{N/4}
X_{N/8}[2k + 4 - k \text{ mod } 4]
$$

最后可以总结出一般的映射公式

记 $\overline{l} = L- l$ , $\overline{l} \in\{ 0,1 ,\cdots,L-1\}$

$$
X_{N/2^{\overline{l}}}[k] = 
X_{N/2^{\overline{l}+1}}[2k - k \text{ mod } 2^{\overline{l}}] 
+\omega^{(k-k \text{ mod } 2^{\overline{l}})/2^{\overline{l}}}_{N/2^{\overline{l}}}
X_{N/2^{\overline{l}+1}}[2k + 2^{\overline{l}} - k \text{ mod } 2^{\overline{l}}]
$$

鉴于 $X_{N/2^{\overline{l}}}$ 的表述方式过于复杂，且实际计算中也不会出现，记 $X_{N/2^{\overline{l}}}$ 为 $X_{l}$ , 表示在第 $l$ 次 FFT 运算中的写入对象

此外 $\omega^{(k-k \text{ mod } 2^{\overline{l}})/2^{\overline{l}}}_{N/2^{\overline{l}}}$ 可以简化为 $\omega^{k-k \text{ mod } 2^{\overline{l}}}_{N}$

上述公式可以简化为
$$
X_{l}[k] = X_{l-1}[2k - k \text{ mod } 2^{\overline{l}}] 
+\omega^{k-k \text{ mod } 2^{\overline{l}}}_{N}
X_{l-1}[2k + 2^{\overline{l}} - k \text{ mod } 2^{\overline{l}}]
$$

最后进行取模，从循环数组转换到实际计算中的数组

$$
X_{l}[k] = X_{l-1}[(2k - k \text{ mod } 2^{\overline{l}})\text{ mod }N] 
+\omega^{k-k \text{ mod } 2^{\overline{l}}}_{N/2^{\overline{l}}}
X_{l-1}[(2k + 2^{\overline{l}} - k \text{ mod } 2^{\overline{l}})\text{ mod }N]
$$

其中 $2k - k \text{ mod } 2^{\overline{l}}$ 还有另一种形式的表述

利用 $m \text{ mod } n = m - n \lfloor \frac{m}{n} \rfloor$

$$
\begin{align*}
2k - k\text{ mod } 2^{\overline{l}} &= 2k - (k - 2^{\overline{l}}\lfloor \frac{k}{2^{\overline{l}}} \rfloor)\\
&= 2^{\overline{l}}\lfloor \frac{k}{2^{\overline{l}}} \rfloor + k\\
\end{align*}
$$

在实际的计算中是两次移位运算、一次加法运算，相较 $2k - k \text{ mod } 2^{\overline{l}}$ (一次移位运行、两次加法运算、一次与运算) 更快

同理 $k - k\text{ mod } 2^{\overline{l}} $ 也可以写为
$$
\begin{align*}
k - k\text{ mod } 2^{\overline{l}} 
&= 2^{\overline{l}}\lfloor \frac{k}{2^{\overline{l}}} \rfloor\\
\end{align*}
$$




# FFT 的优化

通过虚部补零的复数 FFT 因为虚部都是 0 所以浪费了一半的信息。
为了更好的利用信号，可以将两个实数信号合并成一个复数信号，然后进行复数 FFT，最后再分离出两个实数信号。

$$
\begin{align*}
&x[n],y[n] \in \mathbb{R}\\
z[n] &= x[n] + iy[n]\\
Z[k] &= DFT(z[n]) = \sum_{n=0}^{N-1}z[n]\omega_N^{nk}\\
&= \sum_{n=0}^{N-1}(x[n] + iy[n])\omega_N^{nk}\\
&= \sum_{n=0}^{N-1}x[n]\omega_N^{nk} + i\sum_{n=0}^{N-1}y[n]\omega_N^{nk}\\
&= X[k] + iY[k]\\
\end{align*}
$$

看似是两个信号被混在一起了，但在运算前后，没有冗余的信息，信息量不变，那么就有办法得到里面的信息

对于实数信号，在 DFT 变换后有性质 $X[n] = X^*[N-n]$ 注意这里是 $N-n$，$X[N]=X[0]$
接下来就可以利用这个性质来分离出两个实数信号

$$
\begin{align*}
X[k] &= \frac{1}{2}(Z[k] + Z^*[N-k])\\
Y[k] &= -\frac{1}{2i}(Z[k] - Z^*[N-k])\\
\end{align*}
$$


对于二维 FFT 同理

$$
\begin{align*}
&x[m,n],y[m,n] \in \mathbb{R}\\
z[m,n] &= x[m,n] + iy[m,n]\\
Z[k,l] &= DFT(z[m,n]) = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1}z[m,n]\omega_M^{mk}\omega_N^{nl}\\
&= \sum_{m=0}^{M-1}\sum_{n=0}^{N-1}(x[m,n] + iy[m,n])\omega_M^{mk}\omega_N^{nl}\\
&= \sum_{m=0}^{M-1}\sum_{n=0}^{N-1}x[m,n]\omega_M^{mk}\omega_N^{nl} + i\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}y[m,n]\omega_M^{mk}\omega_N^{nl}\\
&= X[k,l] + iY[k,l]\\
X[k,l] &= \frac{1}{2}(Z[k,l] + Z^*[M-k,N-l])\\
Y[k,l] &= -\frac{1}{2i}(Z[k,l] - Z^*[M-k,N-l])\\
\end{align*}
$$


同理可以推广到多维 FFT

我们几乎不增加多少计算量就可以在一次 FFT 中得到两个实数信号的 FFT，这样就可以减少一半的计算量。