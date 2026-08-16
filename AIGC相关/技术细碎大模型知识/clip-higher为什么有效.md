# clip-higher 为什么有效

### Clip-Higher

$$
0.8<\frac{\pi_{\theta}}{\pi_{old}}<1.2,则 \pi_{\theta}<1.2\pi_{old}
$$

故而把上界提高，可以

故而当 $\pi_{old}=0.01$时， $\pi_{\theta}=0.012$;当 $\pi_{old}=0.9$时, $\pi_{\theta}=1.08$。

故而可以看到对于熵低的，其实上界范围相对较低，而对于熵高的，上界范围相对要高；故而把上界提高，可以留给低概率更多空间。

![](/wiki/AIGC%E7%9B%B8%E5%85%B3/%E6%8A%80%E6%9C%AF%E7%BB%86%E7%A2%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86/static/MWdLbvbbYohbLwxkzsqcL1wZnFh.png)
