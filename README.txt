Author：Inki
Contact：inki.yinji@qq.com
Parameters that can be modified:
1) po_label = 0 --> range: [0..9]
The number denotes that the main class is $c$-th class of MNIST.
2) mnist_path = "..."
The path of saved MNIST data.
3) loops = 5
The 5 times 10-cv, others you can set.
Data set description:
If you set po_label to $c \in [0..9]$, the $c$-th class will be set as the positive bags, others as the negative bags.
More parameters you could see original code.
Copyright: None.
Times: 20210512