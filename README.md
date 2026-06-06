## NanoGPT
Inspired and created from karpathy/ng-video-lecture

#### Fixed Hyperparameters
block_size = 256
n_embd = 64
n_head = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
batch_size = 64
max_iters = 5000
eval_interval = 500
eval_iters = 200

All experiments run on Macbook. device is cpu.
