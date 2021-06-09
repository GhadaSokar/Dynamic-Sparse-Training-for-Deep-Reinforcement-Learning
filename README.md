# Dynamic Sparse Training for Deep Reinforcement Learning

This is the Pytorch implementation for the [Dynamic Sparse Training for Deep Reinforcement Learning](https://arxiv.org/pdf/2106.04217.pdf) paper.

# Abstract
In this paper, we introduce dynamic sparse training for Deep Reinforcement Learning (DRL). In particular, we propose a new training algorithm to train DRL agents using sparse neural networks from scratch and dynamically optimize the sparse topology jointly with the parameters. We integrated our proposed method with the [Twin Delayed Deep Deterministic policy gradient (TD3)](https://arxiv.org/abs/1802.09477) algorithm and introduce Dynamic Sparse training for TD3 (DS-TD3). 

Our proposed method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).
The experimental results show the effectiveness of our training algorithm in boosting the learning speed of the agent and achieving higher performance. Moreover, DS-TD3 offers a 50% reduction in the network size and floating-point operations (FLOPs).

# Requirements
* Python 3.8
* PyTorch 1.5
* [Mujoco-py](https://github.com/openai/mujoco-py) 
* OpenAI gym

# Usage

For DS-TD3: Dynamic Sparse training of TD3 algorithm 
```
python main.py --env HalfCheetah-v3 --policy DS-TD3
```

For Static-TD3
```
python main.py --env HalfCheetah-v3 --policy StaticSparseTD3
```

For TD3
```
python main.py --env HalfCheetah-v3 --policy TD3
```

# Results

# Reference 

If you use this code, please cite our paper:
```
 @article{sokar2021dynamic,
  title={Dynamic Sparse Training for Deep Reinforcement Learning},
  author={Sokar, Ghada and Mocanu, Elena and Mocanu, Decebal Constantin and Pechenizkiy, Mykola and Stone, Peter},
  journal={arXiv preprint arXiv:2106.04217},
  year={2021}
}
```

# Acknowledgments
We start from the official code of the TD3 method from the following repository

[TD3](https://github.com/sfujim/TD3)
