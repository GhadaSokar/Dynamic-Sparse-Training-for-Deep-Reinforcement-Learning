# Dynamic Sparse Training for Deep Reinforcement Learning

This is the Pytorch implementation for the paper Dynamic Sparse Training for Deep Reinforcement Learning
Method is tested on [MuJoCo] (http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).

# Requirements
* Python 3.8
* PyTorch 1.5
* [Mujoco-py](https://github.com/openai/mujoco-py) 
* OpenAI gym

#Usage

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

# Acknowledgments
We start from the official code of the TD3 method from the following repository

[TD3](https://github.com/sfujim/TD3)
