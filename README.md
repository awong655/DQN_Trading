# DQN-Trading

This is a framework based on deep reinforcement learning for stock market trading. This project is the implementation
code for the two papers:

- [Learning financial asset-specific trading rules via deep reinforcement learning](https://arxiv.org/abs/2010.14194)
- [A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules](https://arxiv.org/abs/2101.03867)

The deep reinforcement learning algorithm used here is Deep Q-Learning.

## Acknowledgement

- [Deep Q-Learning tutorial in pytorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- This code is heavily derived from https://github.com/MehranTaghian/DQN-Trading

## Requirements

Install pytorch using the following commands. This is for CUDA 11.1 and python 3.8:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- python = 3.8
- pandas = 1.3.2
- numpy = 1.21.2
- matplotlib = 3.4.3
- cython = 0.29.24
- scikit-learn = 0.24.2
- TA-Lib = 0.4.19

## To Run
I suggest installing using conda if on Windows. Could not install TA-Lib using pip.

Run Main_2.py. To perform different experiments, you must comment different parts of the code to make sure that only the models you want to run are uncommented. 
Areas to uncomment/comment: self.test_portfolios (Line 171), in the load_agents() function change the STATE_MODE parameter of each model to select bewteen the state modes provided in Dataloader/DataAutoPatternExtractionAgent.py. 

## References

```
@article{taghian2020learning,
  title={Learning financial asset-specific trading rules via deep reinforcement learning},
  author={Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza},
  journal={arXiv preprint arXiv:2010.14194},
  year={2020}
}

@article{taghian2021reinforcement,
  title={A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules},
  author={Taghian, Mehran and Asadi, Ahmad and Safabakhsh, Reza},
  journal={arXiv preprint arXiv:2101.03867},
  year={2021}
}
```
