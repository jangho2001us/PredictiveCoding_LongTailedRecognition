# Predictive Coding based Long-tailed Recognition in PyTorch

Predictive coding networks ([paper](https://arxiv.org/abs/2106.13082), [code](https://github.com/RobertRosenbaum/Torch2PC)) based implementation of Long-tailed Recognition in PyTorch. 

## Predictive Coding Algorithm

Inspired by the human brain, a predictive coding algorithm was introduced to resolve the biological limitation of backpropagation. Contrary to the neural plasticity of the human brain, the backpropagation algorithm performs global error-guided learning. However, in predictive coding, it performs local learning because its learning is performed with local error nodes in addition to the global error node. It has been demonstrated that an arbitrary computational graph can be trained in a predictive coding manner.

## Long-tailed Recognition

The dataset widely used in deep learning is class-wise balanced, where each class comprises the same or a similar number of samples. However, collecting a balanced dataset is difficult, and real-world datasets often show highly imbalanced distribution. Therefore, it is important to develop an effective solution to deal with imbalance is important.

## Training with Backpropagation

Please note that the training code is here just for demonstration purposes. 

To train the Protonet on this task, cd into this repo's `src` root folder and execute:

    $ python mnist_train_bp.py

The script takes the following command line options:

- `root_data`: the root directory where tha dataset is stored, default to `'./data'`

- `root_model`: the root directory where tha trained model is stored, default to `'./checkpoint'`

- `root_log`: the root directory where tha trained model is stored, default to `'./exp_lt'`

- `dataset`: the dataset used for the experiment, default to `'mnist'`

- `arch`: the network architecture used for the experiment, default to `'mlp'`

- `loss_type`: the learning objective used for the experiment, default to `'CE'`

- `imb_type`: the type of imbalance used for the experiment, default to `'exp'`

- `imb_factor`: the imbalanced degree used for the experiment, default to `'0.01'`

- `learning_rate`: the imbalanced degree used for the experiment, default to `'0.1'`

Running the command without arguments will train the models with the default hyperparamters values (producing results shown above).



## Training with Predictive Coding

Please note that the training code is here just for demonstration purposes. 

To train the predictive coding version of Protonet on this task with predictive coding manner, cd into this repo's `src` root folder and execute:

    $ python mnist_train_pc.py --error_type FixedPred --eta 0.1 --num_iter 20

- `error_type`: parameter update protocol of predictive coding algorithm, default to `FixedPred`

- `eta`: weight learning rate of predictive coding algorithm, default to `0.5`

- `num_iter`: the repetition number of backward iteration , default to `20`

The properties of other parameters are the same as backpropagation-based learning.


## .bib citation
cite the paper as follows (copied-pasted it from arxiv for you):
    
    @article{rosenbaum2022relationship,
      title={On the relationship between predictive coding and backpropagation},
      author={Rosenbaum, Robert},
      journal={Plos one},
      volume={17},
      number={3},
      pages={e0266102},
      year={2022},
      publisher={Public Library of Science San Francisco, CA USA}
    }


## License

This project is licensed under the MIT License

Copyright (c) 2018 Daniele E. Ciriello, Orobix Srl (www.orobix.com).