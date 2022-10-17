import torch.nn as nn

# Define model using Sequential.
mnist_cnn = nn.Sequential(

    nn.Sequential(nn.Conv2d(1, 10, 3),
                  nn.ReLU(),
                  nn.MaxPool2d(2)
                  ),

    nn.Sequential(
        nn.Conv2d(10, 5, 3),
        nn.ReLU(),
        nn.Flatten()
    ),

    nn.Sequential(
        nn.Linear(5 * 11 * 11, 50),
        nn.ReLU()
    ),

    nn.Sequential(
        nn.Linear(50, 30),
        # nn.Dropout2d(0.2), # jangho
        nn.ReLU()
    ),

    nn.Sequential(
        nn.Linear(30, 10)
    )

)

### networks for long-tailed recognition
ncha = 1
size = 28

mlp_mnist = nn.Sequential(
    # fc1
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(ncha * size * size, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc2
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),
    # fc3
    nn.Sequential(
        nn.Linear(800, 800),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    ),

    # last: task 0 (0-4) & task 1 (5-9)
    nn.Sequential(
        nn.Linear(800, 10)
    ),
)