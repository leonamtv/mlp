from core.MLP import MLP
from random import shuffle

EPOCHS = 2000

mlp = MLP(3, 5, 2, 0.05)

dataset = [
    ([ 0, 0, 0 ], [ 1, 1 ]),
    ([ 0, 0, 1 ], [ 0, 1 ]),
    ([ 0, 1, 0 ], [ 0, 1 ]),
    ([ 0, 1, 1 ], [ 0, 1 ]),
    ([ 1, 0, 0 ], [ 1, 0 ]),
    ([ 1, 0, 1 ], [ 0, 1 ]),
    ([ 1, 1, 0 ], [ 1, 0 ]),
    ([ 1, 1, 1 ], [ 0, 1 ])
]

for i in range ( EPOCHS ) :
    erroAproxEpoca = 0
    erroClassEpoca = 0
    data = dataset
    shuffle ( data )
    for sample in data :
        erro_aprox, erro_class = mlp.treinar ( sample[0], sample[1] )
        erroAproxEpoca += erro_aprox
        erroClassEpoca += erro_class
    print(f"Época {i + 1} \t\t| Erro aprox: { erroAproxEpoca } \t\t| Erro class: { erroClassEpoca } ")