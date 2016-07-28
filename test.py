from neural_net.neural_net import create_neural_net
from trainer import forward_propagate 

x = create_neural_net([[0,0],[0,0]],[[0,0],[0,0]])
print(x(1,1))
