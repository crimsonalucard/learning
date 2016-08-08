#!/usr/bin/env python

from neural_net.neural_net import create_neural_net
#from trainer.trainer import create_forward_propopater 
from trainer.trainer import create_forward_propopater

x = create_neural_net([[0,0],[0,0]],[[0,0],[0,0]])
print(x(1,1))
