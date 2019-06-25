import numpy as np
import math as m

def sigmoid(x):#activation function
    #return 1 if (1/(1+m.exp(-x)))>=0.5 else 0
    return (1/(1+m.exp(-x)))
    
def neuron_output(a,b,n):#a is a list of weights, b is the list of inputs, n is the number of inputs
    add=0
    for i in range(0,n):
        add=add+a[i]*b[i]
    #bias=np.random.rand(0,3)
    bias=0
    return add-bias
    
def create_weight(num_of_inputs,num_of_nodes):
    wt=[]
    for i in range(num_of_nodes):
        wt[i]=np.random.rand(num_of_inputs)
    return wt
    
layer1=[1,2,3]#input layer
no_nodes_layer2=4
wt_layer2=create_weight(len(layer1),no_nodes_layer2)
layer2_output=[]
for i in range(no_nodes_layer2):
    layer2_output.append(sigmoid(neuron_output(layer1,wt_layer2[i],len(layer1))))

wt_layer3=create_weight(no_nodes_layer2,1)
layer3_output=sigmoid(neuron_output(layer2_output,wt_layer3[0],4))

actual_y=1

def back_prop(wt,prev_layer_output,diff,n):
    for i in range(n):
        z=sigmoid(prev_layer_output[i])
        wt[i] +=2*diff*z*(1-z)*prev_layer_output[i]
        
back_prop(wt_layer3,layer2_output,actual_y-y,1)
back_prop(np.dot(wt_layer2,layer1),layer1,actual_y-y,no_nodes_layer2)












































