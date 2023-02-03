#include "darknet.h"
#include "convolutional_layer.h"
#include <stdio.h>
#include "nnpack.h"



int run_test(int argc, char **argv)
{
    //test_convolutional_layer();
    convolutional_layer l = make_convolutional_layer(1, 8, 8, 1, 1, 5, 3, 1, 0, STAIR, 1, 0, 0, 0);

    l.batch_normalize = 0;
    float data[] = {
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,

        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0
    };

    float kernel[] = {
        0,1,0,
        0,1,0,
        0,1,0
    };


    l.weights = kernel;
    //l.biases = 0;
        
    network net = *make_network(1);
    net.outputs = l.outputs;
    net.input = data;
   // forward_convolutional_layer(l , net);
    forward_convolutional_layer_nnpack(l , net);
    
    printOut(net, l);

    printf("Hello, World!\n");

    return 0;
}



void printOut(network net, convolutional_layer l){
    int i;
    int step = l.out_h;
    printf("Outputs: %d \n", net.outputs);

    for (i = 0; i < net.outputs; i+=step){
        for(int j = 0; j < step; j++){
            printf("%f ", l.output[i+j]);
           // printf("%f ", net.output[i+j]);
            
        }
        printf("\n");
    }
}