#include <stdio.h>
#include <stdlib.h>

double generateRandom(int lower, int upper)
{
    double num;
    num = rand()%(upper - lower);
    return num + lower;
}

struct Instance {
    unsigned int capacity;
    unsigned int weight[10000];
    unsigned int price[10000];
};

struct Instance getInstance(int items, int maxVal){
    
    struct Instance Inst;
    double temp = 0.0;
    float WP [items];
    
    for (int i = 0; i < items; i++){
        double random = rand()%(maxVal - 1) + 1;
        Inst.weight[i] = random;
        temp += random;
        Inst.price[i] = random + 10;
        WP[i] = Inst.weight[i]/(float)Inst.price[i];
    }
    Inst.capacity = 0.5*temp;

    int i, j, a, b;
    float c;
    for (i = 0; i < items; i++){
        for (j = i + 1; j <items; j++){
            if (WP[i] > WP[j]){
                a = Inst.price[i];
                Inst.price[i] = Inst.price[j];
                Inst.price[j] = a;

                b = Inst.weight[i];
                Inst.weight[i] = Inst.weight[j];
                Inst.weight[j] = b;

                c = WP[i];
                WP[i] = WP[j];
                WP[j] = c;

            }
        }
    }
    // printf("The items in descending order of Weight/Price are:\n");
    // for (i = 0; i < items; ++i){
    //     printf("Weight: %d and Price: %d\n", Inst.weight[i], Inst.price[i]);
    // }
    
    return Inst;
    
}
