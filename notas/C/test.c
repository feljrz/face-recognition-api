#include <stdio.h>
#include <stdlib.h>

/*Calculadora*/


int main(){
    void *p;
    void *malloc(size_t);
    p = (int *) malloc(sizeof(int));
    printf("Tamanho de p: %ld ", sizeof(p));


return 0;
}
