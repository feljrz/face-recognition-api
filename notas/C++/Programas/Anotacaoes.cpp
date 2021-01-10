#include<iostream>
#include<stdio.h>
#include<stdlib.h>

//############## Functions #############

/*Function propotype*/
int show(int, int); 

/*Overload*/
int show(int x, double y){ 
    printf("2)%d e %f\n", x, y);
}

/*Default (after set a default variable, the others ahead need to be default)*/ 
int show(int x=10, float y=20){
    printf("3)%d e %f\n", x, y);
}

char show(char x[]){
    printf("4)%s\n", x);
}

/*Passing value for reference*/
void swap(int &x, int &y){
    int temp;
    temp = x;
    x = y;
    y = temp;
}

/*Pointer to function*/

void swap(int *x, int *y){
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}



int main(){
    int a, b, aux = 1;
    show(2,3);
    show(2,3.0);
    show(2);
    show("Felipinho");
    a = 2, b = 3;
    swap(a, b);
    printf("a=%i b=%i \n", a, b);

    //############## Pointers #############
    int *ptrX, y; //y is a variable
    int *ptrK = NULL;
    int *ptrJ, arr[10] = {1, 2, 3, 4, 5, 6, 7};
    double *ptrDouble;
 
    ptrDouble = new double; //Allocate memory of a variable
    delete ptrDouble; //Dealocate memory

    


    y = 10;
    printf("Valor do ponteiro: %p\n", ptrX);
    ptrX = &y; //& - Passing the address of y
    printf("Valor do ponteiro: %p | Valor do end de y: %x\n", ptrX, &y);
    printf("Valor do conteúdo do ptrX: %i\n", *ptrX);
    ptrK = new int;
    *ptrK = 35;
    swap(ptrX, ptrK);
    printf("*ptrX=%i *ptrK=%i \n", *ptrX, *ptrK);

    ptrJ = arr; //Pointer to the first address of the array (ptrJ = &arr[0])
    for(int i=0; i<=10; i++){
        printf("ptrX[%i] = %i Array[%i] = %i\n", i, *(ptrJ + i), i, arr[i]);
        if (*(ptrJ + i) == 0){
            printf("INN");
            *ptrJ += 1;
        }
    }

    return 0;
}

int show(int x, int y){
    printf("1)%d e %i\n", x, y);
}
