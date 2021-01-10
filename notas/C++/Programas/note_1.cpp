#include <iostream>  //Manipulate flux around the system 
#include <stdio.h> //This library had several I/O functions 
#include <stdlib.h> //Has several funcionts like (malloc, free, realoc) see more: https://www.ime.usp.br/~pf/algoritmos/apend/stdlib.h.html
using namespace std;

/*Funções*/

int main(void){

  
        
    /*  //############ I/O ###########
    /*Along data types we have modifiers: signed, unsigned, short, long*/
    bool boolean = false;
    
    short x; // 2 Bytes 
    int  i, j, k, oper; //2-4 Byts (Also signed int)
    unsigned int c = 0; //4 Bytes (Can oly stor positive integers)
    long m; //at least 4 Bytes
    float u = 4.75; //4 Bytes
    double v = -0.55E10; //8 Bytes
    
    char letra = 'F'; //1 Byte
    char nome_completo[30];
    char ex1_string[] = "Felipe";
    char ex2_string[] = {'R', 'e', 'z', 'e', 'n', 'd', 'e'};

    int vetor[] = {1, 20, 22, 33, 44, 12, 21};
    int temp_arr[5];
    
    //############ I/O ###########
    /* cout << "Most basic way to print";
    cin >> m; //Recieve from user

    printf("Inteiro: %i, String: %s, Char: %c, Float: %f");
    scanf("%144[^\n]",nome_completo); //Restrict scanf to read until '\n' was pressed with MAX 144 characters
    
    gets() //Allow to catch a string with spaces, but has problems beacause can cause overflow
    fgets(ex1_string, 10, stdin) //Restrict gets function parametesrs (variable, size, input) 'stdin' -> keyboard

    for(int i=0; nome_completo[i]!='\0'; i++){
        printf("%c", nome_completo[i]);
    
    }
    printf("\n\n");

    while(nome_completo[c]!='\0'){
        printf("%c", nome_completo[c]);
        c++;
    }

    printf("\n\n");*/

    //############ Conversion ###########

    printf(" M = %i U = %f", m, u);
    /*Implicit conversion*/
    m = u;
    printf("\n int: M = %i e float: U = %f", m, u);

    /*Explicit conversion (type cast)*/
    u = (float) m;
    printf("\n int: M = %i e float: u =  %f", m, u);
    
    printf("\n\n");

    //########## Operators ################
    //Has precedence: !, (), *, /, +, -, %

    /*Assignment Operators*/
    x += 20; //x = x + 20
    x *= m; //x = x * m 

    /*Relational Operators*/
    //>>, <<, ==, !=, >=, <= 

    /*Logical Operators*/
    //and: &&, or: ||, not: !

    /*Binary Operators*/
    //and: &&, or: ||, xor: ^, One's complement:  ~, Shift Left: <<, Shift Right: >>



    //########## Statements ################

    if (x < 79){
        printf("\nX is less than 79");
    }
    else if (x == 80){
        printf("\nx is 80");
    }

    /*Ternary if (ask: This condition is true)*/
    (m != 3) ? (k = 10) : (k = 11);
    printf("\nValor de k: %i", k);

    /*Switch-case*/
    printf("\nType a number: \n");
    scanf("%i", &oper);
    fflush(stdin);
    switch (oper){
        case 1:
            printf("You type 1\n");
            break;

        case 2:
            printf("You type 2\n");
            break;

        default:
            printf("Your type is out of range: %i\n", oper);

    }
    

    //########## Loops ################

    for(int i=0; i<10; i++){
        if(i > 4){
            break; //stop loop
        }
        printf("%i", i);
    }

    /*Range based for loop (variable : collection)*/
    for(int n : vetor){
        printf("%i, ", n);
    }
    /*While*/
    j = 0;
    while(j < 5){
        temp_arr[j] = j;
        j++;
    }
    
    /*do.. while*/
    j = 0;
    do{
        printf("%i", temp_arr[j]);
        j++;
    }while(j<=5);

    /*goto*/
    printf("\nInforme números positivos para soma: \n");
    int soma = 0;
    while(true){
        int temp;
        scanf("%i", temp);
        fflush(stdin);
        if (temp < 0){
            goto jump; //ATENCAAO
        }
        soma += temp;
    }
    jump:
    
    return 0;   

}