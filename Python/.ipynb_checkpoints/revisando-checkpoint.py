#Revisando Python

#PEP8
"""
-4 espaços para identação
-Utilizar 2 linhas em branco para separar funções e classes
-Utilizar CameCase em nome de classe
-Utilizar letra minúscula e uderline em funções
-Imports deve ser feito separadamente
-Em caso de muitos imports -> from lib import (Import_1, Import_2,...)
-Termine sempre com uma linha em branco"""


#Dir e Help
"""
-dir([object]) -> Exibe os métodos disponíveis para o objeto
-help([object]) -> Exibe a documentação, pode ser invocado em um método sem parênteses"""

#Input e output
"""
-input('String') -> Entrada de dados, pode-se fazer casting
x = int(input())
nome = string(input())

-print(sep, flush, end, file)
print(f'Eu me chamo {nome}' e tenho {anos}) -> Tipo mais atual 3.7
print('Eu me chamo {0} e tenho {1}'.format((nome, anos))
"""

# Tipos de dados
"""
Tipagem dinâmica, ou seja o tipo da variável é definido na atribuição
type() -> Exibe o tipo de dado
x, y = 10, 20 -> Dupla atribuição
"""

# Numérico
"""
x = 100_000_000 -> Mesmo valor
1000000000

5 //= 2 -> Inteiro da divisão "arredonda"
2

float_number = 14.44
complex_number = 10j
"""
#String
"""
Pode ser representado por aspas simples, duplas, ou triplas
texto = 'banana de pijamas'
texto.split() -> Transforma a string em uma lista, sep default  é '  '
texto[::1] -> Slice de string; O último parâmetro indica o passo
texto[::-1] -> Inversão da string
texto.replace('B', 'Q') -> Substituí na string
texto.lower() -> Passa o texto para minúsculo
texto.upper() -> Passar o texto para maiúsculo
texto.title() -> Transforma em título, colocando as inicias em maiúsculo

"""
#Estruturas condicionais
"""if; else; elif(else if); switch case statement:
def agua():
    print("Água é vida")

def coca(x):
    print(f"Coca-Cola é {x}")

option_dict = {0: agua, 1: coca}
option_dict[1]('ruim')
"""
#Operadores
"""
-Operador unário: not
-Operador binário: and, or, is

"""
()
#Loops
"""
-Iteráveis:
enumerate() (Cria uma tupla (índice, elemento)), range(), string, dictionary

for _, value in enumerate(text): -> '_' indica que o valor está sendo descartado

while x < 10: Obs: Não existe o do while
    break
"""

#Coleções

"""
-Listas
Funcionam como arrays de outras linguagens porém são dinâmicas e aceitam QUALQUER tipo de dado, inclusive variáveis e funções
Lembrar que a posição começa em 0, porém o inverso começa em -1
lista = ['Água', 1, (1, 3, 5), [9, 8, 56]] OU list(range(1, 8, 2))
lista.sort() -> Ordena a lista; Mais rápido que muitos sorts à mão
lista.append() -> Concatena um objeto
lista.extend([iterável]) -> Concatena elementos um a um no final
lista.insert(pos, value) -> Insere um novo valor deslocando o antigo à direita
lista.index(object, start, end) -> Retorna o índice do objeto na lista
lista.reverse() OU lista[::-1]-> Inverte a lista
lista.pop() -> Retorna o último elemento da lista
string = '  '.join(lista) -> Transforma a lista em uma string

len(lista) -> Retorna o número de elementos da lista


num1, num2, num3, = 44, 55, 66
lista = ['Água', 1, (1, 3, 5), [9, 22, 55, 8, 56]]
lista_2 = list(range(0, 10, 2))
lista_3 = ['Felipe', 'Junio', 'Rezende']
lista_4 = [num1, num2, num3]
#print(f"Lista 1: {lista[3][2]} Lista 2: {lista_2}")
#print(f"Tamanho 1: {len(lista[3])} Tamanho 2:{len(lista)} ")
x = '  '.join(lista_3)
def funcao():
    return 'Dentro da funcão'
lista_4.append(funcao)
print(f"Teste: {lista_4.index(44)} Tipo: {type(lista[2])}")
"""

#Tuplas
"""


"""

dicionario = {1: 'x', 2: 'y'}
tupla = ("Arroz", 1, 21, 14.7, dicionario)
tupla2 = 11, "Feijão" #Não recomendado
tupla_unitaria = ("Salada",)
tupla_aninhada = ("Carne", [5, 22, 43, 22], (1, 21, 33))



print(f'Tupla:\n{tupla}\n')
print(f'Tupla unitaria:\n{tupla_unitaria[0][1:3]}')
print(f'Tupla aninhada:\n{tupla_aninhada[0][2]}, \n {tupla_aninhada[2][3]}')

a, b, c, d, e = tupla
print(f'A: {a} B:{b} C:{c} D:{d} E:{e}')








