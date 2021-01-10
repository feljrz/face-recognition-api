def Lampada:
    pass

def Produto:
    #Atributo de Classe
    imposto = 1.75
    contador = 0
    def __init__(self, nome, valor):
        self.id = Produto.contador +1
        self.nome = nome
        self.valor = (valor * Produto.Imposto)
        self.nome = nome


print(Produto.imposto)
