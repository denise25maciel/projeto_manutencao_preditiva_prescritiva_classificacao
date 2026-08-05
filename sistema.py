"""Ponto de entrada do sistema."""
from prep import carregar_dados


def main():
    dados = carregar_dados()
    print(dados.head())


if __name__ == "__main__":
    main()
