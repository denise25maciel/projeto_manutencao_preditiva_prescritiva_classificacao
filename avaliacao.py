"""Avaliação inicial do sistema."""
from prep import carregar_dados


def avaliar():
    dados = carregar_dados()
    print(f"Linhas carregadas: {len(dados)}")


if __name__ == "__main__":
    avaliar()
