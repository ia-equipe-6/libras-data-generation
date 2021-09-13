# Gerador DataSet

Esse código tem o objetivo de gerar dataset inicial com as poses de vídeos com sinais em libras.

* Giovanna Lima Marques 
* Ricardo Augusto Coelho (@tiorac)
* Tiago Goes Teles 
* Wellington de Jesus Albuquerque 


## Preparando para geração do dataset


1. Clone o repositório.
1. Cria uma pasta chamado "videos" na pasta do código.
1. Cria uma pasta chamado "output" na pasta do código.
1. Para cada palavra, crie uma pasta dentro da pasta "videos" criado anteriormante.
1. Adicione os vídeos dentro das pastas criada no passo anterior.
1. Execute:
    ```cmd
    python .\generate_data.py
    ```

Todas as saídas estarão na pasta "output", incluíndo o CSV gerado.

## Bibliotecas

Esse código utiliza as seguintes bibliotecas para geração do dataset:

* Pandas
* OpenCV 
* MediaPipe