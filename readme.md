# Gerador DataSet

Esse código tem o objetivo de gerar dataset inicial com as poses de vídeos com sinais em libras.

* Giovanna Lima Marques 
* Ricardo Augusto Coelho (https://github.com/tiorac)
* Tiago Goes Teles 
* Wellington de Jesus Albuquerque 


## Preparando para geração do dataset


1. Clone o repositório.
    ```cmd
    git clone https://github.com/ia-equipe-6/libras-data-generation.git
    ```
1. Cria uma pasta chamado "videos" na pasta do código.
    ```cmd
    mkdir videos
    ```
1. Cria uma pasta chamado "output" na pasta do código.
    ```cmd
    mkdir output
    ```
1. Para cada palavra, crie uma pasta dentro da pasta "videos" criado anteriormante.
    ```cmd
    cd videos
    mkdir palavra1
    mkdir palavra2
    cd ..
    ```
1. Adicione os vídeos de cada palavra dentro das pastas criada no passo anterior, uma palavra por vídeo.
1. Execute o gerador de dataset:
    ```cmd
    python .\generate_data.py
    ```

Todas as saídas estarão na pasta "output", incluíndo o CSV gerado.

## Próximos Passos

Utilize o código de transformação do dataset para treinamento:
https://github.com/ia-equipe-6/libras-dataset-transform

## Bibliotecas

Esse código utiliza as seguintes bibliotecas para geração do dataset:

* Pandas
* OpenCV 
* MediaPipe