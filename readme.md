# Gerador DataSet

Esse código tem o objetivo de gerar dataset inicial com as poses de vídeos com sinais em libras, criando um dataset onde cada linha representa a pose de um único frame de um vídeo.

* Giovanna Lima Marques 
* Ricardo Augusto Coelho (https://github.com/tiorac)
* Tiago Goes Teles 
* Wellington de Jesus Albuquerque 

## Processo

Esse código segue os seguintes processos:

1. Procura todas as pastas dentro da pasta "videos", onde o nome da pasta, será a palavra que cada vídeo representa.
1. Para cada pasta encontrado, buscará todos os vídeos para ser processado.
1. Para cada vídeo encontrado, será lido todos os frames do vídeo.
1. Para cada frame, é processado a identificação de poses via mediapipe e gerado uma linha do dataset e uma imagem de validação.
1. Finalizado o processo de todas as pastas, o dataset é salvo.


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