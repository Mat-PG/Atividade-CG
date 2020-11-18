Como treinar uma IA usando python e opencv

	Primeiramente montando o ambiente de trabalho, crie uma pasta para imagens positivas, uma para negativas e uma pasta de treinamento.
	Fa�a uma coleta de imagens do objeto que vc queira que a IA reconhe�a, essas ser�o as imagens positivas, e colete imagens em que esse objeto n�o apare�a, essas ser�o as negativas, pegue uma quantia mais ou menos do dobro das positivas. Certifique-se de renome� las de forma que n�o haja espa�os em seus nomes.

	Exemplo: �picture 01.png� o certo seria �picture01.png�.

Voc� pode usar esse c�digo .py, que remove espa�os, para ajudar:

	import urllib
import numpy as np
import cv2
import os
import os

for file_type in ['nome da pasta de imagens positivas aqui ']:
    for img in os.listdir(file_type):
        os.rename(file_type+"/"+img, file_type + "/"+img.replace(" ", ""))

for file_type in ['nome da pasta de imagens negativas aqui']:
    for img in os.listdir(file_type):
        os.rename(file_type+"/"+img, file_type + "/"+img.replace(" ", ""))

Ent�o criar um .txt listando todas as imagens negativas:(c�digo que criar� automaticamente para voc�)

	import urllib
import numpy as np
import cv2
import os

for file_type in ['nome da pasta de imagens negativas aqui']:
    for img in os.listdir(file_type):
        line = file_type+'/'+img+'\n'
        with open('negativas.txt','a') as f:
            f.write(line)


	Agora fa�a download das aplica��es do opencv em "opencv.org", extraia as onde desejar.
	Abra o cmd(prompt de comando) na pasta do projeto e execute via cmd o app opencv_annotation utilizando a sua localiza��o completa 
(C:\Users\Nome de usuario\Desktop\opencv\build\x64\vc15\bin\opencv_annotation.exe)
ou j� o definindo como vari�vel de ambiente utilizando apenas (opencv_annotation), os exemplos ser�o com a vari�vel de ambiente.
	execute o seguinte c�digo para come�ar o treinamento: 

opencv_annotation --annotations=annotations.txt --images= nome da pasta de imagens positivas

Ent�o ir� come�ar a etapa onde voc� precisa indicar a posi��o do objeto nas imagens positivas, na tela do cmd ir� mostrar as instru��es de bot�es. 
	
	Nota: tome cuidado ao usar imagens positivas com muitas caracter�sticas exteriores semelhantes, exemplo do que eu passei, ao tentar treinar uma IA para reconhecer bolas de t�nis, muitas das minhas imagens positivas eram de pessoas segurando a bola, como resultado as pontas dos dedos acabavam entrando no quadrado de sele��o, ent�o no final al�m da minha IA reconhecer e marcar a bola de t�nis ela tamb�m marcava pontos da minha pele e as pontas dos meu dedos.

Ap�s essa etapa execute o pr�ximo app com seguinte c�digo no cmd:

opencv_createsamples -info annotations.txt -bg negativas.txt -vec vetor.vec -w 24 -h 24

	Esse processo ser� r�pido , ap�s ele vamos a �ltimo app com o comando:

opencv_traincascade -data nome da pasta de treinamento -vec vetor.vec -bg negativas.txt -numPos 142 -numNeg 360 -w 24 -h 24 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -numStages 30 -acceptanceRatioBreakValue 1.0e-5

Essa � a etapa mais demorada, ap�s a sua conclus�o o treinamento estar� completo, o seu haarcascade estar� na pasta de treinamento com o nome de �cascade.xml�, voc� j� pode testar a IA com esse c�digo:

import numpy as np
import cv2

car_cascade = cv2.CascadeClassifier("pasta de treinamento/cascade.xml")
img = cv2.imread("imagem de analise.jpg")
height, width, c = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
objetos = car_cascade.detectMultiScale(gray, 1.2, 5)
print(objetos)
for (x,y,w,h) in objetos:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('Analise', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Voc� saber� que funcionou se o objeto for marcado com um ret�ngulo vermelho.
