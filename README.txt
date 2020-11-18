Como treinar uma IA usando python e opencv

	Primeiramente montando o ambiente de trabalho, crie uma pasta para imagens positivas, uma para negativas e uma pasta de treinamento.
	Faça uma coleta de imagens do objeto que vc queira que a IA reconheça, essas serão as imagens positivas, e colete imagens em que esse objeto não apareça, essas serão as negativas, pegue uma quantia mais ou menos do dobro das positivas. Certifique-se de renomeá las de forma que não haja espaços em seus nomes.

	Exemplo: “picture 01.png” o certo seria “picture01.png”.

Você pode usar esse código .py, que remove espaços, para ajudar:

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

	Então criar um .txt listando todas as imagens negativas:(código que criará automaticamente para você)

	import urllib
	import numpy as np
	import cv2
	import os

	for file_type in ['nome da pasta de imagens negativas aqui']:
	    for img in os.listdir(file_type):
		line = file_type+'/'+img+'\n'
		with open('negativas.txt','a') as f:
		    f.write(line)


	Agora faça download das aplicações do opencv em "opencv.org", extraia as onde desejar.
	Abra o cmd(prompt de comando) na pasta do projeto e execute via cmd o app opencv_annotation utilizando a sua localização completa 
	(C:\Users\Nome de usuario\Desktop\opencv\build\x64\vc15\bin\opencv_annotation.exe) ou já o definindo como variável de ambiente utilizando apenas (opencv_annotation), os exemplos serão com a variável de ambiente.
	Execute o seguinte código para começar o treinamento: 

	opencv_annotation --annotations=annotations.txt --images= nome da pasta de imagens positivas

	Então irá começar a etapa onde você precisa indicar a posição do objeto nas imagens positivas, na tela do cmd irá mostrar as instruções de botões. 
	
	Nota: tome cuidado ao usar imagens positivas com muitas características exteriores semelhantes, exemplo do que eu passei, ao tentar treinar uma IA para reconhecer bolas de tênis, muitas das minhas imagens positivas eram de pessoas segurando a bola, como resultado as pontas dos dedos acabavam entrando no quadrado de seleção, então no final além da minha IA reconhecer e marcar a bola de tênis ela também marcava pontos da minha pele e as pontas dos meu dedos.

	Após essa etapa execute o próximo app com seguinte código no cmd:

	opencv_createsamples -info annotations.txt -bg negativas.txt -vec vetor.vec -w 24 -h 24

	Esse processo será rápido , após ele vamos a último app com o comando:

	opencv_traincascade -data nome da pasta de treinamento -vec vetor.vec -bg negativas.txt -numPos 142 -numNeg 360 -w 24 -h 24 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -numStages 30 -acceptanceRatioBreakValue 1.0e-5

	Essa é a etapa mais demorada, após a sua conclusão o treinamento estará completo, o seu haarcascade estará na pasta de treinamento com o nome de “cascade.xml”, você já pode testar a IA com esse código:

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

	Você saberá que funcionou se o objeto for marcado com um retângulo vermelho.
