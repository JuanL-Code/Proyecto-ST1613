# Proyecto-ST1613

# Modelo deteccion y reconocimiento automatico de piezas en una linea de producción basado en YOLO

# Introduccion
La empreza XYZ se encarga de la producción industrial de tornillos y tuercas. Es por ello que dispone de maquinaria automatizada para este proceso. El problema es que la maquina tiene adecuada a una misma linea tanto los moldes de tornillos como el molde de tuercas cuestion que implica o la adecuacion de una maquina adicional con un molde nuevo dedicado o la automatizacion de la separación de las piezas.
Es por ello que para facilitar esta tarea se plantea un sistema basado en Computer Vision que en primera instancia detecte que hay una pieza y en segunda instacia que etiquete esa pieza entre tornillo y tuerca.
De tal manera que es posible con una sola maquina continuar con la producción y a su vez separar las piezas para su empacado.


# Indicaciones
El algoritmo de deteccion y reconocimiento corre en un framework llamado YOLO el cual funciona por medio de una ingesta de imagenes y un entrenamiento previo en nube sin la necesidad de usar recursos propios como CPU o GPU

# Funcionamiento General
1. Obtención de las imagenes que componen el dataset: Se obtuvieron por medio de Google Images e imagenes propias tomadas. Para este caso en especifico se delegaron 300 imagenes de las cuales 200 seran usadas dentro del entrenamiento y validación en una proporcion 80/20 respectivamente. Se dejan las 100 imagenes restantes para el testeo del modelo.
2. Limpieza: Una vez se tiene el dataset de imagenes crudas sacadas de internet se comienza a hacer una limpieza basada en la eliminación de las imagenes que no tengan relacion con los objetos a detectar y reconocer. Adicionalmente se filtra el dataset para que solo hayan imagenes en formato png y jpg ya que el programa de taggeo es mas optimo con el manejo de estos modelos.
3. Taggeo: Teniendo ya las imagenes limpias y divididas se hace el taggeo por medio de LabelIMG.py para asi obtener un archivo .txt el cual especifica la clase del objeto y la ubicacion cartesiana del mismo dentro de la imagen.
4. Entrenamiento: El conjunto de imagenes y tags se pasan en .zip al modelo.ipynb en Google Colab en donde se hara el proceso de cargar el framework de YOLO y el entrenamiento por medio de la clase train.py y la especificacion de las clases en el customdata.yaml
5. Test: Teniendo ya el modelo creado se tendra un archivo best.pt con el cual se hara el testeo cargando una imagen. Los resultados se cargaran en runs/exp/.
6. Evaluación: Se evalua si el modelo se considera optimo para el caso trabajado. En caso de no tener un taggeo muy optimo se debera hacer un aumento del numero de epochs. Para este caso se probo con 10/20/50 Epochs siendo este ultimo el mejor rendimiento previo a un nivel de Overfitting.

# Metodologia 
## 1. Obtención Datos

Una vez se tiene identifacado los objetos con los cuales trabajar se debe construir dataset. En este caso se trabaja con la deteccion y reconomiento de tornillos y tuercas por lo cual se descargan y limpian la cantidad de 300 Imagenes (150 Tornillos + 150 Tuercas). Estas imagenes deben estar dispuestas en formato jpg y png. De las 300 Imagenes en el dataset solo se usaran 200 en el entrenamiento del modelo manteniendo una disposicion 80/20 (160 imagenes de entrenamiento y 40 de validación) y las otras 100 imagenes seran dispuestas para el test posterior/

1. Para la creacion de un modelo de IA enfocado a un modo de falla o cualquier otro uso se comienza haciendo una recoleccion minima de unas 100 images(Ej: Si se desea un modelo de IA que detecte y reconozca adicionalmente piezas con otras caracteristicas debemos recolectar 100 imagenes minimo de esta clase nueva de pieza)![evidencia proyecto 1](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/6d8b1e46-3c78-4de0-bb02-5ca2e6413d21)![evi 2](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/b17ec363-9c50-4fa2-8415-e59739d66276)
Renombramos las imagenes descargadas con un script en python para facilitar el taggeo.
![evidencia 4](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/43ecd9da-c935-4e81-9fa7-a9878da870bd)
Tendremos un subconjunto ya de imagenes que despues del etiquetado tendran un txt con las coordenadas y la clase dentro de cada clase.
![Screen Shot 2023-05-22 at 4 02 10 PM](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/a642c135-9f9c-4fd2-803e-5c231e98342f)

## 2. Taggeo/Instalacion de las dependencias
Estas imagenes se exportaran a un software basado en python llamado "LabelIMG" en el cual se hace una seleccion de la zona que corresponde a la indicada y se le da un nombre de clase, lo que hace LabelIMG es guardar un archivo .txt el cual guarda las posiciones cartesianas dentro de esa imagen de tal manera que hace una correlacion entre la imagen y este archivo txt. 

```
Ya teniendo las imagenes instalamos el programa basado en Python LabelIMG el cual no ayudara a taggear las imagenes creando por cada imagen un archivo txt con las coordenadas cartesianas del objeto identificado.

Su instalacion es:

$ git clone https://github.com/tzutalin/labelImg
$ conda install pyqt=5
$ conda install -c anaconda lxml
$ cd labelImg
$ pyrcc5 -o libs/resources.py resources.qrc
$ python labelImg.py

```
Al correr la ultima linea nos abrira el software de etiquetado 

Dentro de la carpeta clonada crearemos la siguiente estructura de carpetas

```

--train_data
  -- images
      --train
      --val
  -- labels
      --train
      --val
      
```

Una vez hecha la creacion de las carpetas, guardaremos en train_data/images/train las imagenes que deseamos etiquetar en formato jpg o jpeg y copiamos las imagenes tambien en train_data/images/val. 
Dentro de LabelImg en la barra izquierda seleccionaremos "Open Dir" y abriremos la carpeta  train_data/images/train y en la opcion de "Change save dir" seleccionaremos train_data/labels/train para guardar todos los archivos txt de las etiquetas.
Adiconal en la barra superior seleccionaremos el menu de view y daremos click a la opcion de "Auto Save mode"

De esta manera entonces nos quedaran cargadas todas la imagenes y tambien la ruta de guardado automatico de los taggeos que hagamos 

![evidencia 7](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/bd7182bd-9e7c-4b00-ac86-17433e6695e4)


Para taggear cada imagen usamos las siguientes teclas: con w hacemos el uso del cuadro para seleccionar la zona que deseamos etiquetar, le damos un nombre a la etiqueta y despues al boton de next para continuar con la siguiente imagen, asi hasta complementarlas todas
![EVIDENCIA 6](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/73efcbef-e067-4286-8480-f06e3361faf4)


3. Despues de hacer la creacion de los recuadros dentro de LabelIMG nos quedara una carpeta en la cual estaran todas. Cada imagen genera un archivo txt dentro del cual se ubica por cada recuadro un par organizado de coordenadas. En el ejemplo pasado mostrabamos dentro de labelIMG como etiquetabamos dentro de la imagen #145, el resultado sera entonces un txt homonimo con los 5 taggeos correspondientes, el numero cero al inicio de cada coordenada hace refentencia a la clase a la cual corresponde ese taggeo, en caso de tener dentro de un mismo modelo mas de una clase este numero se modificara(Ej. Si dentro de un modelo queremos identificar piezas corroidas y egrietadas).!

# 3. Preparacion de los archivos

Una vez listo el proceso de taggeo nos quedara en la carpeta train_data/labels/train todos los archivos txt los cuales copiaremos a la carpeta train_data/labels/val

Ahora, la carpeta train_data la comprimiremos para ser usada dentro de google colab

<img width="1230" alt="Screen Shot 2023-05-23 at 7 37 52 AM" src="https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/63e272e1-b2f9-431d-afcc-95ea6558fdfb">

# 4. Entrenamiento en Google Colab

Crearemos una cuenta de Google colab y copiaremos el siguiente notebook [Notebook](https://colab.research.google.com/drive/1AesT58ob5jsr6xEJK4cbenXftvjmY6r0?usp=sharing) como copia en nuestro drive dando en el menu de arhivo/guardar una copia en drive

## 4.1 Como correr el notebook
1. Primero daremos click en run setup para cargar e instalar el framework dentro de nuestro entorno virtual

<img width="1433" alt="Screen Shot 2023-05-23 at 7 41 59 AM" src="https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/ac0aedd9-a483-4935-a91f-0d5fa8ab2eae">


## 4.2 Cargar los datos
Primero subiremos a google colab el .zip "train_data", una vez cargado correremos la linea de cargar los datos

```

!unzip -q ../train_data.zip -d ../

```

## 4.3 Creamos un archivo de configuracion
En Visual Studio/Sublime/Notepad++/Bloc de notas creamos un archivo de configuracion con el siguiente codigo

```
path: ../train_data  # dataset root dir
train: ../train_data/images/train/  # train images (relative to 'path') 160 images
val: ../train_data/images/val/  # val images (relative to 'path') 40 images
test:  # test images (optional)

# Classes
nc: 2  # number of classes
names: ['screw', 'nut']  # class names

```

Este archivo cambia en base a las necesidades y usos que le den al modelo, en este caso los parametros a cambiar seria el numero de clases y el nombre de las mismas
Estes archivo lo guardamos como "customdata.yaml" y debe ser de formato en texto plano y lo subiremos dentro de google colab en la carpeta yolov5/data

## 4.4 Entrenamiento
Corremos la linea de entrenamiento, en este caso use un modelo de 160 imagenes y lo entrene durante 10/20/50 ciclos, tardando 4 horas el ultimo, se puede hacer pruebas en base a cada caso, lo importante es estar por encima de un numero de ciclos para conseguir un entrenamiento bueno pero tampoco superar determinado nivel por uso de tiempo y recursos y adicional evitando un overfitting del modelo.

```

!python train.py --img 640 --batch 4 --epochs 50 --data customdata.yaml --weights yolov5s.pt --cache

```

## 5 Test

Una vez finalizado el entrenamiento subiremos a google colab una imagen de prueba llamada test.jpg, al estar cargadas corremos las siguientes lineas 

```

-!python detect.py --weights runs/train/exp/weights/last.pt --img 640 --conf 0.25 --source "../test.jpg"
display.Image(filename='../test.jpg', width=600)

```

Estamos cargando la ruta del modelo entrenado y de la imagen de test, una vez finalizado iremos a yolov5/runs/detect/exp/test.jpg
Ese sera el resultado, podemos hacer diferentes test solo debemos modificar el parametro donde --source "../test(X).jpg y dado que soo carga el modelo una vez no se requiere hacer un entrenamiento nuevo

## 6. Evaluación

Primer test 10 Epochs:
![evidencia screw](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/a95546a1-4880-40a2-bb40-448563aaceed)


Segundo test 20 Epochs:

![Screen Shot 2023-05-22 at 8 50 51 PM](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/abb597c5-ad67-4994-9695-4293a31fb320)
![Screen Shot 2023-05-22 at 8 54 28 PM](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/8840f169-96f1-479a-95da-3cb0e7c8074d)
![Screen Shot 2023-05-22 at 8 56 53 PM](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/27f7b008-987b-40df-b957-c2c74e21b141)
![Screen Shot 2023-05-22 at 9 00 00 PM](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/6685448c-bb68-42a4-8cb1-37dcafe94419)
![Screen Shot 2023-05-22 at 9 02 15 PM](https://github.com/JuanL-Code/Proyecto_ML/assets/68828858/e1de31ff-e210-421a-98f7-af84300cb5ec)

Tercer test 50 Epochs:

