---
title: "Procesamiento Natural del Lenguaje en Twitter"
date: 2021-05-05
tags: [Twitter, Data science, NLP]
header:
  image: 
  excerpt: "Twitter, Data Science, NLP"
  mathjax: "true"
---



# NLP + Twitter + Iberdrola vs Naturgy

En este extenso ejercicio, vamos a crear un algoritmo para estudiar las palabras más representativas de una série de comentarios, varios modelos de regresión y un contraste de medias. El objetivo es determinar si los obtenidos en Twitter de dos compañías eléctricas, en este caso Iberdrola y Naturagy, son mejores unos que otros.

Éste tipo de valoraciones se pueden aplicar a todo tipo de productos y determinar la satisfacción de los clientes.

Para determinar si un comentario está valorado positivamente o negativamente, partimos de una tabla con 6.000 comentarios de Amazon valorados
en una escala del 1 al 5. Primero debemos estudiar las palabras de cada comentario y sacar las menos representativas, eso lo realizaremos con la funcion Tokenize y Bag of words. Una vez tengamos las palabras más importantes de cada comentario con la función Vectorizer, entrenaremos varios modelos para determinar hasta que punto es capaz de predecir si el comentario es positivo o negativo.

Con el mejor modelo ya obtenido, nos descargaremos una lista de tweets con el hastag de cada compañía. Le aplicaremos el Vectorizador y el modelo de regresión, con el fín de obtener la valoración de cada comentario. 

Con las valoraciones de cada compañía realizaremos un estudio estadístico, con el fín de determinar si una compañía tiene de media, comentarios más bien valorados que la otra.

Vamos a ello!

Importamos librerias
```python
import pandas as pd
import numpy as np
import re
import string
from string import punctuation
from zipfile import ZipFile
import rarfile, csv
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('spanish'))
```

Usamos un dataset externo con los comentarios
```python
dataset = 'amazon_es_reviews_pkl.zip'
```

Abrimos el archivo en formato rar
```python
with ZipFile(dataset, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
```

    File Name                                             Modified             Size
    amazon_es_reviews.pkl                          2021-03-23 16:19:02    199155314
    Extracting all the files now...
    Done!
    


```python
data = pd.read_pickle(dataset)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comentario</th>
      <th>estrellas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Para chicas es perfecto, ya que la esfera no e...</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Muy floja la cuerda y el anclaje es de mala ca...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Razonablemente bien escrito, bien ambientado, ...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hola! No suel o escribir muchas opiniones sobr...</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A simple vista m parecia una buena camara pero...</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Transformamos los datos en un arrays
```python
comentarios = data["comentario"].values
estrellas = data["estrellas"].values
```

Separamos en Train y Test (un 20% del total será Test)
```python
from sklearn.model_selection import train_test_split
```


```python
comentarios_train, comentarios_test, estrellas_train, estrellas_test = train_test_split(comentarios, estrellas, test_size=0.2, random_state=0)
```

## Vectorización de los comentarios


```python
spanish_stopwords = stopwords.words("spanish")
stemmer = SnowballStemmer("spanish")
non_words = list(punctuation)
non_words.extend(['¿', '¡', ',','...','``'])
non_words.extend(map(str,range(10)))
```


```python
def stem_tokens (tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def clean_data (tokens, stop_words_sp = ()):
    clean_tokens = []
    for token in tokens:
        if token.lower() not in spanish_stopwords and token not in non_words:
            clean_tokens.append (token)
    return clean_tokens


def tokenize(text):
    tokens = []
    text = ''.join([c for c in text if c not in non_words]) # Limpieza del texto eliminando 
    tokens =  word_tokenize(text)
    tokens_limpios = clean_data(tokens)
    tokens_stemmed = stem_tokens(tokens_limpios, stemmer)
    return tokens_stemmed


```


```python
vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)
```


```python
tokens = tokenize(data["comentario"][:200])
tokens
```
Solo mostraremos algunas:



    ['chic',
     'perfect',
     'esfer',
     'grand',
     'corre',
     'adapt',
     
     ...]




```python
vectorizer.fit(comentarios_train)
```




    CountVectorizer(stop_words=['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los',
                                'del', 'se', 'las', 'por', 'un', 'para', 'con',
                                'no', 'una', 'su', 'al', 'lo', 'como', 'más',
                                'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí',
                                'porque', ...],
                    tokenizer=<function tokenize at 0x000002704C0EEA60>)



Guardamos un modelo que se podrá usar para más proyectos
```python
filename = 'vectorizador.sav'
pickle.dump(vectorizer, open(filename, 'wb'))
```


```python
filename = 'vectorizador.sav'
vectorizador_coment = pickle.load(open(filename, 'rb'))
```


```python
# Realizamos el transform de los comentarios con el set de train y con el de test
Comentarios_train = vectorizer.transform(comentarios_train)
Comentarios_test  = vectorizer.transform(comentarios_test)
```

## Modelo de regresión para puntuar los comentarios


```python
# Importamos las librerías necesarias para este proceso:
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 
```


```python
#Modelo de regresión, con más tiempo se podría ampliar el max_iter
reglog = LogisticRegression(max_iter=100)

#Grid search
grid_hyper_reglog ={}
            
#Cross validation y scoring
gs_reglog = GridSearchCV(reglog,
                 param_grid=grid_hyper_reglog,
                 cv=8,
                 scoring="r2",
                 n_jobs=-1,
                 verbose=3)
```


```python
# lanzamos el entrenamiento del modelo:
gs_reglog.fit(Comentarios_train, estrellas_train) 
```

    Fitting 8 folds for each of 1 candidates, totalling 8 fits
    

    GridSearchCV(cv=8, estimator=LogisticRegression(), n_jobs=-1, param_grid={},
                 scoring='r2', verbose=3)




```python
gs_reglog.best_score_
```




    0.3665930354051587




```python
#evaluamos el modelo en train y test con r2

from sklearn.metrics import mean_squared_error, r2_score, make_scorer

r2_en_train = r2_score(estrellas_train, y_pred = gs_reglog.predict(Comentarios_train))
print("El modelo tiene un r2 en el conjunto de train de %s" % r2_en_train)
r2_en_test = r2_score(estrellas_test, y_pred = gs_reglog.predict(Comentarios_test))
print("El modelo tiene un r2 en el conjunto de test de %s" % r2_en_test)
```

    El modelo tiene un r2 en el conjunto de train de 0.4166076528556948
    El modelo tiene un r2 en el conjunto de test de 0.37037639690207613
    


```python
#guardamos el modelo
filename = 'finalized_model.sav'
pickle.dump(gs_reglog, open(filename, 'wb'))
```


```python
#cargamos el modelo 
file = 'finalized_model.sav'
loaded_model = pickle.load(open(file, 'rb'))
```

#### Testamos otros modelos 


```python
#Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=20, # 20 árboles
                               criterion="mse",
                               max_depth=3,
                               min_samples_split=10,
                               min_samples_leaf=5,
                               bootstrap=True)

grid_rf_reg = {}


gs_rf_reg = GridSearchCV(rf_reg,
                       grid_rf_reg,
                       cv=5,
                       verbose=1,
                       n_jobs=-1)
```


```python
gs_rf_reg.fit(Comentarios_train, estrellas_train)
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits
    



    GridSearchCV(cv=5,
                 estimator=RandomForestRegressor(max_depth=3, min_samples_leaf=5,
                                                 min_samples_split=10,
                                                 n_estimators=20),
                 n_jobs=-1, param_grid={}, verbose=1)




```python
gs_rf_reg.best_score_
```




    0.12910840154484676




```python
r2_en_train = r2_score(estrellas_train, y_pred = gs_rf_reg.predict(Comentarios_train))
print("El modelo tiene un r2 en el conjunto de train de %s" % r2_en_train)
r2_en_test = r2_score(estrellas_test, y_pred = gs_rf_reg.predict(Comentarios_test))
print("El modelo tiene un r2 en el conjunto de test de %s" % r2_en_test)
```

    El modelo tiene un r2 en el conjunto de train de 0.1291251195914883
    El modelo tiene un r2 en el conjunto de test de 0.12686542001753187
    


```python
#Gradien boosting
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score, make_scorer

#La función mean_squared_error es un función tipo loss, por lo que debemos anular el greater is better del scoring
# que por defecto es True, sino nos selecciona los peores modelos (maximiza el mayor error)

my_func = make_scorer(score_func=mean_squared_error, greater_is_better=False)
```


```python
#Gradient Boosting
GB_reg=GradientBoostingRegressor(n_estimators= 100,
                                 learning_rate=0.1, 
                                 max_depth=3, 
                                 max_features=3, 
                                 random_state=0)

grid_gradient_boosting_reg = {}

#Gradient Boosting
gs_gradient_boosting_reg = GridSearchCV(GB_reg,
                       grid_gradient_boosting_reg,
                       cv=5,
                       scoring= "r2",
                       verbose=1,
                       n_jobs=-1)
```


```python
gs_gradient_boosting_reg.fit(Comentarios_train, estrellas_train)
```

    Fitting 5 folds for each of 1 candidates, totalling 5 fits



    GridSearchCV(cv=5,
                 estimator=GradientBoostingRegressor(max_features=3,
                                                     random_state=0),
                 n_jobs=-1, param_grid={}, scoring='r2', verbose=1)




```python
gs_gradient_boosting_reg.best_score_

```




    0.0021090472578160434




```python
r2_en_train = r2_score(estrellas_train, y_pred = gs_gradient_boosting_reg.predict(Comentarios_train))
print("El modelo tiene un r2 en el conjunto de train de %s" % r2_en_train)
r2_en_test = r2_score(estrellas_test, y_pred = gs_gradient_boosting_reg.predict(Comentarios_test))
print("El modelo tiene un r2 en el conjunto de test de %s" % r2_en_test)
```

    El modelo tiene un r2 en el conjunto de train de 0.002687320614535138
    El modelo tiene un r2 en el conjunto de test de 0.00251333833448808
    

El mejor modelo ha resultado ser la regresión logística con una r2 de 0,41 en el Train. Es un resultado relativamente bajo y se deberian añadir más comentarios para mejorar el modelo, también se podría testar otros modelos y ampliar el los hiperparámetros de los modelos. De momento seguiremos con el modelo de Regresión.

# Obtención de los Tweets con Tweepy


```python
import tweepy
import json
import csv
```

Los datos son privados pero se obtinen al aplciar a una cuenta de desarrolador en twiiter
```python
consumer_key = "####"
consumer_secret = "####"
access_token = "####"
access_secret = "####"
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
```


```python
#hastags
#Abrimos o creamos un archivo donde guardar los tweets (solo se muestran algunos)
csvFile = open('iberdrola.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)


for tweet in tweepy.Cursor(api.search,q="iberdrola",  
                           count=10,
                           #lang="sp",
                           since="2021-05-01").items(1000):
    
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
```

    2021-05-05 10:54:15 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:53:11 @pricilina_ @Carmela_Mendez @Ana___Chaves @bmartinron Creo que soy la hija del de Iberdrola😁
    2021-05-05 10:53:07 RT @adrianomones: Tenemos a más de 100 personas  asistiendo al acto de presentación del #MOVES3 del @IDAEenergia. En nombre de @iberdrola y…
    2021-05-05 10:52:13 RT @adrianomones: Tenemos a más de 100 personas  asistiendo al acto de presentación del #MOVES3 del @IDAEenergia. En nombre de @iberdrola y…
    2021-05-05 10:50:58 @cubitodequeso @mariahvv84 @PabloMontanoB 1/n)De donde saco lo de las comunidades indígenas 🤨 si se refiere al prob… https://t.co/FNyuYwnKkN
    2021-05-05 10:50:03 Estoy escuchando Herrera en COPE en https://t.co/OPqesOR4CA.
    NO TE LO PIERDAS,COPE,EL MARIDO,CORNUDO ME DICE QUE ER… https://t.co/QwvntS4bBT
    2021-05-05 10:47:13 @profunditats @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Si representan a España, que quieres que hagan??😂
    2021-05-05 10:44:19 RT @LomanaGloria: 📌#MujeresInspiradoras ➡️Ángeles Santamaría CEO de @iberdrola 👇
    
    "Igual que en otras #CienciasyTecnología sí que hay mucha…
    2021-05-05 10:43:24 RT @LomanaGloria: 📌#MujeresInspiradoras ➡️Ángeles Santamaría CEO de @iberdrola 👇
    
    "Igual que en otras #CienciasyTecnología sí que hay mucha…
    2021-05-05 10:42:56 📌#MujeresInspiradoras ➡️Ángeles Santamaría CEO de @iberdrola 👇
    
    "Igual que en otras #CienciasyTecnología sí que hay… https://t.co/yj9DhPBtFY
    2021-05-05 10:39:24 #INFO /Aplazamiento 2ª Jornada XXVII Liga Canaria de Clubes de 1ª y 2ª Las Palmas debido a la coincidencia con la L… https://t.co/reooi5AIUt
    2021-05-05 10:38:17 RT @adrianomones: Todo listo para presentar el #MOVES3 con @Arturopdelucia y @AEDIVE  @IDAEenergia @GobAsturias @FundacionFaen @iberdrola @…
    2021-05-05 10:35:42 RT @adrianomones: Tenemos a más de 100 personas  asistiendo al acto de presentación del #MOVES3 del @IDAEenergia. En nombre de @iberdrola y…
    2021-05-05 10:33:42 RT @adrianomones: Tenemos a más de 100 personas  asistiendo al acto de presentación del #MOVES3 del @IDAEenergia. En nombre de @iberdrola y…
    2021-05-05 10:33:37 Tenemos a más de 100 personas  asistiendo al acto de presentación del #MOVES3 del @IDAEenergia. En nombre de… https://t.co/cOace3t8If
    2021-05-05 10:29:08 @MarioAndreu10 @CgYimii @noelia_nessi00 ya vas siendo mayorcito para saber quien fija el precio de la luz.
    Quien fi… https://t.co/F9K4tjme0O
    2021-05-05 10:27:29 @VicsabaterPerez @DSR_Gw @juanrallo en iberdrola? estas flipando...
    2021-05-05 10:27:14 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:21:50 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:21:47 Reto Iberdrola: un ascenso, dos sueños | Informa: Sergio Hervás.
    Futbolísticas - El periódico del #FutFem 
    https://t.co/bVOnz8tq8N
    2021-05-05 10:20:17 RT @fusionforenergy: Another #European #magnet heading to #ITER
    18 Toroidal Field coils will confine the hot plasma🔥
    Each of these 🧲 is 1⃣…
    2021-05-05 10:18:45 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-05 10:18:26 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:16:30 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:15:56 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:15:37 @iberdrola es cierto que estáis llamando pidiendo datos para ver si el titular se acoge al bono social ? Me han lla… https://t.co/JtbRd1ZqlG
    2021-05-05 10:14:24 RT @DHIberdrola: #LigaIberdrolaRugby | @lesabellesrc se despide de la máxima categoría. 🏉🐝
    
   

Creamos otra tabla con la que compararemos más tarde


```python
#Opció hastag
# Open/Create a file to append data
csvFile = open('naturgy.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)


for tweet in tweepy.Cursor(api.search,q="naturgy",  
                           count=10,
                           #lang="sp",
                           since="2021-01-01").items(1000):
    
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
```

    2021-05-05 10:26:20 RT @Petroleo_Arg: @Jotatrader_ok No se si fuiste irónico, pero si ves una factura de Naturgy el IVA se aplica a IIBB y al impuesto a débito…
    2021-05-05 10:10:03 @juliguli 1.- Entras en Área Clientes de la web de Naturgy y das el dato, 
    2.- Te descargas la App de Naturgy Clien… https://t.co/AbBEMd6TXb
    2021-05-05 09:33:37 RT @Alfons_ODG: Aràbia Saudita vol construir Helios, un projecte d'hidrogen verd de 4GW i 5.000 milions per exportar al món. 
    
    Bojeria saud…
    2021-05-05 09:19:00 Queremos agradecer enormemente el apoyo que hemos recibido de @Naturgy en la adecuación de instalaciones auxiliares… https://t.co/gN3AE4n9kM
    2021-05-05 09:16:08 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 09:09:19 @Jotatrader_ok No se si fuiste irónico, pero si ves una factura de Naturgy el IVA se aplica a IIBB y al impuesto a… https://t.co/oWuj6GYS3c
    2021-05-05 09:04:21 RT @Women360Congres: No te pierdas nuestra MESA DE INNOVACIÓN E INTELIGENCIA ARTIFICIAL EN EL SECTOR SANITARIO en el #womentech21
    
    ¡Inscríb…
    2021-05-05 09:04:17 RT @Women360Congres: Dra. @KarinagibertK Vicedegana del Colegio Oficial de Ingeniería Informática de Catalunya, investigadora en Ciencia de…
    2021-05-05 09:04:08 RT @Women360Congres: No te pierdas nuestra MESA DE INNOVACIÓN E INTELIGENCIA ARTIFICIAL EN EL SECTOR SANITARIO en el #womentech21
    
    ¡Inscríb…
    2021-05-05 09:04:05 RT @Women360Congres: Dra. @KarinagibertK Vicedegana del Colegio Oficial de Ingeniería Informática de Catalunya, investigadora en Ciencia de…
    2021-05-05 09:03:51 RT @Women360Congres: Dra. @KarinagibertK Vicedegana del Colegio Oficial de Ingeniería Informática de Catalunya, investigadora en Ciencia de…
    2021-05-05 09:03:48 RT @Women360Congres: No te pierdas nuestra MESA DE INNOVACIÓN E INTELIGENCIA ARTIFICIAL EN EL SECTOR SANITARIO en el #womentech21
    
    ¡Inscríb…
    2021-05-05 09:03:19 RT @Women360Congres: No te pierdas nuestra MESA DE INNOVACIÓN E INTELIGENCIA ARTIFICIAL EN EL SECTOR SANITARIO en el #womentech21
    
    ¡Inscríb…
    2021-05-05 09:03:16 RT @Women360Congres: Dra. @KarinagibertK Vicedegana del Colegio Oficial de Ingeniería Informática de Catalunya, investigadora en Ciencia de…
    2021-05-05 09:02:10 Dra. @KarinagibertK Vicedegana del Colegio Oficial de Ingeniería Informática de Catalunya, investigadora en Ciencia… https://t.co/nbTWdWGaRm
    2021-05-05 08:57:29 @NaturgyClientEs @Naturgy  sigo sin obtener respuesta con respecto al código de 7€ de descuento en compra Amazon. N… https://t.co/XusItWn4SV
    2021-05-05 08:56:04 RT @Women360Congres: No te pierdas nuestra MESA DE INNOVACIÓN E INTELIGENCIA ARTIFICIAL EN EL SECTOR SANITARIO en el #womentech21
    
    ¡Inscríb…
    2021-05-05 08:54:53 No te pierdas nuestra MESA DE INNOVACIÓN E INTELIGENCIA ARTIFICIAL EN EL SECTOR SANITARIO en el #womentech21
    
    ¡Insc… https://t.co/9neIZcT7PA
    2021-05-05 08:48:45 He fotografiado su nueva sede corporativa en la Diagonal de #Barcelona para @Naturgy Como una llama de gas que enci… https://t.co/8N2X8KGXON
    2021-05-05 08:40:14 La caldera de la central térmica de Anllares dinamitará este 6 de mayo #Bierzo #Ponferrada @Naturgy 
    https://t.co/nbqDZkuej9
    2021-05-05 08:18:49 RT @FullsEnginyeria: 🐄 Tres projectes comencen a impulsar el #biometà a Catalunya.
    
    ♻️ Una jornada a @Enginyeria posa sobre la taula el fut…
    2021-05-05 08:18:07 🐄 Tres projectes comencen a impulsar el #biometà a Catalunya.



```python
# Función para limpiar el texto

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text ) #remove mentions
    text = re.sub(r'#', '', text ) #remove #
    text = re.sub(r'RT[\s]+', '', text) #remove RT
    text = re.sub(r'https?:\/\/S+' , '', text) # remove https
    text = re.sub(r'\xf0' , '', text) # remove https
    
    return text
```


```python
#Abrimos la tabla y limpiamos los comentarios y RT
```


```python
data1= pd.read_csv('iberdrola.csv', delimiter="," )
data1.columns= ["TweetDate", "Tweet"]
tweet_clean1 = data1[~data1.Tweet.str.contains("RT")]
tweet_clean1['Tweet']=tweet_clean1['Tweet'].apply(cleanTxt)
data1.head()
```

    <ipython-input-55-6acf0e6838da>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      tweet_clean1['Tweet']=tweet_clean1['Tweet'].apply(cleanTxt)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TweetDate</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-05-05 10:53:11</td>
      <td>b'@pricilina_ @Carmela_Mendez @Ana___Chaves @b...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-05-05 10:53:07</td>
      <td>b'RT @adrianomones: Tenemos a m\xc3\xa1s de 10...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-05-05 10:52:13</td>
      <td>b'RT @adrianomones: Tenemos a m\xc3\xa1s de 10...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-05-05 10:50:58</td>
      <td>b'@cubitodequeso @mariahvv84 @PabloMontanoB 1/...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-05-05 10:50:03</td>
      <td>b'Estoy escuchando Herrera en COPE en https://...</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
data2= pd.read_csv('naturgy.csv', delimiter="," )
data2.columns= ["TweetDate", "Tweet"]
tweet_clean2 = data2[~data2.Tweet.str.contains("RT")]
tweet_clean2['Tweet']=tweet_clean2['Tweet'].apply(cleanTxt)
tweet_clean2.head()
```

    <ipython-input-56-01ac66e000c8>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      tweet_clean2['Tweet']=tweet_clean2['Tweet'].apply(cleanTxt)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TweetDate</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-05-05 10:10:03</td>
      <td>b' 1.- Entras en \xc3\x81rea Clientes de la we...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-05-05 09:19:00</td>
      <td>b'Queremos agradecer enormemente el apoyo que ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-05-05 09:09:19</td>
      <td>b'_ok No se si fuiste ir\xc3\xb3nico, pero si ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2021-05-05 09:02:10</td>
      <td>b'Dra.  Vicedegana del Colegio Oficial de Inge...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2021-05-05 08:57:29</td>
      <td>b'   sigo sin obtener respuesta con respecto a...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Pasamos las tablas a arrays
```


```python
tweet_array1 = tweet_clean1["Tweet"].values
tweet_array2 = tweet_clean2["Tweet"].values
```


```python
tweet_vectorizado1  = vectorizer.transform(tweet_array1)
tweet_vectorizado2  = vectorizer.transform(tweet_array2)
```


```python
#Aplicamos el modelo para puntuar los comentarios
```


```python
y_pred1 = gs_reglog.predict(tweet_vectorizado1)
y_pred1 = pd.DataFrame(y_pred1)
y_pred1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>348</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>349</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>350</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>351</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>352</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>353 rows × 1 columns</p>
</div>




```python
y_pred2 = gs_reglog.predict(tweet_vectorizado2)
y_pred2 = pd.DataFrame(y_pred2)
y_pred2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>379</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>380</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>382</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>383</th>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>384 rows × 1 columns</p>
</div>




```python
#Juntamos los arrays en una tabla
frames = [y_pred1, y_pred2]

result = pd.concat(frames, axis=1)
result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.columns = ["Iberdola", "Naturgy"]
```


```python
result.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iberdola</th>
      <th>Naturgy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>353.000000</td>
      <td>384.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.764873</td>
      <td>2.817708</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.715143</td>
      <td>1.718612</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import style
import matplotlib.pyplot as plt
```


```python
#grafico de Iberdola
plt.hist(x=result["Iberdola"], bins=5, color='lawngreen', rwidth=0.95)
plt.title('Histograma de valoración de Tweets de Iberdrola')
plt.xlabel('Valoración del Tweet del 1 al 5')
plt.ylabel('Número de Tweets')
plt.show()
```


{% include figure image_path="/images/output_59_0.png" %}      

    



```python
#grafico de Naturgy
plt.hist(x=result["Naturgy"], bins=5, color='darkorange', rwidth=0.95)
plt.title('Histograma de valoración de Tweets de Naturgy')
plt.xlabel('Valoración del Tweet del 1 al 5')
plt.ylabel('Número de Tweets')
plt.show()
```


{% include figure image_path="/images/output_60_0.png" %}     

    



```python
boxplot = result.boxplot(rot=90)

```


{% include figure image_path="/images/output_61_0.png" %}      

    
Se demustra que la variación es muy elevada, y que los comentarios son o muy positivos o muy negativos, hay poco término medio.

# Son las medias iguales?


```python
from scipy.stats import ttest_ind
from scipy import stats
import scipy.stats
```


```python
ibe_mean = np.mean(result["Iberdola"])
naturgy_mean = np.mean(result["Naturgy"])
```


```python
print(ibe_mean)
print(naturgy_mean)
```

    2.764872521246459
    2.8177083333333335
    


```python
ibe_std=np.std(result["Iberdola"])
naturgy_std=np.std(result["Naturgy"])
```


```python
print(ibe_std)
print(naturgy_std)
```

    1.7127123587969484
    1.7163730407258673
    


```python
#calculo de la varianza
var_ibe = result["Iberdola"].var(ddof=1)
var_naturgy = result["Naturgy"].var(ddof=1)
```


```python
print(var_ibe)
print(var_naturgy)
```

    2.9417171001802838
    2.9536281549173236
    


```python
#si queremos comprar las dos medias, primero devemos saber si las varainzas son iguales
#std deviation
s = np.sqrt((var_ibe + var_naturgy)/2)
print(s)

N=len(result["Iberdola"])+len(result["Naturgy"])
print(N)
```

    1.7168787457327335
    768
    


```python
#calculo del intervalo de confiaza de las varianzas
f=1.17 #obtenido de la tabla de Fisher
#intervalos a y b
int_a= (var_ibe/var_naturgy)/f
int_b= (var_ibe/var_naturgy)*f
print(int_a)
print(int_b)
```

    0.8512541145798318
    1.1652817574483316
    

Con una confianza del 95 podemos decir que las varianzas son iguales ya que tenemos el 1 entre los intervalos


```python
n_ibe=len(result["Iberdola"])
n_naturgy =len(result["Naturgy"])
N = n_ibe + n_naturgy
```


```python

#Degrees of freedom
df = N - 2
df
```




    766




```python
# t valor con N-2 grados de libertad, dato obtenido de la tabla
t = 1.96

int_a_media = (ibe_mean-naturgy_mean) - t*s*np.sqrt((1/n_ibe)+(1/n_naturgy))
int_b_media = (ibe_mean-naturgy_mean) + t*s*np.sqrt((1/n_ibe)+(1/n_naturgy))
print(int_a_media)
print(int_b_media)
```

    -0.29568971156048607
    0.19001808738673687
    

Con una confinza del 95% podemos decir que las medias son iguales. Tanto Iberdorla como Naturgy tienen una media estadísticamente igual en la valoración de los tweets.

# Conclusiones

Se ha trabajado con NLP para detectar las palabras más representativas de los comentarios, con ello se ha aplicado un modelo para determinar su valoración.
Una vez con el Vectorizador y el modelo, se ha aplicado a Tweets descargados de los días 3 al 5 de mayo de 2021 con los hastagas Iberdrola y Naturgy.

El contraste de hipótesis nos ha demostrado que no había diferencias significativas entre los tweets de las dos compañías. Además las dos compañías han obtenido un 2,76 y 2,81 de valoración en los comentarios sobre 5. Si lo calculamos sobre una puntuación máxima de 10, podemos decir que han obtenido un 5,52 para Iberdrola y un 5,62 para Naturgy.

Hay que destacar que lso Tweets incluyen quejas de los usuarios, autoplublicidad de las compañías, comentarios respecto al despempeño en bolsa de las acciones, etc.. y solo son tweets de 3 días. Podria ser más efectivo si hiciera el mismo análisis pero para comentarios del buzon de clientes de una compañía en concreto.

Teniendo los algorítmos desarrollados, se peuden comprar sobre todos los temas imaginables, des de productos, ciudades, personas, partidos políticos, acciones.. en este cas han sido compañías pero se puede adaptar a cualquier producto.

# Agradecimientos
La parte del codigo de NLP se ha adaptado de Hector Escaso (https://hescaso.github.io/), posteriormente se ha creado un modelo de regresión para poder determinar la puntuación de cada modelo.
