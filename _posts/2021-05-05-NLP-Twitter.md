---
title: "Procesamiento Natural del Lenguaje en Twitter"
date: 2021-05-05
tags: [data, Data science, finance]
header:
  image: 
  excerpt: "Twitter, Data Science, NLP"
  mathjax: "true"
---



# NLP + Twitter + Iberdrola vs Naturgy

En este extenso ejercicio, vamos a crear un algoritmo para estudiar las palabras más representativas de una série de comentarios, varios modelos de regresión y un contraste de medias. El objetivo es determinar si los obtenidos en Twitter de dos compañias eléctricas, en este caso Iberdrola y Naturagy, son mejores unos que otros.

Éste tipo de valoraciones se pueden aplicar a todo tipo de productos y determinar la satisfacción de los clientes.

Para determinar si un comentario está valorado positivamente o negativamente, partimos de una tabla con 6.000 comentarios de Amazon valorados
en una escala del 1 al 5. Primero debemos estudiar las palabras de cada comentario y sacar las menos representativas, eso lo realizaremos con la funcion Tokenize y Bag of words. Una vez tengamos las palabras más importantes de cada comentario con la función Vectorizer, entrenaremos varios modelos para determinar hasta que punto es capaz de predecir si el comentario es positivo o negativo.

Con el mejor modelo ya obtenido, nos descargaremos una lista de tweets con el hastag de cada compañia. Le aplicaremos el Vectorizador y el modelo de regresión, con el fín de obtener la valoración de cada comentario. 

Con las valoraciones de cada compañia realizaremos un estudio estadístico, con el fín de determinar si una compañia tiene de media, comentarios más bien valorados que la otra.

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


```python
dataset = 'amazon_es_reviews_pkl.zip'
```


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




```python
comentarios = data["comentario"].values
estrellas = data["estrellas"].values
```


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




    ['chic',
     'perfect',
     'esfer',
     'grand',
     'corre',
     'adapt',
     'muñec',
     'fin',
     'pelin',
     'gord',
     'gust',
     'carg',
     'movimient',
     'dur',
     'despues',
     '1-2',
     'dias',
     'llev',
     'para.muy',
     'floj',
     'cuerd',
     'anclaj',
     'mal',
     'calid',
     'metal',
     'dobl',
     'facil',
     'recomiendorazon',
     'bien',
     'escrit',
     'bien',
     'ambient',
     'quizas',
     'previs',
     'histori',
     'interes',
     'personaj',
     'bien',
     'dibuj',
     'parec',
     'libr',
     'recomendable.hol',
     'suel',
     'escrib',
     'much',
     'opinion',
     'product',
     'compr',
     'verd',
     'product',
     'merec',
     'desencant',
     'maquin',
     'afeit',
     'hac',
     'años',
     'compr',
     'philips',
     'cuchill',
     'rotatori',
     'piel',
     'extrem',
     'sensibl',
     'usar',
     'afeit',
     'hic',
     'polv',
     'car',
     'vam',
     'carn',
     'viv',
     'dej',
     'volv',
     'usar',
     'vez',
     'mism',
     'part',
     'apur',
     'bien',
     'barb',
     'asi',
     'despues',
     'mal',
     'experient',
     'pas',
     'cuchill',
     'gillett',
     'mach3',
     'verd',
     'iban',
     'mejor',
     'dej',
     'car',
     'ardor',
     'dur',
     'dia',
     'hac',
     'vari',
     'seman',
     'leyend',
     'for',
     'demas',
     'vi',
     'afeit',
     'cuchill',
     'lamin',
     'funcion',
     'mejor',
     'piel',
     'delic',
     'ademas',
     'apur',
     'asi',
     'dediqu',
     'investig',
     'decid',
     'compr',
     'dud',
     'si',
     'compr',
     'braun',
     'panasonic',
     'obvi',
     'decid',
     'segund',
     'consegu',
     'prestacion',
     'potenci',
     'panasonic',
     'braun',
     'pag',
     'dobl',
     'sab',
     'si',
     'final',
     'iba',
     'ir',
     'bien',
     'decid',
     'panasonic',
     'experient',
     'despues',
     'vari',
     'seman',
     'uso',
     'excelent',
     'maquin',
     'buen',
     'calid',
     'material',
     'robust',
     'ergonom',
     'result',
     'apur',
     'buenisim',
     'potent',
     'encim',
     'bat',
     'carg',
     'tan',
     'sol',
     'hor',
     'llev',
     'cinc',
     'afeit',
     'sig',
     'bat',
     'maravill',
     'limpi',
     'sup',
     'facil',
     'pued',
     'usar',
     'sec',
     'moj',
     'debaj',
     'duch',
     'espum',
     'gel',
     'afeit',
     'ah',
     'import',
     'dañ',
     'absolut',
     'piel',
     'rojec',
     'escozor',
     'tra',
     'recort',
     'patill',
     'bigot',
     'aun',
     'usad',
     'afeit',
     'bien',
     'cuell',
     'car',
     'dud',
     'recom',
     'cien',
     'cien',
     'pas',
     'ver',
     'mal',
     'apur',
     'cuchill',
     'rotatori',
     'philips.',
     'simpl',
     'vist',
     'm',
     'pareci',
     'buen',
     'cam',
     'm',
     'decepcion',
     'mux',
     'prim',
     'bañ',
     'entro',
     'agu',
     'adi',
     'cam',
     'k',
     'prob',
     'primer',
     'acuari',
     'd',
     'cas',
     '....',
     'm',
     'yev',
     'gran',
     'decepcion',
     '.....',
     'pas',
     'rat',
     'personaj',
     'ningun',
     'dialog',
     'sos',
     'protagon',
     'falt',
     'cinc',
     'hervor',
     'narrat',
     'exist',
     'fiabil',
     'histor',
     'total',
     'nula.el',
     'fabric',
     'deci',
     'compat',
     'd610',
     'maner',
     'igual',
     'hag',
     'mal',
     'compact',
     'si',
     'funciona.el',
     'libr',
     'buen',
     'condicion',
     'inclu',
     'cd',
     'audi',
     'sol',
     'serv',
     'mediasbu',
     'aspect',
     'falt',
     'fortalez',
     'util',
     'marc',
     'cort',
     'circul',
     'veo',
     'poc',
     'fuerz',
     'resistent',
     'hac',
     'mas',
     'pruebas.expl',
     'form',
     'simpl',
     'sencill',
     'pensamient',
     'grand',
     'filosof',
     'grand',
     'corrient',
     'permit',
     'entend',
     'concept',
     'evolucion',
     'pas',
     'pas',
     'histori',
     'simpl',
     'entretenidarel',
     'calid',
     'preci',
     'buen',
     'son',
     'fuert',
     'man',
     'libr',
     'escuch',
     'perfect',
     'lector',
     'usb',
     'funcion',
     'bien',
     'conexion',
     'radi',
     'movil',
     'sencill',
     'conect',
     'radi',
     'tambienel',
     'funcion',
     'product',
     'correct',
     'buen',
     'diseñ',
     'tamañ',
     'correct',
     'calid',
     'preci',
     'acord',
     'esperadoest',
     'bien',
     'aunqu',
     'diferent',
     'presentd',
     'poquit',
     'caroy',
     'si',
     '3d',
     'genial',
     'niñ',
     'concienci',
     'mux',
     'mayor',
     'peli',
     'buen',
     'efect',
     'ser',
     'dibuj',
     'buen',
     'calidadconsider',
     'histori',
     'entreten',
     'llen',
     'creativ',
     'cont',
     'lenguaj',
     'clar',
     'armoni',
     'flu',
     'denot',
     'autor',
     'pose',
     'vast',
     'conoc',
     'cultur',
     'puebl',
     'nordic',
     'empec',
     'leerl',
     'cautiv',
     'conoc',
     'destin',
     'ivarr',
     'si',
     'gust',
     'desarroll',
     'descenlace.cu',
     'van',
     'mand',
     'sirv',
     'prevision',
     'recib',
     'inform',
     'plaz',
     'entreg',
     'cas',
     'fij',
     'fech',
     'previst',
     'deb',
     'cumpl',
     'si',
     'culp',
     'editorial',
     'cre',
     'buen',
     'polit',
     'car',
     'client',
     'vuelv',
     'hac',
     'caso.recomend',
     'si',
     'amant',
     'clasic',
     'edicion',
     'bien',
     'contien',
     'peli',
     'par',
     'idiom',
     'tres',
     'mas',
     'apr',
     'conten',
     'extras',
     'maloq',
     'ue',
     'edicion',
     'bden',
     'estet',
     'comod',
     'ajust',
     'esper',
     'calz',
     'result',
     'comod',
     'cuant',
     'usas.est',
     'marc',
     'demasi',
     'renombr',
     'calid',
     'diseñ',
     'tipic',
     'sobri',
     'encuadern',
     'bien',
     'hoj',
     'demasi',
     'gramaj',
     'pued',
     'transparent',
     'si',
     'escrib',
     'tint',
     'cuid',
     'tamañ',
     'aunqu',
     'pong',
     'tamañ',
     'grand',
     'realid',
     'a5un',
     'muj',
     'millonari',
     'elizabeth',
     'taylor',
     've',
     'ventan',
     'asesinat',
     'viej',
     'caseron',
     'enfrent',
     'mar',
     'laurenc',
     'harvey',
     'mejor',
     'amig',
     'cre',
     'asim',
     'polic',
     'recurr',
     'insistent',
     'sig',
     'insist',
     'exasper',
     'polic',
     'final',
     'cre',
     'muj',
     'alucin',
     'cabal',
     'aunqu',
     'pelicul',
     'dej',
     'monton',
     'cab',
     'suelt',
     'pued',
     'neg',
     'entretien',
     'maxim',
     'ultim',
     '20',
     'minut',
     'da',
     'gir',
     'bastant',
     'inesper',
     'ningun',
     'obra',
     'maestr',
     'men',
     'si',
     'quier',
     'pas',
     'rat',
     'pelicul',
     'conseguira.',
     'pen',
     'piez',
     'tabler',
     'carton',
     '....',
     'dobl',
     'esquin',
     'facil',
     'siend',
     'jueg',
     'viaje.diseñ',
     'bonit',
     'buen',
     'preci',
     'montaj',
     'demasi',
     'complej',
     'opinion',
     'part',
     'podr',
     'mejor',
     'tornill',
     'atornill',
     'direct',
     'chasis',
     'plastic',
     'tamañ',
     'bastant',
     'pequeñ',
     'just',
     '2/3',
     'años.cumpl',
     'funcion',
     'perfect',
     'preci',
     'estupend',
     'uso',
     'micr',
     'pc',
     'habl',
     'skype',
     'similar',
     'oye',
     'clar',
     'ruid',
     'si',
     'requier',
     'equip',
     'profesional',
     'recom',
     'uso',
     'oficin',
     'cas',
     'prob',
     'exterior',
     'asi',
     'sensibl',
     'ruid',
     'ambientales.l',
     'sill',
     'buen',
     'calid',
     'buen',
     'acab',
     'embarg',
     'devolv',
     'encontr',
     'defect',
     'imped',
     'normal',
     'funcion',
     'problem',
     'amazon',
     'devolverlo.tod',
     'satisfactori',
     'plaz',
     'entreg',
     'estipul',
     '.muy',
     'recomend',
     'cumpl',
     'tod',
     'caracterist',
     'anunci',
     'contentoel',
     'product',
     'fot',
     'facil',
     'manej',
     'buen',
     'calid',
     'recomend',
     '100',
     'dud',
     'volv',
     'comprar.n',
     'ningun',
     'peg',
     'destac',
     'aunqu',
     'vez',
     'puest',
     'algun',
     'zon',
     'qued',
     'poquit',
     'torc',
     'huec',
     'cam',
     'conexion',
     'cabl',
     'cargador',
     'coincid',
     'perfect',
     'boton',
     'bloqu',
     'pantall',
     'flechit',
     'sub',
     'baj',
     'volum',
     'cubr',
     'fund',
     'ahor',
     'mas',
     'dur',
     'puls',
     'esper',
     'fund',
     'mas',
     'blandit',
     'elast',
     'amortigu',
     'caid',
     'bastant',
     'rig',
     'men',
     'mas',
     'proteg',
     'qued',
     'movil',
     'fund',
     'da',
     'sensacion',
     'si',
     'cae',
     'raj',
     'lad',
     'hech',
     'call',
     'much',
     'altur',
     'movil',
     'fund',
     'raj',
     'pantall',
     'aunqu',
     'sig',
     'siend',
     'funcional.hac',
     'comet',
     'principal',
     'mol',
     'lumbrer',
     'diseñ',
     'deb',
     'hab',
     'pens',
     'tap',
     'sac',
     'derram',
     'caf',
     'interior',
     'pues',
     'llev',
     'lengüet',
     "''",
     'entra',
     'dentr',
     'molinill',
     'engorr',
     'ahi',
     'darl',
     'sol',
     'estrell',
     'envi',
     'embalaj',
     'estrellas.com',
     'dvd',
     'tempor',
     'region',
     'ped',
     'product',
     'pes',
     'avis',
     'relat',
     'region',
     'aparec',
     'algun',
     'opinion',
     'usuari',
     'lleg',
     'disc',
     'indic',
     'region',
     'asi',
     'abrirl',
     'escrib',
     'proveedor',
     'traves',
     'amazon',
     'indic',
     'prob',
     'insist',
     'podr',
     'devolv',
     'si',
     'funcion',
     'dec',
     'funcion',
     'atencion',
     'marveli',
     'sid',
     'impec',
     'unic',
     'fall',
     "''",
     'doblaj',
     'español',
     'sudamerican',
     'suen',
     'rar',
     'aqu',
     'demas',
     'perfecto.està',
     'bien',
     'cumpl',
     'funcion',
     'falt',
     'cierr',
     'seri',
     'perfect',
     'b',
     'b',
     'b',
     'b',
     'b',
     'b',
     'b',
     'bproduct',
     '100',
     'recomend',
     'ideal',
     'orden',
     'toshib',
     'seri',
     'c55',
     'perfect',
     'bolsill',
     'lateral',
     'vien',
     'genial',
     'guard',
     'cargador',
     'raton.el',
     'dia',
     'septiembr',
     'realiz',
     'ped',
     'pag',
     '10€',
     'gast',
     'envi',
     'lug',
     'gast',
     'envi',
     'estand',
     'envi',
     'dia',
     'siguient',
     'paquet',
     'lleg',
     'dias',
     'gust',
     'devolv',
     'gast',
     'envi',
     'equipar',
     'gast',
     'envi',
     'estand',
     'product',
     'lleg',
     'dia',
     'lleg',
     'juev',
     'septiembr',
     'graci',
     'demas',
     'product',
     'adecu',
     'necesidades.muy',
     'recomend',
     'viaj',
     'quier',
     'carg',
     'pes',
     'tripod',
     'sup',
     'resistent',
     'aguant',
     'pes',
     'cam',
     'reflex',
     'objet',
     'medi',
     'pes',
     'adapt',
     'cualqui',
     'siti',
     'pued',
     'coloc',
     'sup',
     'recomendable.d',
     'preis',
     'war',
     'via',
     'germany',
     'günstig',
     'und',
     'die',
     'belieferung',
     'trotzdem',
     'sehr',
     'schnell',
     'werd',
     'wied',
     'bei',
     'ihnen',
     'kauf',
     'und',
     'soi',
     'weiterempfehl',
     'compr',
     'noki',
     '5800',
     'bat',
     'original',
     'apen',
     'dur',
     'unas',
     'hor',
     'tras',
     'años',
     'funcion',
     'bat',
     'dur',
     'sorprendent',
     'dia',
     'aunqu',
     'dud',
     'si',
     'bat',
     'dur',
     'tan',
     'propi',
     'movil',
     'fund',
     'bat',
     'tan',
     'rapidamente.h',
     '10',
     'metr',
     'funcion',
     'trat',
     'postvent',
     'sid',
     'horribl',
     'ped',
     'devolv',
     'dos',
     'product',
     'compr',
     'dand',
     'vuelt',
     'dias',
     'final',
     'permit',
     'devolu',
     'dad',
     'ningun',
     'facil',
     'explic',
     'hac',
     'devolu',
     'envi',
     'reembols',
     'ahor',
     'hac',
     'carg',
     'aconsej',
     'compr',
     'vendedor',
     'devolv',
     'perd',
     'it',
     'about',
     '10',
     'meters',
     'not',
     'working',
     'the',
     'sal',
     'deal',
     'been',
     'horribl',
     'i',
     'asked',
     'return',
     'one',
     'of',
     'the',
     'two',
     'products',
     'i',
     'bought',
     'them',
     'and',
     'i',
     'wer',
     'wandering',
     'around',
     'for',
     'days',
     'in',
     'the',
     'end',
     'allow',
     'the',
     'return',
     'but',
     'i',
     'hav',
     'not',
     'giv',
     'any',
     'explanation',
     'of',
     'how',
     'easily',
     'or',
     'mak',
     'refund',
     'i',
     'had',
     'to',
     'send',
     'cod',
     'and',
     'do',
     'not',
     'tak',
     'car',
     'now',
     'i',
     'advis',
     'you',
     'not',
     'to',
     'buy',
     'anything',
     'from',
     'the',
     'sell',
     'as',
     'you',
     'hav',
     'to',
     'return',
     'thes',
     'lost',
     '.el',
     'prim',
     'contact',
     'maquinill',
     'afeit',
     'buen',
     ...]




```python
vectorizer.fit(comentarios_train)
```

    C:\Users\isart\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:484: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
      warnings.warn("The parameter 'token_pattern' will not be used"
    




    CountVectorizer(stop_words=['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los',
                                'del', 'se', 'las', 'por', 'un', 'para', 'con',
                                'no', 'una', 'su', 'al', 'lo', 'como', 'más',
                                'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí',
                                'porque', ...],
                    tokenizer=<function tokenize at 0x000002704C0EEA60>)




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
#Modelo de regresión
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
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:  2.5min remaining:  7.4min
    [Parallel(n_jobs=-1)]: Done   5 out of   8 | elapsed:  2.5min remaining:  1.5min
    [Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:  2.5min remaining:    0.0s
    [Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:  2.5min finished
    C:\Users\isart\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




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
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed: 23.5min remaining: 35.2min
    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed: 23.6min finished
    




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
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:   34.4s remaining:   51.7s
    [Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   34.7s finished
    




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
    

El mejor modelo ha resultado ser la regresión logística

# Obtención de los Tweets con Tweepy


```python
import tweepy
import json
import csv
```


```python
consumer_key = "Jo1lDBKNxj2CqszGMJVYbNS9h"
consumer_secret = "Occ91jX6K4SgQHhx7V3zDfMdy3FmI02VzSgR2POBD7sSYU1u31"
access_token = "393446438-t0lqRajYB8pnFXVhi58GNCjKekmBsdQaXtR40TXb"
access_secret = "POyeQU5o3SfAcX32urObMJsJ9L3ywASfXVphZf1sAG0AS"
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
```


```python
#hastags
#Abrimos o creamos un archivo donde guardar los tweets
csvFile = open('iberdrola.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)


for tweet in tweepy.Cursor(api.search,q="iberdrola",  
                           count=10,
                           #lang="sp",
                           since="2021-01-01").items(1000):
    
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
    
    ➡️ @Rugby_Cisneros (b)15-17 @RugbyfemSevilla…
    2021-05-05 10:13:09 RT @soraya264: Mis fotos Liga Iberdrola 1-5-2021 @Rugby_femenino Jornada 9 @Rugby_Cisneros 15 @RugbyfemSevilla 17 https://t.co/gVzRQHNn1j
    2021-05-05 10:12:51 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @Rugb…
    2021-05-05 10:10:06 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:07:27 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:03:46 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:01:27 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 10:01:25 @osiris1278 A la mia de 7 años, le toca pagar a la hacienda foral 10 centimos, creo que ya paga mas que iberdrola.
    2021-05-05 09:59:42 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 09:58:30 ¿A ver cómo te explico yo esto, Iberdrola? Devuélveme la luz.
    2021-05-05 09:57:55 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 09:56:57 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-05 09:56:55 Muy interesante #DesayunoVP sobre #sostenibilidad. Ha acudido nuestro pte, @MiguelBurdeos, junto a @GVAivace… https://t.co/toUS3EmRbN
    2021-05-05 09:53:30 📍Hoy nuestra directora de RSC y Capital Humano, @pilarblayarsc ha participado en el #DesayunoVP sobre… https://t.co/OdI5aKfNGL
    2021-05-05 09:50:24 RT @adrianomones: Todo listo para presentar el #MOVES3 con @Arturopdelucia y @AEDIVE  @IDAEenergia @GobAsturias @FundacionFaen @iberdrola @…
    2021-05-05 09:46:45 RT @GimnasiaFdex: Buenos días, esta semana tenemos ¡excelentes noticias para el Club Gimnástico Almendralejo y @clubhadar! #gimansiaartisti…
    2021-05-05 09:44:10 Noticias de Bizkaia en @SueltaLaOlla, magazine de @halabedi y @97irratia 
    Desahucios, desalojos, desokupa y ertzain… https://t.co/AoZYltA8tI
    2021-05-05 09:41:47 RT @EuromecaSL: Electrificar la producción #cerámica https://t.co/EHAumtEA7f Vía: @csinformacion
    2021-05-05 09:41:29 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @C…
    2021-05-05 09:33:06 @DSR_Gw @juanrallo Un cargo en iberdrola,  un programa en la tv de roures,  un cargo en Telefonica. 
    
    Todo apesta
    2021-05-05 09:32:00 Comprometidos con la #AcciónClimática ➡️ Aumentamos un 13,3% la producción con #renovables en el primer trimestre d… https://t.co/TBk8nD7kxL
    2021-05-05 09:31:17 RT @lmransanz: @MarianoFuentesS Tras el fin de la cesión que nunca debió ser, @iberdrola aún okupa como parking la parcela dotacional públi…
    2021-05-05 09:31:05 Ante todas las malas experiencias pasadas en mi breve paso por Iberdrola, me marcho de esta compañía: Cambio no con… https://t.co/cfr2tzNHKQ
    2021-05-05 09:27:26 Y todo esto ya sin meternos en los intereses de las empresas, cuantos ex-políticos hay en las cúpulas de Indra, Ibe… https://t.co/FhJ74vFo5V
    2021-05-05 09:25:11 Electrificar la producción #cerámica https://t.co/EHAumtEA7f Vía: @csinformacion
    2021-05-05 09:24:17 RT @sapiensenergia: 🌀 Participamos en el #DesayunoVP sobre #sostenibilidad, junto a @GVAivace, @GrupoGimeno, 
    @CEV_CV, @infoHIDRAQUA, @FVal…
    2021-05-05 09:23:25 @MarcosFutFem Me hace gracia cuando se les llena la boca diciendo k apuestan x el futfem. Es cierto tambien k Gol y… https://t.co/ySD4MiFgEd
    2021-05-05 09:22:34 SP Energy Networks and Siemens trial first ‘clean-air’ UK substation ♻️⚡️🇬🇧
    ➡️https://t.co/8wYUPcCt9A
    #transformer… https://t.co/FhVzM5kcCH
    2021-05-05 09:19:15 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 09:17:00 Vuelve la #poesía a #Instagram con Zenda e @iberdrola 
    
    Tenemos nuevo concurso: #versosprimaverales
    
    🗓️ Desde el lu… https://t.co/tGD2xIEm6W
    2021-05-05 09:15:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-05 09:14:43 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 09:14:15 @CgYimii @noelia_nessi00 Pues claro que es imprescindible, si estamos en la puta ruina tio 😂 y da gracias de que es… https://t.co/GGyMbYCs06
    2021-05-05 09:14:11 @JGonzalez_GCF Las mismas que votan a los que se meten en iberdrola, endesa.. 🙃
    2021-05-05 09:08:19 En Pontevedra o centro perfectamente ordeado, limpo e con zonas verdes. Semella un deses espazos que saen nos spots… https://t.co/KQ2KwFU6k1
    2021-05-05 09:03:32 RT @sapiensenergia: 🌀 Participamos en el #DesayunoVP sobre #sostenibilidad, junto a @GVAivace, @GrupoGimeno, 
    @CEV_CV, @infoHIDRAQUA, @FVal…
    2021-05-05 09:02:37 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 09:02:32 🌀 Participamos en el #DesayunoVP sobre #sostenibilidad, junto a @GVAivace, @GrupoGimeno, 
    @CEV_CV, @infoHIDRAQUA,… https://t.co/a72Xy5umSi
    2021-05-05 09:01:45 RT @alvaro_j_campos: Por si había alguna duda de que eran el lado oscuro
    2021-05-05 09:00:44 ENAGAS 📊
    
    Si desde el año 2014, compras #Enagas cuando ha llegado a 17€ por acción, siempre habrías Ganado Dinero👇… https://t.co/w1D6Ylhe3g
    2021-05-05 08:58:12 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 08:58:10 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-05 08:54:52 RT @AlhamaElPozo: ▶Jornada 6 - Playoffs de ascenso a Primera Iberdrola
    
    🆚@GranadaFemenino 
    
    🗓 Domingo 9 de Mayo
    ⏰12:00 h
    🏟 Estadio de La Ju…
    2021-05-05 08:54:21 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @Rugb…
    2021-05-05 08:54:06 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @CR…
    2021-05-05 08:53:46 @RFEVB @VoleyFemAbs @teledeporte @PalomadelrioTVE @c_corcelles @deportegob @COE_es @iberdrola @Luanvi… https://t.co/6xJ0UDLvB9
    2021-05-05 08:53:24 Lamento informar a posibles clientes que en @iberdrola  y @TuIberdrola tienen un pésimo servicio de atención a sus… https://t.co/rPfpWzqPfY
    2021-05-05 08:52:40 @NatyReynos0 Nuestra federación funciona así, también prometieron que se darían todos los partidos de la primera Ib… https://t.co/hY8jgHl0Pk
    2021-05-05 08:49:25 RT @DavidGalanBolsa: 🔊📻Audio de mi Consultorio de #bolsa semanal en Capital Radio+Periscope 23 de marzo
    Analicé índices,CaixaBank,Santander…
    2021-05-05 08:46:39 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @C…
    2021-05-05 08:44:36 RT @ConsuorF: La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en #Renteria al dar…
    2021-05-05 08:42:07 La última víctima de Lemóniz se produjo poco después de Ángel, fue el niño de diez años Alberto Muñagorri en… https://t.co/IsAlhMGOYQ
    2021-05-05 08:39:56 ⚠️ El Bono Social de Iberdrola es un descuento aplicado para consumidores considerados vulnerables, que aplica desc… https://t.co/sVX8JlJWpw
    2021-05-05 08:37:20 RT @adrianomones: Todo listo para presentar el #MOVES3 con @Arturopdelucia y @AEDIVE  @IDAEenergia @GobAsturias @FundacionFaen @iberdrola @…
    2021-05-05 08:36:36 RT @SmartGreenPeopl: 🔥 SORTEO⁣
    ¡Welcome, mayo! Comienza la Operación bikini💪🏻. Controla todos tus movimientos con este reloj solar y ¡recar…
    2021-05-05 08:36:07 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @San…
    2021-05-05 08:35:45 Todo listo para presentar el #MOVES3 con @Arturopdelucia y @AEDIVE  @IDAEenergia @GobAsturias @FundacionFaen… https://t.co/4xwfUkJF88
    2021-05-05 08:33:49 RT @EduardoMontejoA: @AEDIVE @AsociacionAUVE @AVVEinfo @electromaps Nuevo punto de recarga rápida (dos tríos) de Iberdrola, ya operativo en…
    2021-05-05 08:32:55 Quanta ingenuïtat… Pedrito @sanchezcastejon fracasat? Ell ja té l’única cosa que volia: la seva pensió vitalicia i… https://t.co/Y32xAQGZVW
    2021-05-05 08:31:22 Excelente debate en #DesayunoVP donde la Sostenibilidad es estrategia junto a @bbva, @GVAivace @GrupoGimeno @CEV_CV… https://t.co/49ugGbZYGw
    2021-05-05 08:29:41 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @C…
    2021-05-05 08:29:37 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @CR…
    2021-05-05 08:29:34 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @San…
    2021-05-05 08:29:32 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @Rugb…
    2021-05-05 08:23:42 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-05 08:23:14 RT @SmartGreenPeopl: 🔥 SORTEO⁣
    ¡Welcome, mayo! Comienza la Operación bikini💪🏻. Controla todos tus movimientos con este reloj solar y ¡recar…
    2021-05-05 08:18:06 RT @AlhamaElPozo: ▶Jornada 6 - Playoffs de ascenso a Primera Iberdrola
    
    🆚@GranadaFemenino 
    
    🗓 Domingo 9 de Mayo
    ⏰12:00 h
    🏟 Estadio de La Ju…
    2021-05-05 08:17:33 @MMPRG @rintereconomia @CriadoSusana #Iberdrola es una buena opción para mantener sin sobresaltos, para ahorradores… https://t.co/0QaSLES751
    2021-05-05 08:12:25 A las 10,45 toda la información de Bizkaia en su corresponsalía. Temas de hoy:
    Amenazas de desahucios, desalojos ap… https://t.co/DMls5ybotB
    2021-05-05 08:11:46 ▶Jornada 6 - Playoffs de ascenso a Primera Iberdrola
    
    🆚@GranadaFemenino 
    
    🗓 Domingo 9 de Mayo
    ⏰12:00 h
    🏟 Estadio de… https://t.co/5UPARkPl3h
    2021-05-05 08:07:37 @naiindi @iberdrola No porque era mi lamparita inalámbrica!!!
    2021-05-05 08:05:46 RT @xavidelaossa: Las gloriosas están a tres partidos de subir a Primera Iberdrola por primera vez en su historia. El club monta un vídeo q…
    2021-05-05 08:05:43 RT @SilverBulletPR: Van Oord Starts Installing Saint-Brieuc Offshore Wind Farm Foundations
    https://t.co/CrYIkkmjlA
    #OffshoreWind #France #S…
    2021-05-05 08:05:05 Van Oord Starts Installing Saint-Brieuc Offshore Wind Farm Foundations
    https://t.co/CrYIkkmjlA
    #OffshoreWind… https://t.co/I6sflcPowI
    2021-05-05 08:01:34 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @CR…
    2021-05-05 07:58:19 RT @1961_pilar: "Hay rincones para leer que cobijan".
    https://t.co/9jIIptcjdQ
    
    Presentada al III Concurso de fotografía en Instagram #paraí…
    2021-05-05 07:58:08 @PabloIglesias ha entrado en el grupo de consejero de Iberdrola. Viva el comunismo jajajajaja tus cojones Pablito
    2021-05-05 07:57:46 Another #European #magnet heading to #ITER
    18 Toroidal Field coils will confine the hot plasma🔥
    Each of these 🧲 is… https://t.co/LQhb3sI5dv
    2021-05-05 07:56:26 RT @SmartGreenPeopl: 🔥 SORTEO⁣
    ¡Welcome, mayo! Comienza la Operación bikini💪🏻. Controla todos tus movimientos con este reloj solar y ¡recar…
    2021-05-05 07:54:16 RT @1961_pilar: "Hay rincones para leer que cobijan".
    https://t.co/9jIIptcjdQ
    
    Presentada al III Concurso de fotografía en Instagram #paraí…
    2021-05-05 07:50:22 RT @iberdrola: ✔️TERMINADA 🔚 El astillero de @NavantiaOficial en Fene ensambla la primera jacket para nuestro parque eólico marino de Saint…
    2021-05-05 07:46:03 RT @MarchenaSecreta: Enrique Mateo Martín es un estudiante de cuarto de la ESO del IES López de Arenas que ha diseñado un prototipo de saté…
    2021-05-05 07:45:24 RT @xavidelaossa: Las gloriosas están a tres partidos de subir a Primera Iberdrola por primera vez en su historia. El club monta un vídeo q…
    2021-05-05 07:43:56 @MariaPSoroa @TuIberdrola @iberdrola Hola María. Te facilitamos información en este enlace https://t.co/UCpkdhoYvS Saludos
    2021-05-05 07:43:24 RT @AVANGRID: Our Q1'21 results highlight our strong start to the year &amp; continued positive momentum. Contributing to these results are our…
    2021-05-05 07:42:56 Las gloriosas están a tres partidos de subir a Primera Iberdrola por primera vez en su historia. El club monta un v… https://t.co/XF63G38OUe
    2021-05-05 07:38:42 El Club Daysan sigue sumando medallas nacionales. Liga Nacional Iberdrola de clubes de Gimnasia Rítmica. 
    Más info:… https://t.co/cH6lZLD5eH
    2021-05-05 07:35:45 @davedidave @orriolsderipoll España tiene, con Iberdrola, la mayor utillity del mundo. La aseguradora líder en Amér… https://t.co/vpHQ4AxYD4
    2021-05-05 07:28:45 RT @NiHostiax: Qué es madrid?? Yo solo conozco lucía gonzález manos libres liga iberdrola 2021
    2021-05-05 07:28:21 RT @scalenergy: RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerd…
    2021-05-05 07:25:47 @BugenhagenQQ @Sophistidomme Ya venía de hacer televisión. También te parecerá puerta giratoria que vuelva a dar cl… https://t.co/zPalX7BHrT
    2021-05-05 07:19:19 Ahora a mofarse y criticar que Iglesias tenga un trabajo en una TV
    
    Pero a los políticos de PSOE y PP que se van a… https://t.co/1g190MppBQ
    2021-05-05 07:19:07 The New York Times alaba a Iberdrola como promotora de la energía eólica y solar | https://t.co/BPhBoBW3h8… https://t.co/UWZnO50Alw
    2021-05-05 07:12:06 @TraderSpain @martinvars y puedes contratar con una comercializadora que garantiza a través de certificados de orig… https://t.co/8Hk7RWWqZg
    2021-05-05 07:05:15 @iberdrola @NavantiaOficial Está bien lo de los molinillos en medio del mar, pero entre vosotros y @iDE_Iberdrola ,… https://t.co/AGP7qyK4AS
    2021-05-05 07:03:02 El @BalonmanoAdesal y su consigna: "Ya no valen más errores". La capitana, Ángela Ruiz, lanza un llamamiento al gru… https://t.co/u5uBshdGVZ
    2021-05-05 07:02:58 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-05 07:00:58 RT @UCAM_Deportes: 🔝 Así mismo, felicitar a dos supertalentos del hockey hierba español 🏑🇪🇸 que forman parte también de la familia @UCAM:…
    2021-05-05 07:00:49 RT @UCAM_Deportes: 🏑 Enhorabuena a nuestras jugadoras de hockey hierba @9marialopez y @Beaperlag10 por proclamarse CAMPEONAS DE LIGA IBERDR…
    2021-05-05 07:00:10 RT @elCugatenc: ESPORTS (@SantCugatCreix) 🏑 || El primer equip femení d'hoquei sobre herba del @ClubJunior1917 és subcampió de la Lliga Ibe…
    2021-05-05 06:59:31 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-05 06:52:36 RT @elEconomistaes: 🔝Su estrategia de inversión en energía limpia y redes llevará a @iberdrola a:
    
    👉Ser una compañía "neutra en carbono" en…
    2021-05-05 06:45:00 Buenos días, esta semana tenemos ¡excelentes noticias para el Club Gimnástico Almendralejo y @clubhadar!… https://t.co/syTOGTyN14
    2021-05-05 06:41:03 RT @JulianMaciasT: Roberto Salinas, además de responsable también preside Business México Forum, financiado por grandes empresas como Iberd…
    2021-05-05 06:32:01 Iberdrola - Daily
    #ibex35
    Short Term: €11.2 / €11.7. Small range €11.2 / €11.4.
    Medium and Long Term: €11.2 / €11.6… https://t.co/niXDUb5NRt
    2021-05-05 06:26:57 Seguimos avanzando con los programas Frida Rural y Frida Joven, financiados por @FundlaCaixa y fundación @iberdrola… https://t.co/EOenI1w9kM
    2021-05-05 06:23:36 RT @castillayleon: 🎨 ¿Te imaginas pasear entre cuadros de Velázquez, Goya o Tiziano? La plaza de la Concordia de #Salamanca ha sido el prim…
    2021-05-05 06:22:33 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-05 06:21:13 @mcgregor_0704 @zumi_de_manzana @JorgeVarade Cuando ayudas a todas las televisiones pero es una minoritaria la que… https://t.co/pQYfvtYXg2
    2021-05-05 06:20:41 RT @ferugby: #FERugby |🚨Tenemos buenas noticias: el rugby femenino crece con el nuevo Campeonato de España M14 y M16 ‼️#MujeresEnRugby #Des…
    2021-05-05 06:14:41 RT @TRIATLONSP: Semana de #campeonatos. Nacionales de #Duatón #SuperSprint #RelevosParejas #SuperSprint2x2 en #LaNucía #SomosTriatlon  
    
    In…
    2021-05-05 06:08:05 RT @randyshoup: The Green and Sustainable Future of Transport by @Doctor_Astro 🚅🚁✈️🚀
    
    Interesting and realistic overview of a carbon-free f…
    2021-05-05 06:06:13 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-05 06:02:00 ✔️TERMINADA 🔚 El astillero de @NavantiaOficial en Fene ensambla la primera jacket para nuestro parque eólico marino… https://t.co/e6cKmCOR02
    2021-05-05 06:00:13 RT @ETC_energy: Launched this week: The #GreenHydrogen Catapult initiative sees ETC members @Envisioncn, @Iberdrola, @Orsted &amp; @Snam, along…
    2021-05-05 06:00:00 ESPORTS (@SantCugatCreix) 🏑 || El primer equip femení d'hoquei sobre herba del @ClubJunior1917 és subcampió de la L… https://t.co/l6hqnGuXkR
    2021-05-05 05:57:28 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-05 05:54:53 RT @addbreizhou: Tout y est expliqué. Un scandale financier, ecologique, économique.@barbarapompili arrêtez de vous entêter avec ce projet,…
    2021-05-05 05:53:57 RT @EMUASA_Clientes: Nuestra apuesta por la energía renovable, limpia y respetuosa con el medio ambiente se ve legitimada por este certific…
    2021-05-05 05:53:12 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-05 05:53:04 @_LuisMMora_ @bruuj00 @Rodolfo30534460 @aracelibs @rocionahle Teniendo hidroeléctricas y usándolas a un porcentaje… https://t.co/tOVycHbHKj
    2021-05-05 05:49:09 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-05 05:36:09 Si Pablo Iglesias ficha ahora por Iberdrola o Gas Natural sería tan coherente como la casa que se compró.
    Y sí, tie… https://t.co/Uw4wP9H4lV
    2021-05-05 05:30:24 @radhalathgupta Primera Iberdrola
    2021-05-05 05:06:10 RT @iberdrola: Tecnología 🔝 para revisar el parque eólico marino de East Anglia One ➡️ El dron salmantino @Aracnocoptero inspeccionará una…
    2021-05-05 04:35:08 RT @1961_pilar: "Hay rincones para leer que cobijan".
    https://t.co/9jIIptcjdQ
    
    Presentada al III Concurso de fotografía en Instagram #paraí…
    2021-05-05 04:34:20 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-05 04:28:34 @luispazos1 Eran contratos  leoninos, sobretodo  con OHl, iberdrola, tantita madre
    2021-05-05 04:21:59 RT @starrett_mike: Sharing a great job in US Offshore Wind located in Boston. Good potential fit for advocacy/stakeholder-type backgrounds…
    2021-05-05 03:54:39 RT @iberdrolamex: De acuerdo con el estudio de percepción #ImpulsoSTEM, el 90% de la juventud oaxaqueña desea continuar con sus estudios a…
    2021-05-05 03:46:35 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-05 03:05:27 RT @SmartJoules: Look forward to seeing many more #climatedeals like this where companies march fast toward #zerocarbon across all their fa…
    2021-05-05 02:57:35 @usamahzaid I think he means this. https://t.co/KhccIshnZ6.
    2021-05-05 02:47:38 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-05 02:33:42 The New York Times' alaba a Iberdrola como promotora de la energía eólica y solar (España) |… https://t.co/9Q4m1Ekzcj
    2021-05-05 02:22:13 @Asociacion_DEC @aegonseguros @eoi @ImqEuskadi @iDE_Iberdrola @la_Mutua @Solimat72 @BainAlerts @igslasalle… https://t.co/UwM2tqbkcC
    2021-05-05 01:48:40 A Bet 20 Years Ago Made It the Exxon of Green Power https://t.co/17QhVrYlNq A Spanish company wisely investing in t… https://t.co/w4F5u8mSh1
    2021-05-05 01:47:05 RT @starrett_mike: Sharing a great job in US Offshore Wind located in Boston. Good potential fit for advocacy/stakeholder-type backgrounds…
    2021-05-05 01:34:06 RT @la_informacion: 📰 La portada de @la_informacion de este martes:
    
    👉 Iberdrola revisa a fondo el protocolo por corrupción bajo la lupa de…
    2021-05-05 01:25:52 RT @Alfons_ODG: Històric! Una victòria dels moviments socials representats per l'Aliança Contra la Pobresa Energètica (@APE_Cat). 
    
    5 anys…
    2021-05-05 01:12:41 Sharing a great job in US Offshore Wind located in Boston. Good potential fit for advocacy/stakeholder-type backgro… https://t.co/ORZ6H6HGWU
    2021-05-05 00:45:24 @aracelibs @rocionahle La pendejez se ensaña con algunas personas más que otras .
    De que año y modelo es tu Tesla?… https://t.co/CNELDZx6dY
    2021-05-05 00:45:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-05 00:20:30 @floricote @InfotravelsRO @mosomero @JorgeBustos1 No a Endesa o Iberdrola
    2021-05-05 00:04:20 @MumisMero @MaxKaiser75 En si en base a qué inversión extranjera te refieres? Hablas sobre Iberdrola o las demás co… https://t.co/NDrVfWfltI
    2021-05-04 23:57:41 @Marcusin77 @jordievole @davidhaz14 Y a florentino Pérez y a bbva y a iberdrola y a ... yo a Ana Rosa quintana no l… https://t.co/IKfH6J72sH
    2021-05-04 23:45:02 ... Iberdrola, Enel X, la compañía automovilística Volvo Cars, Ikea o Uber son algunas de las empresas que han soli… https://t.co/ufkI4J5Qjn
    2021-05-04 23:40:54 RT @DESCOLONIZANTE: "el sobre ano" va a resultar peor que #iberdrola 
    
    mas facil seria preguntar:
    
    ¿quien no esta trabajando o colaborando…
    2021-05-04 23:33:11 @ronovmen @rocionahle @CFEmx @Pemex Ok ... mi rey ! Aún así Toyota, VW,  Nissan, etc... No sé van porque el peso fr… https://t.co/IqVAqKiExy
    2021-05-04 23:28:33 The New York Times alaba a Iberdrola como promotora de la energía eólica y solar | https://t.co/ylC9KNQjhR https://t.co/G3p4cxFQoS
    2021-05-04 23:25:46 RT @ferugby: #FERugby |🚨Tenemos buenas noticias: el rugby femenino crece con el nuevo Campeonato de España M14 y M16 ‼️#MujeresEnRugby #Des…
    2021-05-04 23:20:03 @Carmen45465296 @EsmeAbenia @PabloIglesias @eldiarioes No con Iberdrola...
    2021-05-04 23:15:43 RT @lmransanz: @MarianoFuentesS Tras el fin de la cesión que nunca debió ser, @iberdrola aún okupa como parking la parcela dotacional públi…
    2021-05-04 23:10:43 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 23:08:42 RT @IracundoIsidoro: Hola Iberdrola
    2021-05-04 22:59:03 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 22:58:08 RT @ferugby: #FERugby |🚨Tenemos buenas noticias: el rugby femenino crece con el nuevo Campeonato de España M14 y M16 ‼️#MujeresEnRugby #Des…
    2021-05-04 22:47:05 RT @AMikofsky: I HATE these obsolete telegraph poles to carry local electricity. Green Energy Iberdrola king J.I. Sanchez Galan was shocked…
    2021-05-04 22:46:52 Estoy absolutamente consternado porque que será ahora del chaval lechuguero que le pedía un puesto en Iberdrola a Pablo inglesias ☹️
    2021-05-04 22:43:23 RT @NiHostiax: Qué es madrid?? Yo solo conozco lucía gonzález manos libres liga iberdrola 2021
    2021-05-04 22:42:36 I HATE these obsolete telegraph poles to carry local electricity. Green Energy Iberdrola king J.I. Sanchez Galan wa… https://t.co/jUeRhlspwe
    2021-05-04 22:42:04 RT @IracundoIsidoro: Hola Iberdrola https://t.co/Zb0WdD3SZk
    2021-05-04 22:41:11 ni los sacos de yeso para marcar los campos, ni el albero, por que lo debían, junto a las deudas del agua de la man… https://t.co/P5PyhcnklC
    2021-05-04 22:39:53 @Rodpac @USTradeRep @AmbassadorTai Dale un vistazo 
    
    https://t.co/2OEeMWaVXV
    2021-05-04 22:36:06 @TxarlyCat @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Y a mucha honra
    2021-05-04 22:32:50 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 22:29:44 A Bet 20 Years Ago Made It the Exxon of Green Power https://t.co/QA0kZfmljH
    2021-05-04 22:28:32 RT @vaquilla13: @iberdrola Que una energética esté representada por un soldado de un sistema dictatorial como el "imperio" de star wars, es…
    2021-05-04 22:28:29 RT @IracundoIsidoro: Hola Iberdrola
    2021-05-04 22:27:49 RT @vaquilla13: @iberdrola Que una energética esté representada por un soldado de un sistema dictatorial como el "imperio" de star wars, es…
    2021-05-04 22:24:59 RT @AytoValledeMena: Nuevos puntos de recarga para vehículos eléctricos. 
    
    El #ValledeMena amplía con cuatro puntos más las posibilidades d…
    2021-05-04 22:23:41 @TraderSpain @martinvars Unicornios que generan electricidad verde? 
    Ahorq mismo EDP renovables o Iberdrola son las… https://t.co/2qV5kdmasc
    2021-05-04 22:19:56 RT @BadmintonESP: ‼ Publicada la convocatoria de la selección 🇪🇸 para el @iberdrola  Spanish Junior International 2021.
    
    ▶ Del 11 al 13 de…
    2021-05-04 22:16:36 @InesArrimadas @BalEdmundo Inesita, on pares guapa? De Jerez a Catalunya i de Catalunya a Mandril…com a bona soldat… https://t.co/fjJZTwXlLc
    2021-05-04 22:12:40 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 22:11:36 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 22:10:51 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 22:10:21 RT @ferugby: #FERugby |🚨Tenemos buenas noticias: el rugby femenino crece con el nuevo Campeonato de España M14 y M16 ‼️#MujeresEnRugby #Des…
    2021-05-04 22:01:26 RT @adrianomones: Mañana tendrá lugar, de 11:00 a 13:00 horas, un webinario organizado por @AEDIVE para conocer en detalle el plan #MOVES3…
    2021-05-04 22:00:41 RT @SantaBadajoz: JORNADA 29 | PRIMERA IBERDROLA 🔴⚪️
    
    🆚 @VCF_Femenino
    🗓 09/05
    🕛 12:00
    📍 IDM El Vivero | Aforo limitado 50%
    
    𝕄𝕒𝕤𝕥𝕖𝕣ℂ𝕙𝕖: la r…
    2021-05-04 21:57:55 @cruztena1 Bajarse cuando considera que no es útil convierte en verdad todo lo que dijo...
    
    Me parece que hay que s… https://t.co/7GwSXavUGO
    2021-05-04 21:55:33 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:55:24 RT @BadmintonESP: ‼ Publicada la convocatoria de la selección 🇪🇸 para el @iberdrola  Spanish Junior International 2021.
    
    ▶ Del 11 al 13 de…
    2021-05-04 21:55:02 RT @smolny7: 6 ERTE, 1 fraude / Hostelería: 17,5% trabajadores menos / La banca roba, los pensionistas protestan / Colas del hambre: Aluche…
    2021-05-04 21:54:20 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:54:11 @_robertomendez_ @claudiacorchon Y lo mejor que le ha pasado a España si señor bye bye Pablito. Próximamente en el… https://t.co/SCD7cr3Jzv
    2021-05-04 21:53:34 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:52:51 RT @____oscar_: ENDESA
    BBVA
    SANTANDER
    IBERDROLA
    REPSOL
    laCaixa
    
    El Conseller Delegat de quina d’aquestes empreses està oferint feina a l’Ar…
    2021-05-04 21:52:49 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 21:51:25 @UNDEFINED1488 @Meritocratico87 @el_pais un tìo que tiene una hipoteca a 30 años y que después de dejar la política… https://t.co/bpqkMYfT1Q
    2021-05-04 21:50:41 Por cierto: si PI Turrión acaba en la TV no sería puerta giratoria? ¿O solo lo es cuando acaban en Red Eléctrica/Iberdrola/Indra?
    2021-05-04 21:48:56 ENDESA
    BBVA
    SANTANDER
    IBERDROLA
    REPSOL
    laCaixa
    
    El Conseller Delegat de quina d’aquestes empreses està oferint feina a l’Arrimadas?
    2021-05-04 21:47:35 RT @IracundoIsidoro: Hola Iberdrola
    2021-05-04 21:46:28 Viendo que Endesa e Iberdrola ya están ocupadas, que compañía eléctrica acogerá al ex viceloquesea?
    2021-05-04 21:45:45 RT @MBenzEspana: Controla el estado de la carga del #NuevoEQA, 100% eléctrico, ¡desde tu Smartphone! Ahora, con el Paquete Plug&amp;Go: disfrut…
    2021-05-04 21:45:25 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:45:11 RT @iberdrola: Tecnología 🔝 para revisar el parque eólico marino de East Anglia One ➡️ El dron salmantino @Aracnocoptero inspeccionará una…
    2021-05-04 21:45:05 @buttercri @donarfonzo Os deja el amado líder...Endesa o Iberdrola????
    2021-05-04 21:44:29 @matthewbennett Llega un poco tarde. Los verdes cosa de alemanes del siglo pasado o se va a colocar en Iberdrola, que también puede ser.
    2021-05-04 21:41:22 RT @IracundoIsidoro: Hola Iberdrola
    2021-05-04 21:40:00 La #cubana Lorena Téllez destaca hoy con 21 goles entre las máximas anotadoras de la Liga Guerreras Iberdrola en el… https://t.co/BzGjCLUOz0
    2021-05-04 21:39:01 @aburrido_de @gerardotc Endesa , Iberdrola  ....?? Informanos 🤣🤣🤣
    2021-05-04 21:38:35 @bulgabesas Iberdrola también es verde.
    2021-05-04 21:37:48 @Arturopdelucia @Jon52189524 @APPA_Renovables @Gesternova @iberdrola @NexusEnergia @AEDIVE @jgonzalezcortes… https://t.co/4MIkeqJLBu
    2021-05-04 21:37:03 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:36:54 @ismaesmtnz_ A trabajar a iberdrola😈
    2021-05-04 21:36:02 RT @IracundoIsidoro: Hola Iberdrola
    2021-05-04 21:35:32 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:34:58 @pabloimro @iescolar En que lado, en las puertas giratorias de Endesa e Iberdrola? Además de esclavos sois unos pedazos de mierda
    2021-05-04 21:34:56 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:34:24 @iberdrola ficha a Pablo Iglesias en su Consejo de Administración.
    
    @elmundotoday
    2021-05-04 21:33:08 Hola Iberdrola https://t.co/Zb0WdD3SZk
    2021-05-04 21:31:27 RT @SmartGreenPeopl: 🔥 SORTEO⁣
    ¡Welcome, mayo! Comienza la Operación bikini💪🏻. Controla todos tus movimientos con este reloj solar y ¡recar…
    2021-05-04 21:28:03 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Toma dosis de nacionalismo casposo, solo falta el toro y… https://t.co/3CpD9g4PPy
    2021-05-04 21:26:53 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:25:16 @Arturopdelucia @Jon52189524 @Gesternova @iberdrola @NexusEnergia @AEDIVE @jmgmoya @jgonzalezcortes @jmg_velez… https://t.co/LYMt36thz5
    2021-05-04 21:23:40 RT @PowerEngInt: Renault has signed a long-term #renewables PPA with utility @iberdrola for the provision of clean energy for its Portugues…
    2021-05-04 21:23:13 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 21:21:49 RT @LilaMarija: @Renewables4ever @iberdrola What about that ground beneath? The grass won't grow!
    2021-05-04 21:21:16 RT @luiztitofbastos: @Renewables4ever @iberdrola A energia Eólica e Solar são o futuro das energias renováveis.
    2021-05-04 21:20:57 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:18:45 @sandrasankarate @JesusDelMoralK1 @Metaphase07 @Finetwork @daedo @UCAM @Pelayo_Seguros @KiaEspana @adidas_ES… https://t.co/wIJVb6wUxl
    2021-05-04 21:18:36 RT @Arturopdelucia: @Jon52189524 @APPA_Renovables @Gesternova @iberdrola @NexusEnergia @AEDIVE @jmgmoya @jgonzalezcortes @jmg_velez @Contig…
    2021-05-04 21:17:43 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:16:57 Subieron el nivel al golpe blando, las patrocinan la ONG de Iberdrola y Claudio X González. Son perversos, no debem… https://t.co/vz8Gmn1Sm9
    2021-05-04 21:16:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:16:07 RT @Quino78600795: @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Muchas felicidades
    2021-05-04 21:15:23 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Muchas felicidades
    2021-05-04 21:14:08 @Jon52189524 @APPA_Renovables @Gesternova @iberdrola @NexusEnergia @AEDIVE @jmgmoya @jgonzalezcortes @jmg_velez… https://t.co/uFuStL0RhY
    2021-05-04 21:14:04 RT @seattletimes: Iberdrola looks well positioned to take advantage of what is likely to be a clean-energy boom in the coming years as the…
    2021-05-04 21:12:41 RT @f4f_barcelona: #MayThe4thBeWithYou i amb totes les persones que fan i donen suport als moviments climàtics. Ni una complicitat amb els…
    2021-05-04 21:10:39 RT @zendalibros: Vuelve la #poesía a #Instagram con Zenda e @iberdrola Tenemos nuevo concurso: #versosprimaverales  
    
    🗓️ Desde el lunes 3 h…
    2021-05-04 21:08:06 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 21:07:05 @EstebanAbad11 @NortenaCatrina Por eso te digo que debes "ilustrarte" un poco. Lee algo, busca información, no le c… https://t.co/jDvrL0YEWe
    2021-05-04 21:06:39 @iberdrola Si que nos acompañe siempre que tengamos el dineral que pedís. Después nos vamos con los Ewoks a vivir e… https://t.co/6yTaVlgwNg
    2021-05-04 21:05:41 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola https://t.co/tFrU485PeF
    2021-05-04 21:05:15 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:05:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:03:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 21:01:41 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:58:30 RT @sanjosecolegio: Roberto Cózar, de 4ª de E.S.O, seleccionado entre estudiantes de toda España en el Congreso Nacional Iberdrola INNOVA I…
    2021-05-04 20:57:58 Sakoneta arranca con un cuarto puesto la Liga Iberdrola https://t.co/IHi29Xpyfu vía @elcorreo_com
    2021-05-04 20:54:43 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:53:27 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:45:14 @vaquilla13 @iberdrola Esto es un Zasca brutal.
    2021-05-04 20:42:42 RT @iberdrolamex: De acuerdo con el estudio de percepción #ImpulsoSTEM, el 90% de la juventud oaxaqueña desea continuar con sus estudios a…
    2021-05-04 20:41:25 RT @cordobadeporte: Además de este éxito junto al Club Rítmico Colombino, la gimnasta del @grseneca celebra el reconocimiento como deportis…
    2021-05-04 20:41:17 RT @cordopolis_es: BASE | Noa Ámber, octava en la Liga Nacional Iberdrola. La gimnasta del @grseneca cuaja una gran actuación en la competi…
    2021-05-04 20:41:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:40:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:40:05 @GirautaOficial Cuando entre de consejero en iberdrola.
    2021-05-04 20:36:50 RT @NoticiasdeAlava: El @cluboskitxo logra el bronce en la primera fase de la Liga Iberdrola https://t.co/GIXq3zm9Yf
    2021-05-04 20:36:19 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:36:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:36:10 RT @f4f_barcelona: #MayThe4thBeWithYou i amb totes les persones que fan i donen suport als moviments climàtics. Ni una complicitat amb els…
    2021-05-04 20:34:47 Qué es madrid?? Yo solo conozco lucía gonzález manos libres liga iberdrola 2021
    2021-05-04 20:32:39 @ferugby @PGR_RUGBY @AmaiaErbina @mlosgif @annefdecorres @deportegob @COE_es @csed_csd @iberdrola @Tour_UM Nosotras… https://t.co/eglizCOuWm
    2021-05-04 20:31:44 RT @EllasSonFutbol: 📝⚽⚔️  R E S U L T A D O S
    
    #RetoIberdrola Gr. D
    Jornada 7️⃣
    🔲 Grupo Norte ⬇️
    https://t.co/50O60lHSlt
    
    🔳 Grupo Sur ⬇️
    ht…
    2021-05-04 20:31:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:30:40 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:29:32 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:26:32 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Máquinas!!!
    2021-05-04 20:26:22 RT @f4f_barcelona: #MayThe4thBeWithYou i amb totes les persones que fan i donen suport als moviments climàtics. Ni una complicitat amb els…
    2021-05-04 20:26:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:25:42 RT @LaurentMauduit: Un navire de la gendarmerie maritime arrive au large de #SaintQuayPortrieux. Pour surveiller les pêcheurs justement en…
    2021-05-04 20:24:54 RT @joseviruete: UF https://t.co/WX8lFkyPQe
    2021-05-04 20:24:23 El Club Ximnasia Porriño clasifica a dos de sus gimnastas para la Copa de la Reina individual Iberdrola… https://t.co/bnFF4sICCL
    2021-05-04 20:20:26 RT @f4f_barcelona: #MayThe4thBeWithYou i amb totes les persones que fan i donen suport als moviments climàtics. Ni una complicitat amb els…
    2021-05-04 20:17:20 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 20:16:48 @aytoabadiano Aupa, en una parte del barrio de Muntzaraz nos hemos quedado sin luz por la noche desde ayer. Es el a… https://t.co/iwCS9B860b
    2021-05-04 20:15:35 RT @joseviruete: UF
    2021-05-04 20:14:41 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:14:40 UF https://t.co/WX8lFkyPQe
    2021-05-04 20:12:58 RT @rfe_hockey: ¡¡La Final Four @iberdrola también tuvo a sus destacadas individualmente!!:
    - MVP: @begogrc ( @CCVMDHF )
    - Mejor Portera: J…
    2021-05-04 20:12:31 RT @f4f_barcelona: #MayThe4thBeWithYou i amb totes les persones que fan i donen suport als moviments climàtics. Ni una complicitat amb els…
    2021-05-04 20:12:13 #MayThe4thBeWithYou i amb totes les persones que fan i donen suport als moviments climàtics. Ni una complicitat amb… https://t.co/Ezyt773NE9
    2021-05-04 20:11:49 @science_dirk @ThomasRotthier @RubenBaetens In de EU hebben Orsted en Iberdrola als grootste elk +/- 3% marktaandee… https://t.co/q3AslV1NB8
    2021-05-04 20:11:29 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 20:10:33 NoticiasDeportivas El Colonial Sport se proclama vencedor de la primera Fase Liga Clubes Iberdrola de rítmica en Va… https://t.co/Z2jgwVPKt0
    2021-05-04 20:09:01 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 20:07:49 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:07:02 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola EllAs, ellOs y ellEs
    2021-05-04 20:06:50 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 20:01:18 RT @ecoticiasRED: Renault e Iberdrola una ‘acuerdo’ para alcanzar la huella de carbono cero | https://t.co/0DZNpKm3OV https://t.co/W4OHSUGz…
    2021-05-04 20:00:55 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:59:12 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:58:23 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:55:57 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 19:53:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:51:51 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:49:49 A Bet 20 Years Ago Made It the Exxon of Green Power https://t.co/K02Ce6xwyZ
    2021-05-04 19:49:02 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:47:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:45:05 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-04 19:42:10 https://t.co/szhQXgvKy0
    2021-05-04 19:41:46 RT @rkyte365: Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an important driv…
    2021-05-04 19:39:04 RT @EMUASA_Clientes: Nuestra apuesta por la energía renovable, limpia y respetuosa con el medio ambiente se ve legitimada por este certific…
    2021-05-04 19:35:31 Siguiendo el magnífico tutorial para editar #fineart de 
    @Ivanf_foto
    ... la torre #Iberdrola de #bilbao vistas de o… https://t.co/7Y4eAZgbJU
    2021-05-04 19:32:04 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:30:10 Renault has signed a long-term #renewables PPA with utility @iberdrola for the provision of clean energy for its Po… https://t.co/Mwm4znkxcn
    2021-05-04 19:29:36 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:27:53 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:25:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:25:11 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:22:04 RT @iberdrola: 💧 Así avanza TÂMEGA, el complejo hidroeléctrico más INNOVADOR de Europa:
    
    🔋 Almacenamiento de energía por bombeo
    🇵🇹 Energía…
    2021-05-04 19:18:51 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:18:41 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:18:39 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 19:16:00 💧 Así avanza TÂMEGA, el complejo hidroeléctrico más INNOVADOR de Europa:
    
    🔋 Almacenamiento de energía por bombeo
    🇵🇹… https://t.co/mViwik1wg0
    2021-05-04 19:15:47 Novena jornada de la Segunda fase de la Liga Iberdrola de Rugby. https://t.co/D7LhhDWiHn
    2021-05-04 19:15:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:15:34 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:14:38 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Enhorabuena
    2021-05-04 19:13:22 RT @epigmenioibarra: Yo me pronuncio a favor de que @FGRMexico y @SNietoCastillo abran la caja de Pandora dla corrupción en el sexenio de @…
    2021-05-04 19:12:29 RT @SmartJoules: Look forward to seeing many more #climatedeals like this where companies march fast toward #zerocarbon across all their fa…
    2021-05-04 19:08:02 Dr. Anita Sengupta (@doctor_astro)--an engineering professor, experienced pilot and CEO/founder of electric aviatio… https://t.co/epphiloYQJ
    2021-05-04 19:07:57 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 19:03:33 Great to see this story of ⁦@iberdrola⁩ via @NYTimes - and the foresight of their CEO Ignacio Galan. He was an impo… https://t.co/JbP1C9hX66
    2021-05-04 19:02:11 El capitalismo atenta contra todos los derechos constitucionales. Para carta magna las facturas hinchadas de intere… https://t.co/1SiJabth7I
    2021-05-04 19:01:49 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 19:01:40 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:58:04 @rivasactual Rítmica Elegance consigue el noveno puesto en la 1ª fase de la Liga Iberdrola @EleganceRitmica #Rivas… https://t.co/NIroa9v0qD
    2021-05-04 18:57:24 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:54:23 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:51:37 RT @EnergySysCat: 🗞️ @Iberdrola_En is launching a Challenge call...
    
    ..inviting #innovators developing products &amp; services such as Anti-isl…
    2021-05-04 18:50:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:49:34 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:49:16 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:48:19 RT @alvaro_j_campos: Por si había alguna duda de que eran el lado oscuro
    2021-05-04 18:48:17 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 18:47:15 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:45:55 @FelixMorenoIbi @AAretxaba @iberdrola @Endesa 
    te vendo luz o mejillones
    
    (estoy en tratos, no me jo... el negocio… https://t.co/UW0YZHrQ5B
    2021-05-04 18:45:51 RT @SmartGreenPeopl: 🔥 SORTEO⁣
    ¡Welcome, mayo! Comienza la Operación bikini💪🏻. Controla todos tus movimientos con este reloj solar y ¡recar…
    2021-05-04 18:45:00 El @cluboskitxo logra el bronce en la primera fase de la Liga Iberdrola https://t.co/GIXq3zm9Yf
    2021-05-04 18:42:48 RT @FutbolisticasES: La madridista Olga Carmona, MVP de la Jornada 28 en Futbolísticas | Informa: Sara Portillo.
    Futbolísticas - El periódi…
    2021-05-04 18:42:46 @Renewables4ever @iberdrola exxon mobile only cares about one green and that's money https://t.co/Q461khwY35
    2021-05-04 18:42:17 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 18:38:25 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 18:38:01 @fns_k @iberdrola 🤣🤣🤣
    2021-05-04 18:37:19 Three of Europe’s multinational renewable energy generators recently announced #greenhydrogen  production plans for… https://t.co/iDxXWZiPMM
    2021-05-04 18:36:54 $IBE Un pequeño adelanto👇 Long!
    
    Avangrid (Iberdrola) eleva un 39% su beneficio trimestral y mejora sus perspectiva… https://t.co/j2c0uqkYcj
    2021-05-04 18:35:31 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 18:29:51 RT @christinayiotis: #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture Suz…
    2021-05-04 18:28:53 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:28:15 @phatima68 Hola. Disculpa, ¿pero es el teléfono que tienes registrado como cliente? Si nos das la Referencia o NIF… https://t.co/NfZKELA2OY
    2021-05-04 18:27:38 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:27:10 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 18:26:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:25:12 @Felipe_Vlcia @iberdrola https://t.co/EUlLmsauZ4
    2021-05-04 18:24:51 @Felipe_Vlcia @iberdrola https://t.co/ObJvJ1tcPG
    2021-05-04 18:24:28 Look forward to seeing many more #climatedeals like this where companies march fast toward #zerocarbon across all t… https://t.co/0xiRnS3Uul
    2021-05-04 18:23:14 @fns_k @iberdrola La timotarofa, obviously!!
    
    Oh....wait...no. 🤣
    2021-05-04 18:22:43 @iberdrola Ibertrola
    2021-05-04 18:22:08 RT @alvaro_j_campos: Por si había alguna duda de que eran el lado oscuro
    2021-05-04 18:21:58 RT @pierrevabres: «@AilesMarines SAS, dont l’actionnaire principal est la multinationale espagnole de l’énergie @iberdrola, remporte la mis…
    2021-05-04 18:21:02 RT @fotosport_es: 📷GALERÍA | Ya tienes en la web las imágenes del @PozuFem - @CE_Seagull de ayer 👉 https://t.co/SZQZN0Zu7Q https://t.co/roc…
    2021-05-04 18:20:38 @Felipe_Vlcia @iberdrola ¿cuál? ¿La PVPC o la super-tarifa-timo de las comercializadoras?
    2021-05-04 18:19:00 @fns_k @iberdrola Yo quiero, pero no se dnd se contrata
    2021-05-04 18:18:58 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 18:17:33 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:17:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:17:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:16:05 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 18:15:42 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola 🥰👏👏👏👏👏👏👏👏👏
    2021-05-04 18:14:59 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:13:56 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:13:55 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola 🏅🏅🏅🏅🏅👍👍👍👍👍👍
    2021-05-04 18:11:30 Encuentro digital con Joan Cabrero: AMD, Alibaba, IBEX, Total, Microsoft, Ence, Inditex, Solaria, Gamesa, Dow Jones… https://t.co/NeIPkbWurc
    2021-05-04 18:09:54 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 18:07:21 Me encanta cuando me llaman de @iberdrola (esta última, pero antes otras...) para ofrecerme una tarifa para ahorrar… https://t.co/W8jtBJjraS
    2021-05-04 18:06:09 RT @villararzobispo: Iberdrola comunica al vecindario que mañana miércoles de 08.30 a 12.30 hrs, se cortará el suministro eléctrico en Inge…
    2021-05-04 18:05:52 @Tour_UM @CCVMDHF @ccvmoficial @rfe_hockey @iberdrola Gracias!!
    2021-05-04 18:05:38 @crislt3 @CCVMDHF @ccvmoficial @rfe_hockey @iberdrola 😘😘😘
    2021-05-04 18:05:31 @1nachin @CCVMDHF @ccvmoficial @rfe_hockey @iberdrola 🥳🥳🥳
    2021-05-04 18:05:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:03:36 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:02:33 Iberdrola comunica al vecindario que mañana miércoles de 08.30 a 12.30 hrs, se cortará el suministro eléctrico en I… https://t.co/umDD7XjqBE
    2021-05-04 18:01:50 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 18:01:46 A Bet 20 Years Ago Made It the Exxon of Green Power - The New York Times https://t.co/PaIfs3Wjah
    2021-05-04 17:59:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:58:57 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 17:58:11 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:56:57 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 17:56:45 RT @SmartGreenPeopl: 🔥 SORTEO⁣
    ¡Welcome, mayo! Comienza la Operación bikini💪🏻. Controla todos tus movimientos con este reloj solar y ¡recar…
    2021-05-04 17:56:41 RT @qtf: La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto con el Museo N…
    2021-05-04 17:55:20 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:55:15 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:55:15 La restauradora mexicana Silvia Ixchel García Valencia obtuvo la Beca Internacional Fundación Iberdrola en conjunto… https://t.co/WUxvKlKW9N
    2021-05-04 17:54:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:53:51 @EvielKhon @cosmicJunkBot Porque a Anaya lo estaba incubando e impulsando lo más corrupto del PAN, PRI, PRD y las t… https://t.co/51NXGiotE5
    2021-05-04 17:52:49 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:52:12 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:49:37 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 17:46:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:45:28 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:45:00 .@iberdrola is the Spanish energy brand in the sector’s top 10, according to intangible valuation consultancy… https://t.co/g4zM4iJoWB
    2021-05-04 17:44:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:42:00 RT @Fund_Diversidad: 👉@iberdrola participando en el evento de lanzamiento del #MesEuropeoDeLaDiversidad de la @ComisionEuropea y @EU_Justic…
    2021-05-04 17:41:49 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:41:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:41:17 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:39:31 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:38:48 RT @JavierH48698750: Este es el susodicho 
    José Ramón Cossío 
    Quien defiende a los X González 
    A Iberdrola y demás compinches ladrones de l…
    2021-05-04 17:36:33 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:36:05 @ellasvalenoro @atletismoRFEA @MujeryAtletismo @iberdrola Bravas!!!!!!!! 👏👏👏👏👏
    2021-05-04 17:35:12 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:34:59 RT @Tour_UM: 🎥 Después de su lesión de rodilla entrevistamos a @CarolinaMarin y lo tenía muy claro, quería VOLVER A CONQUISTAR EL TÍTULO EU…
    2021-05-04 17:34:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:33:34 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:28:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:25:55 RT @zendalibros: Vuelve la #poesía a #Instagram con Zenda e @iberdrola Tenemos nuevo concurso: #versosprimaverales  
    
    🗓️ Desde el lunes 3 h…
    2021-05-04 17:24:40 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:24:37 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:23:19 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Pri… https://t.co/Y28OWiFlTt
    2021-05-04 17:22:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:21:58 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:21:48 @macs_df No sé... lo de los moches y su operación cuando fue Presidente del Congreso demostró que iba pa'peón de Ma… https://t.co/JqJn6eXtVF
    2021-05-04 17:21:11 Por si había alguna duda de que eran el lado oscuro https://t.co/5lif7WnmcI
    2021-05-04 17:18:14 RT @idiomafutve: Gabriela García marcó y dio una asistencia para guiar al Deportivo al triunfo sobre el Madrid 3-0, en la Primera Iberdrola…
    2021-05-04 17:16:14 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 17:16:00 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:15:38 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:15:11 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:13:00 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 17:12:58 RT @Existimos_: En Segovia @iberdrola construirá la 2ª planta solar más grande de Europa https://t.co/Ah4FlCB7Mv
    
    Es necesario que las empr…
    2021-05-04 17:10:08 @luismiGmz @z7taldea @REALJOSEIDAD Sabes que este cariño a empezado este año y no se el porque, bueno si que lo se,… https://t.co/u08llyPHxv
    2021-05-04 17:09:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:08:49 RT @rfe_hockey: ¡¡La Final Four @iberdrola también tuvo a sus destacadas individualmente!!:
    - MVP: @begogrc ( @CCVMDHF )
    - Mejor Portera: J…
    2021-05-04 17:07:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 17:01:04 La gimnasta del Club Gimnástico Fuenlabrada, Irene García, compitió el pasado 1 de mayo en la 1ª fase de la Liga Ib… https://t.co/LddDIZvlW6
    2021-05-04 17:00:28 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 16:59:00 🤾‍♀️El @BalonmanoAdesal cayó ante el Zuazo (23-30) y se complica sus opciones de permanencia en la Liga Guerreras I… https://t.co/33bB9aRRAI
    2021-05-04 16:57:31 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:55:53 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:55:45 RT @christinayiotis: #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture Suz…
    2021-05-04 16:51:12 @Renewables4ever @iberdrola What about that ground beneath? The grass won't grow!
    2021-05-04 16:51:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:50:31 A Bet 20 Years Ago Made It the Exxon of Green Power https://t.co/P9WWLSNbip
    2021-05-04 16:50:20 📢📢Rubén García y María de la O Pérez, convocados con la Selección Española
    https://t.co/i7A8DvRB65
    🏆Iberdrola Spani… https://t.co/3F2av56mjM
    2021-05-04 16:50:06 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:46:55 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:41:10 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:37:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:37:46 RT @zendalibros: Vuelve la #poesía a #Instagram con Zenda e @iberdrola Tenemos nuevo concurso: #versosprimaverales  
    
    🗓️ Desde el lunes 3 h…
    2021-05-04 16:36:38 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:36:08 @RFEVB @VoleyFemAbs @teledeporte @PalomadelrioTVE @c_corcelles @deportegob @COE_es @iberdrola @Luanvi… https://t.co/MnoRaESwih
    2021-05-04 16:35:27 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:34:36 @Renewables4ever @iberdrola Best wishes for business growth
    2021-05-04 16:34:09 @Renewables4ever @iberdrola Best wishes
    2021-05-04 16:33:24 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:33:16 RT @nytimesworld: Iberdrola is one of a handful of utilities seen as leading a new generation of “renewable majors,” comparable to the way…
    2021-05-04 16:32:59 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:32:57 RT @nytclimate: Iberdrola is one of a handful of utilities seen as leading a new generation of “renewable majors,” comparable to the way oi…
    2021-05-04 16:31:28 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:31:16 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:30:35 @Viri_Rios 👁️  ESTA CHAYOTERA ESCRIBE PARA EL PAÍS, NO PUES WOW, ESO LO EXPLICA TODO, EL PAÍS OTRO PASQUÍN INMUNDO… https://t.co/auWMfqIMvB
    2021-05-04 16:29:56 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:29:39 RT @christinayiotis: #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture Suz…
    2021-05-04 16:29:35 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:29:29 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:29:00 Vuelve la #poesía a #Instagram con Zenda e @iberdrola Tenemos nuevo concurso: #versosprimaverales  
    
    🗓️ Desde el lu… https://t.co/b188Y8xAVg
    2021-05-04 16:28:20 RT @lmransanz: @MarianoFuentesS Tras el fin de la cesión que nunca debió ser, @iberdrola aún okupa como parking la parcela dotacional públi…
    2021-05-04 16:28:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:27:16 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:27:14 RT @Gremlin_moja0: Asquerosos
    2021-05-04 16:24:25 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:23:43 @TodoJingles Joder que pesados los de Iberdrola
    2021-05-04 16:23:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:22:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:22:13 RT @vaquilla13: @iberdrola Que una energética esté representada por un soldado de un sistema dictatorial como el "imperio" de star wars, es…
    2021-05-04 16:22:10 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:21:46 Ejemplo del buen reciclaje
    Mira lo que te propone...
    https://t.co/lDwt68cLUy https://t.co/Ptwj3BZCMv
    2021-05-04 16:21:11 RT @christinayiotis: #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture Suz…
    2021-05-04 16:20:31 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 16:20:06 BASE | Noa Ámber, octava en la Liga Nacional Iberdrola. La gimnasta del @grseneca cuaja una gran actuación en la co… https://t.co/CRC0HN6gl9
    2021-05-04 16:19:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:18:52 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:18:47 @Malbem4 @V_TrujilloM Y nada, suma el subsidio eléctrico que se da a Iberdrola, el impago de las mineras, el dinero… https://t.co/3j8CS2Q0qT
    2021-05-04 16:16:36 @aegonseguros | @eoi | @ImqEuskadi | @iDE_Iberdrola | @la_Mutua | @Solimat72 | @BainAlerts | @igslasalle |… https://t.co/iZNdJsZQBJ
    2021-05-04 16:15:38 Estáis muy pesaditos, @TuIberdrola @iberdrola … Insistentes hasta decir basta. ¿Podéis dejar de llamar a todas hora… https://t.co/Ct2Mfh1m9N
    2021-05-04 16:13:43 @nachoblz @iberdrola @consumogob @consumidores @CNMC_ES Me acaban de llamar de este teléfono, pidiendo datos person… https://t.co/trPkckTUTb
    2021-05-04 16:13:04 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:12:17 RT @WorldEnergyTrad: Tres grandes del sector energético estudian el desarrollo del mayor proyecto de hidrógeno verde en la Comunidad Valenc…
    2021-05-04 16:10:45 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:10:26 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:08:35 qualcun ha luce e gas con la compagnia Iberdrola?
    sono affidabili?
    ho letto recensioni contrastanti :(
    2021-05-04 16:08:16 RT @epigmenioibarra: Yo me pronuncio a favor de que @FGRMexico y @SNietoCastillo abran la caja de Pandora dla corrupción en el sexenio de @…
    2021-05-04 16:06:57 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:06:20 RT @Tour_UM: 🎥 Después de su lesión de rodilla entrevistamos a @CarolinaMarin y lo tenía muy claro, quería VOLVER A CONQUISTAR EL TÍTULO EU…
    2021-05-04 16:05:59 Y’a un commercial d’iberdrola qui est venu nous vendre un contrat l’autre jour, bref on signe car il nous assure qu… https://t.co/w3yuaIkbOw
    2021-05-04 16:05:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:04:31 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:03:47 RT @BarcaFem: 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Primera Iberd…
    2021-05-04 16:03:32 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 16:03:27 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:03:01 https://t.co/I5SidScnVL via @iranandworld
    2021-05-04 16:02:51 📊 Barça Femení under Lluís Cortés: 
    
    90 Matches
    81 Wins
    3 Draws
    6 Losses 
    341 Goals Scored
    40 Goals Against  
    
    🏆Pri… https://t.co/VKjb4OlRSv
    2021-05-04 16:02:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:01:45 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:01:00 ¿Quieres ahorrar en tu factura de la luz? 👛 Pues con estos consejos, ¡podrás hacerlo sin problema! Lee, lee… 📚… https://t.co/X0oLqVKGYe
    2021-05-04 16:00:14 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 16:00:00 Tres grandes del sector energético estudian el desarrollo del mayor proyecto de hidrógeno verde en la Comunidad Val… https://t.co/j6uFVDuzin
    2021-05-04 15:59:45 RT @AytoValledeMena: Nuevos puntos de recarga para vehículos eléctricos. 
    
    El #ValledeMena amplía con cuatro puntos más las posibilidades d…
    2021-05-04 15:59:20 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:59:03 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:59:00 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:58:44 @TuIberdrola Me da igual quién gestiona qué. No me faltaba más que saber cómo funciona internamente @iberdrola. Sol… https://t.co/ChEyxZ7GhE
    2021-05-04 15:58:34 RT @No8Do: @iberdrola Eehhh... Los Stormtroopers eran los malos... No sé si lo sabíais 🙄
    2021-05-04 15:58:34 Scegli l'energia 100% rinnovabile di Iberdrola! 👇
    🌎  Se sottoscrivi l'offerta EcoTua Web Luce riceverai uno sconto… https://t.co/y1Holw2AKk
    2021-05-04 15:57:15 Do you know Iberdrola, the world leader in combined wind and solar power outside of China?
    (NYT) https://t.co/174iVD10h3
    2021-05-04 15:56:27 RT @Fund_Diversidad: 👉@iberdrola participando en el evento de lanzamiento del #MesEuropeoDeLaDiversidad de la @ComisionEuropea y @EU_Justic…
    2021-05-04 15:55:39 @InversorExperto @JRHernan87 Bancos a 6 veces beneficios, ACS, Mapfre Repsol 10 veces, Inditex a 23 (quizas de las… https://t.co/jSDzRB2xEK
    2021-05-04 15:55:02 @iberdrola Os da cosita no vender la energía nuclear como limpia, verdad? 😍
    2021-05-04 15:55:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:54:32 @HermanosConsola @iberdrola Eso díselo al estado, bro
    2021-05-04 15:54:11 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:53:36 RT @aytocabanillas: AVISO: CORTE DE SUMINISTRO ELÉCTRICO EN CALLE CASPUEÑAS
    La suministradora @iberdrola ha comunicado al Ayuntamiento que…
    2021-05-04 15:52:17 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:51:32 And yes, I'm ridiculously proud of myself. French peeps, if someone from a comapny named "Iberdrola" asks to check… https://t.co/rH7K2fI3wE
    2021-05-04 15:48:36 @Renewables4ever @iberdrola #ExxontrumputinGlobalClimateDisaster
    2021-05-04 15:46:47 RT @SeedsMentor: Hoy disponible VÍDEO de #ResumenSemanalTrading sobre las 14:30 hs con #análisisBursátil entre otros  de:
    #IBERDROLA
    #GRIFO…
    2021-05-04 15:46:44 RT @koliblish: Sí viviéramos en la ÉPOCA DE CALDERÓN, ya hubiéramos conseguido las VACUNAS a través de OHL, IBERDROLA. Pero sí ellos no la…
    2021-05-04 15:46:18 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:46:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:40:19 @Renewables4ever @iberdrola @RichyGh4
    2021-05-04 15:39:58 @_tracia__ Seguro que irá a Endesa, Iberdrola o alguna otra gran empresa. Las puertas giratorias así funcionan para todos los políticos, no?
    2021-05-04 15:39:18 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 15:38:33 RT @christinayiotis: #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture Suz…
    2021-05-04 15:37:00 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:37:00 Tecnología 🔝 para revisar el parque eólico marino de East Anglia One ➡️ El dron salmantino @Aracnocoptero inspeccio… https://t.co/PHBVFofARa
    2021-05-04 15:34:56 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:34:28 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:34:20 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:33:16 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:31:22 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:30:28 The Green and Sustainable Future of Transport by @Doctor_Astro 🚅🚁✈️🚀
    
    Interesting and realistic overview of a carbo… https://t.co/SpEFTFICH0
    2021-05-04 15:30:02 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 15:28:27 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:25:53 RT @CeliaCC11: Si piensa la oposición, OEA, ONU, PRIAN, COPARMEX, CLAUDIO X GONZÁLEZ, EEUU, IBERDROLA, EXPRESIDENTES, INE, JUECES CORRUPTOS…
    2021-05-04 15:24:57 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:23:53 Les respostes 👌🤣🤣🤣 https://t.co/QEdINuHlA3
    2021-05-04 15:23:15 RT @H2TechOnline: @bp_plc, @iberdrola and @enagas plan to develop green H2 facility in Valencia region
    Read More: https://t.co/KAailIxkSe…
    2021-05-04 15:22:57 RT @vaquilla13: @iberdrola Que una energética esté representada por un soldado de un sistema dictatorial como el "imperio" de star wars, es…
    2021-05-04 15:22:17 @iberdrola bajad la luz hijos de puta
    2021-05-04 15:21:46 RT @EMUASA_Clientes: Nuestra apuesta por la energía renovable, limpia y respetuosa con el medio ambiente se ve legitimada por este certific…
    2021-05-04 15:21:11 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:20:22 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:17:57 RT @EMUASA_Clientes: Nuestra apuesta por la energía renovable, limpia y respetuosa con el medio ambiente se ve legitimada por este certific…
    2021-05-04 15:17:02 Sí viviéramos en la ÉPOCA DE CALDERÓN, ya hubiéramos conseguido las VACUNAS a través de OHL, IBERDROLA. Pero sí ell… https://t.co/0uNtQfn2R7
    2021-05-04 15:16:15 RT @christinayiotis: #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture Suz…
    2021-05-04 15:16:11 vous êtes une société d'énergie espagnole ?" "Non non j'ai jamais dit ça, on est une société française implantée à… https://t.co/2LMpHcoETT
    2021-05-04 15:16:10 avertissant d'une arnaque, mais ça je ne l'ai vu qu'après. La commerciale, puisque c'est de ça qu'il s'agit, dit d'… https://t.co/GeMqgpvYsh
    2021-05-04 15:16:04 J'ai vendu du rêve à ma voisine. On s'est fait démarcher par quelqu'un de la compagnie @iberdrola qui s'est présent… https://t.co/W9uLBNCv0V
    2021-05-04 15:15:58 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:15:12 #HackTheCapitol 4.0 " #ICS #Security in #Europe " @kaspersky @shipulin_anton @iberdrola @A_Valencia_Gil @Accenture… https://t.co/huV8RDQn3m
    2021-05-04 15:14:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:14:30 RT @ReinasDelBalon: Desde @ReinasDelBalon queremos agradecer a @iberdrola su apoyo por la #IgualdadDeGénero y por apostar por la mujer en e…
    2021-05-04 15:14:14 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 15:13:35 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:10:10 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:09:31 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 15:08:42 @Renewables4ever @iberdrola 👍👍👍
    2021-05-04 15:08:33 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:05:09 Iberdrola revisa toda su política sobre corrupción bajo la lupa del caso Villarejo https://t.co/yZUM3xn9ln
    2021-05-04 15:05:04 @bp_plc, @iberdrola and @enagas plan to develop green H2 facility in Valencia region
    Read More:… https://t.co/GncwOi3K3s
    2021-05-04 15:04:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:02:16 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 15:02:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 15:01:07 🔝Su estrategia de inversión en energía limpia y redes llevará a @iberdrola a:
    
    👉Ser una compañía "neutra en carbono… https://t.co/xMpIuaUJBP
    2021-05-04 14:59:44 @numer344 No sabía yo que Iberdrola era social comunista jajajajaja
    2021-05-04 14:57:54 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:57:33 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:57:29 @EDFetMoi @WailPuck Après recherche et questionnement de la personne en question, je pense qu'il s'agit d'un démarc… https://t.co/TLJEBnJONY
    2021-05-04 14:56:23 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:55:52 @RFEVB @VoleyFemAbs @teledeporte @PalomadelrioTVE @c_corcelles @deportegob @COE_es @iberdrola @Luanvi… https://t.co/wGQe3aFRvd
    2021-05-04 14:55:30 RT @RSI: #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS, @iberdrola https://t.…
    2021-05-04 14:54:53 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:53:17 RT @epigmenioibarra: Yo me pronuncio a favor de que @FGRMexico y @SNietoCastillo abran la caja de Pandora dla corrupción en el sexenio de @…
    2021-05-04 14:50:38 RT @Existimos_: En Segovia @iberdrola construirá la 2ª planta solar más grande de Europa https://t.co/Ah4FlCB7Mv
    
    Es necesario que las empr…
    2021-05-04 14:50:03 #HackTheCapitol Panel 2: ICS Security in Europe - @InfosecManBlog @Accenture, @shipulin_anton @KasperskyICS,… https://t.co/YyGZaTeVZo
    2021-05-04 14:49:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:48:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:46:50 @SaraFPerez @emgalego Supoño que sera en todos lado igual pero en Iberdrola nos ensinaran que en casos asi, o que h… https://t.co/5XtaztSFXS
    2021-05-04 14:46:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:45:35 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 14:44:00 Nuestra apuesta por la energía renovable, limpia y respetuosa con el medio ambiente se ve legitimada por este certi… https://t.co/ooCBC1Xdtz
    2021-05-04 14:43:45 AVISO: CORTE DE SUMINISTRO ELÉCTRICO EN CALLE CASPUEÑAS
    La suministradora @iberdrola ha comunicado al Ayuntamiento… https://t.co/KdUU4XmldS
    2021-05-04 14:43:43 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 14:42:50 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 14:40:00 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 14:39:02 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 14:38:44 RT @MemorialVT: Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por ETA-m en…
    2021-05-04 14:37:51 RT @Chiquilukisss: Análisis Semifinales UWCL+ Jornada de Primera Iberdrola https://t.co/CcFkDOR9lT 
    
    Hoy les traemos junto a @Lora099 @1802…
    2021-05-04 14:35:44 @MemorialVT @iberdrola D.E.P. Uno de tantos inocentes que nos robaron.
    Mi apoyo a la familia siempre y alegría de q… https://t.co/iLa8kvfOlr
    2021-05-04 14:35:13 RT @RedElectricaREE: 🗓️ El próximo 18 de mayo nuestra presidenta @BeatrizCorredor participa en el #CongresoCEAPI @CEAPIconsejo 
    
    👉Conversar…
    2021-05-04 14:33:58 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 14:33:11 Recordamos al ingeniero @iberdrola y jefe de proyectos de la central de Lemóniz ÁNGEL PASCUAL MÚGICA asesinado por… https://t.co/XtFDpfsSGN
    2021-05-04 14:32:39 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:31:50 Nuevos puntos de recarga para vehículos eléctricos. 
    
    El #ValledeMena amplía con cuatro puntos más las posibilidade… https://t.co/p789VLkniX
    2021-05-04 14:31:18 RT @derus1977: @futmondo se buscan usuarios interesados en participar para la próxima temporada en la Primera,Segunda,Premier,Serie A,Super…
    2021-05-04 14:30:19 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:27:24 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:26:50 RT @epigmenioibarra: Yo me pronuncio a favor de que @FGRMexico y @SNietoCastillo abran la caja de Pandora dla corrupción en el sexenio de @…
    2021-05-04 14:26:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:25:28 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:24:55 RT @iberdrolamex: De acuerdo con el estudio de percepción #ImpulsoSTEM, el 90% de la juventud oaxaqueña desea continuar con sus estudios a…
    2021-05-04 14:24:43 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:21:13 Innovation @iberdrola using #renewables in American and worldwide #electricutilities is changing the global industr… https://t.co/ltDyC3MZwW
    2021-05-04 14:20:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:19:59 @vayatoalla4 un problema de deslegitimar las instituciones es que se abre paso a monstruos muy peligrosos. tengamos… https://t.co/dmn28hJIbd
    2021-05-04 14:19:27 @Renewables4ever @iberdrola we are supplying medical, surgical, FMCG &amp; Solar Products at the least price to our cus… https://t.co/57KLg0LZ8v
    2021-05-04 14:19:22 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:19:04 @Elias_T3 @EndesaClientes Me cambie a cureenergia de Iberdrola hace dos meses pero tampoco me facturan, porque dond… https://t.co/pDvS3p0vMz
    2021-05-04 14:18:46 RT @FutbolisticasES: La madridista Olga Carmona, MVP de la Jornada 28 en Futbolísticas | Informa: Sara Portillo.
    Futbolísticas - El periódi…
    2021-05-04 14:18:28 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:17:10 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:17:00 🏅 | Siemens e Iberdrola, entre las mejores compañías para trabajar según Linkedin. Más detalles de esta clasificaci… https://t.co/9E3lJYcZt0
    2021-05-04 14:16:32 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:16:22 RT @rivasactual: Rítmica Elegance consigue el noveno puesto en la 1ª fase de la Liga Iberdrola @EleganceRitmica #Rivas #RivasActual #Elegan…
    2021-05-04 14:15:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:15:39 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:15:17 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:15:03 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:13:05 Iberdrola, Renault sign long-term PPA for Spanish and Portuguese facilities - Power Engineering International https://t.co/DQwDSmTGtS
    2021-05-04 14:09:47 RT @rfe_hockey: ¡¡La Final Four @iberdrola también tuvo a sus destacadas individualmente!!:
    - MVP: @begogrc ( @CCVMDHF )
    - Mejor Portera: J…
    2021-05-04 14:09:03 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:07:55 RT @AlbertFabrega: Parada para desayunar. En Cetina. Llegó sobrado de energía 30%, pero tengo hambre. Cargando a 29 kW en cargador de @iber…
    2021-05-04 14:06:45 @Renewables4ever @iberdrola “Exxon of green power?”
    You mean they’re a bunch of theiving cheats?
    2021-05-04 14:06:12 @nolo14 @aileon25 @edukrator Hasta Iberdrola lo sabe
    
    https://t.co/qKvGErGP9c
    2021-05-04 14:04:45 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:04:29 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 14:04:00 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:02:00 RT @asociacionMKT: «En MKT nos preocupan mucho las connotaciones negativas que se asocian al #marketing. Haríamos mal si reconociésemos cas…
    2021-05-04 14:01:54 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:01:25 @iberdrola Son ustedes unos cachondos, mucho ñiñiñí, pero nos cobran su electricidad como si estuvieran iluminado l… https://t.co/Z2dtMChIz1
    2021-05-04 14:01:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:00:35 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 14:00:14 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:59:46 LEVANTE UD vs SEVILLA ⚽ Fútbol Femenino 🏆 1era Iberdrola | 2-1 | FULL MA... https://t.co/JcYkKnRjxB a través de @YouTube
    2021-05-04 13:57:08 🗓️ El próximo 18 de mayo nuestra presidenta @BeatrizCorredor participa en el #CongresoCEAPI @CEAPIconsejo 
    
    👉Conver… https://t.co/J8M0DkAK2X
    2021-05-04 13:55:58 @AEDIVE @AsociacionAUVE @AVVEinfo @electromaps Nuevo punto de recarga rápida (dos tríos) de Iberdrola, ya operativo… https://t.co/b3QqXd8qya
    2021-05-04 13:53:27 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:52:33 Rítmica Elegance consigue el noveno puesto en la 1ª fase de la Liga Iberdrola @EleganceRitmica #Rivas #RivasActual… https://t.co/qAtKqDVoFT
    2021-05-04 13:51:50 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:49:53 RT @addbreizhou: Tout y est expliqué. Un scandale financier, ecologique, économique.@barbarapompili arrêtez de vous entêter avec ce projet,…
    2021-05-04 13:49:34 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:47:07 @iberdrola Eehhh... Los Stormtroopers eran los malos... No sé si lo sabíais 🙄
    2021-05-04 13:45:13 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:43:58 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 13:42:23 RT @Jeep_35000: Le gouvernement envoie des bateaux de la marine nationale protéger iberdrola sté Espagnole qui va nous détruite la baie de…
    2021-05-04 13:41:00 La seguridad es siempre lo PRIMERO ➡️ Vigila la ubicación de las redes eléctricas al instalar un invernadero y real… https://t.co/uXXPy2MRJU
    2021-05-04 13:40:41 @iberdrola Con esas facturas estelares es muy acertada la campaña
    2021-05-04 13:38:15 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 13:37:42 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:35:13 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:33:37 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:33:08 @AlbertFabrega @iberdrola @AVIAenergias Está tirado de vehículo!!! 😂😂 👍👍💪💪
    2021-05-04 13:32:52 Axudas nin unha soíña, xa voló digo eu, para eso estan Iberdrola e INDITEX.
    
    O desta xente, como onde tantas outras… https://t.co/1hvslIz79T
    2021-05-04 13:32:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:31:41 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:30:49 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:30:40 Haciendo palmas con las orejas con el MWh a 80€ y quemando gas a todo trapo. https://t.co/PXbZ8eYVja
    2021-05-04 13:26:58 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:24:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:21:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:21:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:20:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:18:34 RT @ferugby: #FERugby |🚨Tenemos buenas noticias: el rugby femenino crece con el nuevo Campeonato de España M14 y M16 ‼️#MujeresEnRugby #Des…
    2021-05-04 13:18:26 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 13:18:02 RT @footters: Recta final de Reto Iberdrola, sólo quedan 6 plazas en la permanencia y 14 equipos luchando a muerte por ellas 🤯
    
    Muy atentos…
    2021-05-04 13:17:04 RT @somosfutfem: 🔴 HORARIOS I Consulta los horarios de los partidos del fin de semana en #RetoIberdrola
    
    ➡️ Los subgrupos Norte y Sur C jug…
    2021-05-04 13:16:28 RT @BadmintonESP: ‼ Publicada la convocatoria de la selección 🇪🇸 para el @iberdrola  Spanish Junior International 2021.
    
    ▶ Del 11 al 13 de…
    2021-05-04 13:16:13 RT @AlbertFabrega: Parada para desayunar. En Cetina. Llegó sobrado de energía 30%, pero tengo hambre. Cargando a 29 kW en cargador de @iber…
    2021-05-04 13:15:37 @emnavalon @iberdrola 🤣🤣🤣🤣
    2021-05-04 13:13:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:13:38 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @CR…
    2021-05-04 13:13:13 RT @wilcocks: Ves lo de marcelo y lo del barsa y queda claro que la ley NUNCA es igual para todos. Los ricos siempre favores. Inaginad con…
    2021-05-04 13:12:13 RT @castillayleon: 🎨 ¿Te imaginas pasear entre cuadros de Velázquez, Goya o Tiziano? La plaza de la Concordia de #Salamanca ha sido el prim…
    2021-05-04 13:11:12 @JakeELeefan @iberdrola Hola. Te facilitamos información sobre cómo reclamar https://t.co/2z3Yq3DaDf Saludos
    2021-05-04 13:11:12 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:08:22 Ves lo de marcelo y lo del barsa y queda claro que la ley NUNCA es igual para todos. Los ricos siempre favores. Ina… https://t.co/wbVgjADHo7
    2021-05-04 13:07:14 @Frakn007 @lopezdoriga Si no han tocado a los que regalaron las obras carreteras de cobro y el segundo piso elevado… https://t.co/U9vvXvllQD
    2021-05-04 13:04:32 @RFEVB @VoleyFemAbs @teledeporte @PalomadelrioTVE @c_corcelles @deportegob @COE_es @iberdrola @Luanvi… https://t.co/vxyizSRJJK
    2021-05-04 13:02:55 Hoy me han llamado de Iberdrola para preguntarme sobre el contrato de electricidad de la empresa en la que trabajé… https://t.co/VmrelnDrA7
    2021-05-04 13:02:14 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 13:01:47 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 13:00:39 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 12:59:59 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:59:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:58:43 SILENCE, BRAND https://t.co/7Z0W0HwKxp
    2021-05-04 12:58:20 RT @EllasSonFutbol: 📝⚽⚔️  R E S U L T A D O S
    
    #RetoIberdrola Gr. D
    Jornada 7️⃣
    🔲 Grupo Norte ⬇️
    https://t.co/50O60lHSlt
    
    🔳 Grupo Sur ⬇️
    ht…
    2021-05-04 12:58:03 RT @alexlopezpuig: La millor temporada del primer equip femení del @hockeyjunior en els 104 anys del @ClubJunior1917 Campiones de la Copa d…
    2021-05-04 12:57:44 📝⚽⚔️  R E S U L T A D O S
    
    #RetoIberdrola Gr. D
    Jornada 7️⃣
    🔲 Grupo Norte ⬇️
    https://t.co/50O60lHSlt
    
    🔳 Grupo Sur… https://t.co/nZPJsXKEfF
    2021-05-04 12:57:24 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:55:45 @aradecas @iberdrola Les decía (con todo el respeto) que tengo mucho dinero y no me importa pagar más. Jajaja. Tamb… https://t.co/5acgI1PRBw
    2021-05-04 12:53:48 A Bet 20 Years Ago Made It the Exxon of Green Power https://t.co/kfLBtQr7CA
    2021-05-04 12:52:25 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:49:13 RT @SDEibar: ✅ Nerea Gantxegi es la primera eibarresa que ha marcado con el Eibar en la Liga Primera Iberdrola
    
    #Eibarfem I #EibarHistory 📜
    2021-05-04 12:49:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:48:50 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:48:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:46:20 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 12:45:48 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 12:45:26 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:45:07 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 12:44:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:44:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:44:37 #HockeyEspaña  #PiliCampoy 
    Campeonato Liga Iberdrola 
      #CLubDeCampo 
    By #ArnauMartínezBenavent https://t.co/boJ3RX1c54
    2021-05-04 12:42:43 RT @cdpozoalbensefe: ⚽️ ¡Volvemos a la competición! Y lo hacemos en casa 👇
    
    🏆 Jornada 6 PlayOffs ascenso 1ª Iberdrola.
    🆚 @FundaAlbaFem
    🗓 Sá…
    2021-05-04 12:42:21 #HockeyEspaña  #PiliCampoy 
    Campeonato Liga Iberdrola 
      #CLubDeCampo 
    By #ArnauMartínezBenavent https://t.co/1Ourgjmqmv
    2021-05-04 12:42:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:42:08 @montcorgi @AlbertFabrega @iberdrola @AVIAenergias Por otra parte su precio de salida en 81mil euros
    2021-05-04 12:42:05 #HockeyEspaña  #PiliCampoy 
    Campeonato Liga Iberdrola 
      #CLubDeCampo 
    By #ArnauMartínezBenavent https://t.co/jKIMDF2vTu
    2021-05-04 12:40:38 @montcorgi @AlbertFabrega @iberdrola @AVIAenergias Ahora con 64€ más o menos tendrá para 1500km ya que aún le queda… https://t.co/imrZDygcef
    2021-05-04 12:39:29 It is remarkable to compare the trajectory of Jose Galan at @iberdrola to the fate of David Crane, the former CEO w… https://t.co/THvV9jCBqc
    2021-05-04 12:39:23 @iberdrola Tarifazo. Muy bonita la campaña de márketing.
    2021-05-04 12:39:16 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:37:53 #Renault &amp; @iberdrola aims for greener factories 
    
    ‘The energy company will supply the carmaker with long-term gree… https://t.co/ckcCia0V15
    2021-05-04 12:37:25 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 12:35:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:35:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:35:23 @marco_artime @AlbertFabrega @iberdrola @AVIAenergias bueno el precio esta a la par un deposito son unos 70 euros y… https://t.co/mvoK9Y54x0
    2021-05-04 12:34:49 @iberdrola Y q no nos dejen.
    2021-05-04 12:34:01 📢 Twitter's most shared #renewables content is tracked in FREE realtime reports and weekly digests by… https://t.co/HMMCZYcekp
    2021-05-04 12:33:06 Spanish energy giant Iberdrola has partnered with oil and gas majors bp and Enagas on a feasibility study to develo… https://t.co/Tpg4vqyDxd
    2021-05-04 12:30:06 Renault e Iberdrola una ‘acuerdo’ para alcanzar la huella de carbono cero | https://t.co/0DZNpKm3OV https://t.co/W4OHSUGzal
    2021-05-04 12:29:29 RT @GreenInnovati0n: A global solution for a global problem.
    Look the rise of #renewable energy 🙌 The #GreenEnergy of the future is already…
    2021-05-04 12:27:53 RT @Existimos_: En Segovia @iberdrola construirá la 2ª planta solar más grande de Europa https://t.co/Ah4FlCB7Mv
    
    Es necesario que las empr…
    2021-05-04 12:27:47 @emnavalon @iberdrola Dime cómo porque te juro que si no me llaman 20 veces al día.... Es desesperante
    2021-05-04 12:27:11 "Iberdrola está perfectamente posicionada para aprovechar el auge de las energías renovables en el mundo y en Estad… https://t.co/wv00SaX6cB
    2021-05-04 12:26:51 @aradecas @iberdrola Es que los que dan la chapa no son de Iberdrola, sino de intermediarios que comercializan el s… https://t.co/GmNnE4jvd3
    2021-05-04 12:26:44 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:26:36 @Renewables4ever @iberdrola A energia Eólica e Solar são o futuro das energias renováveis.
    2021-05-04 12:24:15 RT @AlbertFabrega: Parada para desayunar. En Cetina. Llegó sobrado de energía 30%, pero tengo hambre. Cargando a 29 kW en cargador de @iber…
    2021-05-04 12:22:51 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:22:22 RT @AtletiFemenino: 🏧 Bernabé continuará como rojiblanca hasta 2⃣0⃣2⃣3⃣
    
    ℹ https://t.co/CY0JmIkZJK
    
    🔴⚪ #AúpaAtleti https://t.co/S6eEdyIg9E
    2021-05-04 12:22:08 @iberdrola Les pido POR FAVOR que dejen el acoso día y noche que NO estoy interesada en NINGUNA de sus ofertas. Y,… https://t.co/H8yKaI5H9u
    2021-05-04 12:22:07 RT @FCBfemeni: 🚨 [ÚLTIMA HORA] 🚨
    
    🤝 Acord per a la renovació de @Llcortes14  ▶ https://t.co/UL4Eo6WnUe
    
    🤝 Acuerdo para la renovación de Llu…
    2021-05-04 12:20:02 📰 ¿Aún no has leído el reportaje realizado por @InTecnica a nuestro CEO Emilio Sánchez Escámez sobre la mayor plant… https://t.co/5tWcrwVo2I
    2021-05-04 12:19:37 @iberdrola Muchas gracias! Qué suerte!
    2021-05-04 12:18:44 RT @sustraierak: Iberdrola quiere poner #ParqueSolar de 600 hectáreas en Peralta y Andosilla.
    Graves #impactos para fauna!!
    Se pueden prese…
    2021-05-04 12:18:27 Talls de llum de dimecres
    https://t.co/S0eHw7Fy3H
    
    Iberdrola recorda que demà dimecres es tallarà el subministramen… https://t.co/oyafmFVIXl
    2021-05-04 12:17:05 Bajad la luz, pedazos de sith. https://t.co/sq5f8JraBr
    2021-05-04 12:14:13 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:13:54 RT @ReinasDelBalon: Desde @ReinasDelBalon queremos agradecer a @iberdrola su apoyo por la #IgualdadDeGénero y por apostar por la mujer en e…
    2021-05-04 12:13:44 RT @AUBOURG2: @Porglo @cemoi83 @lagrostA @E2Villiers Ici finance espagnole iberdrola 
    Éolien offshore. Le projet de parc en baie de Saint-B…
    2021-05-04 12:13:03 RT @CICconstruccion: El CSCAE e Iberdrola sellan una alianza estratégica para impulsar la acción del Observatorio 2030  https://t.co/qy5rsI…
    2021-05-04 12:12:51 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:12:41 Ya está el resumen de la jornada 28 de @PrimerIberdrola en el blog 
    
    https://t.co/D0UaaXwJr0
    
    #FutbolFemenino… https://t.co/X4H51QQaqI
    2021-05-04 12:11:18 @luisitowisper @robertoallende4 @frantastikko @AlbertFabrega @iberdrola @AVIAenergias Además que el coche te vale e… https://t.co/kEXHSvQCHk
    2021-05-04 12:11:00 #Avangrid, cierra el primer trimestre del año con un beneficio neto de 334 millones de dólares (unos 278,2 millones… https://t.co/SPxWNpTZMq
    2021-05-04 12:09:54 RT @trastomina: @omy_rr @0rllugoso @facua @aldroenergia @Naturgy @Endesa @iberdrola @masmovil @Telefonica Es una práctica habitual y diaria…
    2021-05-04 12:08:33 @AlexisJR92 @robertoallende4 @frantastikko @AlbertFabrega @iberdrola @AVIAenergias Por eso digo que es un Timo🤣osea… https://t.co/Si1stUv6qw
    2021-05-04 12:06:25 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 12:03:36 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 12:03:35 RT @AlbertFabrega: Parada para desayunar. En Cetina. Llegó sobrado de energía 30%, pero tengo hambre. Cargando a 29 kW en cargador de @iber…
    2021-05-04 12:03:31 @robertoallende4 @AlbertFabrega @iberdrola @AVIAenergias Tambien te dejo esto, quizás sea de tu interés. Si le das… https://t.co/KH0i8BkBIf
    2021-05-04 12:01:48 Además de este éxito junto al Club Rítmico Colombino, la gimnasta del @grseneca celebra el reconocimiento como depo… https://t.co/yb6kx7GXgw
    2021-05-04 12:01:41 RT @elexportadordi: Son varios los países campeones en la lucha contra la #pandemia, y uno de ellos es #Australia. Su mercado ofrece excele…
    2021-05-04 12:00:27 No es así, brand https://t.co/SoKfvlEhp2
    2021-05-04 12:00:03 @robertoallende4 @AlbertFabrega @iberdrola @AVIAenergias Menuda respuesta de subnormal. Pero bueno, veo que no tien… https://t.co/qd6VjX23L5
    2021-05-04 11:59:32 @nuncdrama Yo casi be dicho que la social action a la que me refería era quemar Iberdrola
    2021-05-04 11:58:42 RT @fotosport_es: 📷GALERÍA | Ya tienes en la web las imágenes del @PozuFem - @CE_Seagull de ayer 👉 https://t.co/SZQZN0Zu7Q https://t.co/roc…
    2021-05-04 11:58:37 la madre que los parió 🥺
    Estais acabando con el monte en Galicia, q solo se ven mierdas gigantes de estas y os vais… https://t.co/QXY1Jgqm71
    2021-05-04 11:58:26 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:57:09 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 11:57:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:56:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:55:18 RT @elexportadordi: Las #infraestructuras son uno de los #sectores más atractivos de #Australia para la oferta española. #Carreteras, #vías…
    2021-05-04 11:55:13 Enrique Mateo Martín es un estudiante de cuarto de la ESO del IES López de Arenas que ha diseñado un prototipo de s… https://t.co/4dRXo8m46O
    2021-05-04 11:55:10 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:54:30 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:54:15 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:53:25 RT @PKayDee: https://t.co/q1K9K7zMn6 BP and Iberdrola to study solar-to-green H2 to partially decarbonise a refinery in Spain. Project woul…
    2021-05-04 11:52:54 @OffshoreWINDbiz @AilesMarines @iberdrola @Iberdrola_En Nope, fishing has a clear priority over useless #offshore… https://t.co/PtU913cwBu
    2021-05-04 11:51:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:51:23 @luisitowisper @robertoallende4 @frantastikko @AlbertFabrega @iberdrola @AVIAenergias Jajaja claro, claro... Con lo… https://t.co/sak2i1Jg79
    2021-05-04 11:51:06 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:49:06 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:49:04 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 11:48:33 👏 Un acuerdo entre @CSCAE e @iberdrola para actuar contra la #emergencia #climática desde la rehabilitación y la re… https://t.co/pKrIPBv5Qy
    2021-05-04 11:48:32 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Est… https://t.co/qDrw9VQi4B
    2021-05-04 11:48:18 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:48:11 Los 15 Perfiles Digitales más demandados por las empresas vía @iberdrola 
    #empleo #DigitalTransformation #Formación 
    https://t.co/1qkymBsXtF
    2021-05-04 11:48:00 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:47:47 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:45:41 #FERugby |🚨Tenemos buenas noticias: el rugby femenino crece con el nuevo Campeonato de España M14 y M16 ‼️… https://t.co/8VZzhy3Wsq
    2021-05-04 11:45:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:45:03 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:44:45 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:43:56 RT @WSjp_insight: 🇪🇸https://t.co/NiW06IPJBF
    caixabank bancosantander GrupoBPopular BBVA BancoSabadell Bankia Abengoa Abertis acerinox Amade…
    2021-05-04 11:43:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:42:54 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 11:42:25 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:41:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:41:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:41:00 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:39:29 @robertoallende4 @frantastikko @AlexisJR92 @AlbertFabrega @iberdrola @AVIAenergias Si quieres gastar mucho en combu… https://t.co/dMLe0n6Beh
    2021-05-04 11:38:52 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:35:16 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 11:33:39 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:33:25 @robertoallende4 @AlexisJR92 @AlbertFabrega @iberdrola @AVIAenergias Tú lo que eres es un poco tonto,permútate si l… https://t.co/zWFAZ4kaNT
    2021-05-04 11:32:05 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:31:39 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:31:25 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:29:17 #LigaIberdrolaRugby​​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 e… https://t.co/hEhe5DON5Z
    2021-05-04 11:27:43 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:27:04 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:26:51 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:25:57 Ningún equipo de liga iberdrola quiere a jessic4?
    2021-05-04 11:23:44 @AytoParla El teléfono de la Policía local no funciona y quiero dar aviso de que hay supuestos técnicos de… https://t.co/33CxkSRAHP
    2021-05-04 11:22:26 ¡Enhorabuena, @LauraCartoon! Ha sido la ganadora en Twitter de una Alexa, esperamos que la disfrutes (y tu madre ta… https://t.co/YSmqodjShb
    2021-05-04 11:22:03 Iberdrola revisa a fondo el protocolo por corrupción bajo la lupa del caso Villarejo https://t.co/TbaylSNsOB
    2021-05-04 11:20:50 RT @Iberdrola_En: GREEN Hydrogen 💚 A new REVOLUTION in clean energy:
    
    ✅Lower emissions in factories and transportation.
    🔋👷🏻More growth and…
    2021-05-04 11:18:36 #RÍTMICA | El @clubMabel, el @clubhadar y @colonial_sport líderes de las 3 divisiones de la primera fase de… https://t.co/YYAF0rGItq
    2021-05-04 11:18:09 @consumidores Hola quería realizar una cosulta para saber como puedo reclamar contra @iberdrola... El Lunes de la s… https://t.co/tCSDR66BFf
    2021-05-04 11:18:01 Forward thinking is not rewarded as much as lobbying and crying on CNBC or 60 minutes about tax rates is
    
    Contrast… https://t.co/3BAoE1TK2N
    2021-05-04 11:17:46 @iberdrola Las facturas quien las emite, Darth Vader?
    2021-05-04 11:16:41 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:16:30 RT @gallifantes: - Buenas tardes llamamos de Iberdrola
    - No me voy a cambiar
    - Te ofrezco el 20% de descuento.
    - Como si fuese el 80. Estoy…
    2021-05-04 11:16:25 Alba Franco, Campeona de España🇪🇸 en el aparato de cuerda en la primera fase de la Liga Iberdrola de la tercera div… https://t.co/KbHeq91Tp1
    2021-05-04 11:16:20 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:14:01 La jugadora armera Nerea Gantxegi marcó el pasado sábado el gol de la victoria frente al UDG Tenerife en el complej… https://t.co/CAPOuoSHNA
    2021-05-04 11:12:44 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:10:32 RT @somosfutfem: 🔴 HORARIOS I Consulta los horarios de los partidos del fin de semana en #RetoIberdrola
    
    ➡️ Los subgrupos Norte y Sur C jug…
    2021-05-04 11:10:08 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:09:44 RT @ElJigo: Sérieusement, allez vous faire foutre avec vos Éoliennes en fond marin. Rien d’écologique, juste de la maille pour Iberdrola. V…
    2021-05-04 11:09:35 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:09:26 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:09:19 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:09:08 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:07:28 «En MKT nos preocupan mucho las connotaciones negativas que se asocian al #marketing. Haríamos mal si reconociésemo… https://t.co/dzFvPIu9x9
    2021-05-04 11:07:20 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:07:06 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:05:35 👏 Un acuerdo entre @CSCAE e @iberdrola para actuar contra la #emergencia #climática desde la rehabilitación y la re… https://t.co/aACWN4844P
    2021-05-04 11:03:42 @frantastikko @AlexisJR92 @AlbertFabrega @iberdrola @AVIAenergias A nadie le meten nada por vena, ahí tienes 2 opci… https://t.co/hj9VEd7TzU
    2021-05-04 11:03:42 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 11:03:23 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:01:21 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 11:01:15 @AlexisJR92 @AlbertFabrega @iberdrola @AVIAenergias Lo siento no discuto con personas que no saben escribir, sería ponerme a su altura
    2021-05-04 10:58:59 RT @addbreizhou: Tout y est expliqué. Un scandale financier, ecologique, économique.@barbarapompili arrêtez de vous entêter avec ce projet,…
    2021-05-04 10:57:29 RT @DeportivasI: 🏈"#Jornada9 de la @DHIberdrola: el @Rugby_Cisneros lidera el Grupo 1 y @lesabellesrc desciende"
    💻¡En la web os contamos lo…
    2021-05-04 10:55:03 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @CR…
    2021-05-04 10:54:10 RT @iberdrolamex: De acuerdo con el estudio de percepción #ImpulsoSTEM, el 90% de la juventud oaxaqueña desea continuar con sus estudios a…
    2021-05-04 10:53:28 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @San…
    2021-05-04 10:53:26 RT @jdominguezfd: En definitiva, la única opción de que 4 equipos de la Primera Iberdrola jugaran la próxima Champions sería que el Barça f…
    2021-05-04 10:52:54 En definitiva, la única opción de que 4 equipos de la Primera Iberdrola jugaran la próxima Champions sería que el B… https://t.co/GY1opCOoBb
    2021-05-04 10:52:37 Partnering for a #netzero future! We're teaming up with Iberdrola, Enagas on green H2 project at Spanish refinery… https://t.co/91dSlaJGzK
    2021-05-04 10:52:15 RT @lmransanz: @MarianoFuentesS Tras el fin de la cesión que nunca debió ser, @iberdrola aún okupa como parking la parcela dotacional públi…
    2021-05-04 10:50:58 Lamentablemente, la plaza en fase de grupos que obtendría el @FCBfemeni como campeón de Champions no repercutiría e… https://t.co/ZVcMIcvGOP
    2021-05-04 10:50:47 RT @DeportivasI: 🏈"#Jornada9 de la @DHIberdrola: el @Rugby_Cisneros lidera el Grupo 1 y @lesabellesrc desciende"
    💻¡En la web os contamos lo…
    2021-05-04 10:49:26 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 10:49:20 @PPEEBPORG @iberdrola @barbarapompili @Le_Figaro @afpfr @LePoint @lemonde_planete @EELV @LRcontreleolien… https://t.co/FohtO7dChU
    2021-05-04 10:48:12 #LigaIberdrolaRugby​​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 en… https://t.co/h4peVU1boS
    2021-05-04 10:48:02 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:46:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:45:37 La próxima temporada la @UWCL estrenará formato con una fase de grupos con 16 equipos. La Primera Iberdrola otorga… https://t.co/C8kHF42FgA
    2021-05-04 10:44:32 🏈"#Jornada9 de la @DHIberdrola: el @Rugby_Cisneros lidera el Grupo 1 y @lesabellesrc desciende"
    💻¡En la web os cont… https://t.co/zXD87vZFGf
    2021-05-04 10:43:23 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:42:17 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 10:42:01 RT @sustraierak: Iberdrola quiere poner #ParqueSolar de 600 hectáreas en Peralta y Andosilla.
    Graves #impactos para fauna!!
    Se pueden prese…
    2021-05-04 10:40:59 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @San…
    2021-05-04 10:40:44 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 10:40:22 RT @MarietaPazos: @MarianoFuentesS Todo el tema de la operación del aparcamiento de @iberdrola en la parcela pública que era para uso depor…
    2021-05-04 10:40:02 Avangrid (Iberdrola) eleva un 39% su beneficio en el primer trimestre y mejora sus perspectivas para 2021… https://t.co/i2Ct6ma6hc
    2021-05-04 10:39:35 @MarianoFuentesS Todo el tema de la operación del aparcamiento de @iberdrola en la parcela pública que era para uso… https://t.co/QtiH8QK2o3
    2021-05-04 10:38:12 RT @AUBOURG2: @Porglo @cemoi83 @lagrostA @E2Villiers Ici finance espagnole iberdrola 
    Éolien offshore. Le projet de parc en baie de Saint-B…
    2021-05-04 10:37:06 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:36:36 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:35:57 RT @AlbertFabrega: Parada para desayunar. En Cetina. Llegó sobrado de energía 30%, pero tengo hambre. Cargando a 29 kW en cargador de @iber…
    2021-05-04 10:34:45 @MarianoFuentesS Lo que realmente importa es que se avance DE VERDAD!... ha pasado ya más de medio año desde el ple… https://t.co/NpVL5NiX1U
    2021-05-04 10:34:39 RT @AlbertFabrega: Parada para desayunar. En Cetina. Llegó sobrado de energía 30%, pero tengo hambre. Cargando a 29 kW en cargador de @iber…
    2021-05-04 10:34:14 bp, Iberdrola y Enagás estudian un proyecto de hidrógeno verde https://t.co/x4XiUdjSzQ #hidrógenoverde #gasescombustibles
    2021-05-04 10:33:17 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:33:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:32:57 IBERDROLA
    De la propuesta de dos arriba, hizo la parte 1 y 2, ¿hará la C y después bajará? Ni idea. https://t.co/4DUkguNuMp
    2021-05-04 10:31:11 RT @vpmadrid: Seguimos trabajando por un sueño! Un mes para reunir los apoyos necesarios para jugar la liga Iberdrola 21/22. Nos acompañas?…
    2021-05-04 10:30:56 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:30:19 El próximo partido del @AtletiFemenino será el domingo a las 12 horas visitando al Madrid CFF en la jornada 29 de la Primera Iberdrola.
    2021-05-04 10:30:17 #Automaker @GroupeRenaultEV selects @iberdrola to #decarbonise operations in #Spain and #Portugal.… https://t.co/OXSLVLHCkw
    2021-05-04 10:29:45 Avangrid (Iberdrola) eleva un 39% su beneficio trimestral y mejora sus perspectivas para 2021 https://t.co/dizBOonNxz
    2021-05-04 10:29:22 RT @MarketCurrents: $AGR - Avangrid EPS beats by $0.40, beats on revenue https://t.co/2NaEqcQkbz
    2021-05-04 10:28:50 @GabrielMariya @EUeic @EUgreendeal @EnelGroup @Iberdrola_En @GalpPress @innovationatEDP @LafargeHolcim @SynerLeap… https://t.co/N74QWc2EW1
    2021-05-04 10:28:30 RT @iberdrola: #MayThe4thBeWithYou Pura buena energía hoy y TODOS LOS DÍAS 🎇. Unamos nuestras fuerzas para ganar esta lucha con seguridad,…
    2021-05-04 10:26:42 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:26:41 RT @Am__today: Ce partenariat a pour but de réduire les émissions de CO2.
    #ecologie #environnement 
    https://t.co/SDHzHxFi1A https://t.co/6C…
    2021-05-04 10:26:07 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 10:26:06 RT @lmransanz: @MarianoFuentesS Tras el fin de la cesión que nunca debió ser, @iberdrola aún okupa como parking la parcela dotacional públi…
    2021-05-04 10:23:46 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:22:17 Iberdrola revisa toda su política sobre corrupción bajo la lupa del caso Villarejo https://t.co/YMggMTACTU
    2021-05-04 10:22:13 RT @FutbolisticasES: La madridista Olga Carmona, MVP de la Jornada 28 en Futbolísticas | Informa: Sara Portillo.
    Futbolísticas - El periódi…
    2021-05-04 10:21:58 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:20:26 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:20:02 El @CordobaFemenino certifica su permanencia gracias al calendario. La composición de las jornadas finales le garan… https://t.co/Nimg7a4K30
    2021-05-04 10:19:55 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 10:19:22 RT @adrianomones: Mañana tendrá lugar, de 11:00 a 13:00 horas, un webinario organizado por @AEDIVE para conocer en detalle el plan #MOVES3…
    2021-05-04 10:18:50 RT @exp_empresas: Avangrid (Iberdrola) eleva un 39% su beneficio trimestral y mejora sus perspectivas para 2021 https://t.co/zZoyMHGdpU
    2021-05-04 10:18:46 RT @Jeep_35000: Le gouvernement envoie des bateaux de la marine nationale protéger iberdrola sté Espagnole qui va nous détruite la baie de…
    2021-05-04 10:17:06 #LigaIberdrolaRugby​​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 ent… https://t.co/gCLxPxWDBN
    2021-05-04 10:15:58 RT @pierrevabres: «@AilesMarines SAS, dont l’actionnaire principal est la multinationale espagnole de l’énergie @iberdrola, remporte la mis…
    2021-05-04 10:15:35 Avangrid (Iberdrola) eleva un 39% su beneficio trimestral y mejora sus perspectivas para 2021 https://t.co/zZoyMHGdpU
    2021-05-04 10:13:35 @9marialopez @CCVMDHF @ccvmoficial @rfe_hockey @iberdrola OLEEEEEEEEEEEEEEEEEEEEEEEE
    2021-05-04 10:13:01 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:12:45 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:12:31 RT @castillayleon: 🎨 ¿Te imaginas pasear entre cuadros de Velázquez, Goya o Tiziano? La plaza de la Concordia de #Salamanca ha sido el prim…
    2021-05-04 10:10:09 Avangrid (Iberdrola) eleva un 39% su beneficio en el primer trimestre y mejora sus perspectivas para 2021 https://t.co/UBwHgLYeyZ
    2021-05-04 10:10:00 🎨 ¿Te imaginas pasear entre cuadros de Velázquez, Goya o Tiziano? La plaza de la Concordia de #Salamanca ha sido el… https://t.co/u0BKLJ45e5
    2021-05-04 10:09:42 @Renewables4ever @iberdrola https://t.co/T3sIFNg16q
    2021-05-04 10:07:59 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @Rugb…
    2021-05-04 10:07:42 @9marialopez @CCVMDHF @ccvmoficial @rfe_hockey @iberdrola SUPERCAMPEONAS!!!!!!!!
    2021-05-04 10:05:13 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 10:01:47 @frantastikko @robertoallende4 @AlbertFabrega @iberdrola @AVIAenergias Y no olvides lo más importante, que indirect… https://t.co/CT8q8eEEFA
    2021-05-04 10:00:23 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:59:10 @atletismoRFEA @andreathlete @deportegob @COE_es @WorldAthletics @WASilesia21 @JomaSport @iberdrola… https://t.co/MYqKFvYzMR
    2021-05-04 09:58:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:56:22 ¡Estamos de estreno! En Grupo ADE ya tenemos disponible el chalet piloto de la Urbanización Almenar de Guadaíra. ¡A… https://t.co/w61aRVJzFP
    2021-05-04 09:55:52 @AlbertFabrega @iberdrola @AVIAenergias 64,71€ para hacer poco más de mil km, no entiendo el concepto eléctrico más allá de lo ecológico
    2021-05-04 09:55:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:55:36 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:55:35 RT @DHIberdrola: #LigaIberdrolaRugby​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entre @Rugb…
    2021-05-04 09:55:25 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:54:03 A Bet 20 Years Ago Made It the Exxon of Green Power 
    
    Iberdrola is a leader in wind and solar power, thanks largely… https://t.co/YFl8PThyQV
    2021-05-04 09:53:25 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:53:08 #LigaIberdrolaRugby​​​​​​​​​​​​ | Resumen del partido de la jornada 9 de la Liga @iberdrola de Rugby 2020-2021 entr… https://t.co/Gl1qmOzKMY
    2021-05-04 09:51:22 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:50:44 RT @nytclimate: Iberdrola is one of a handful of utilities seen as leading a new generation of “renewable majors,” comparable to the way oi…
    2021-05-04 09:49:53 El @CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo… https://t.co/bKwPhnOaKl
    2021-05-04 09:49:41 RT @Existimos_: En Segovia @iberdrola construirá la 2ª planta solar más grande de Europa https://t.co/Ah4FlCB7Mv
    
    Es necesario que las empr…
    2021-05-04 09:49:03 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 09:48:54 RT @BadmintonESP: ‼ Publicada la convocatoria de la selección 🇪🇸 para el @iberdrola  Spanish Junior International 2021.
    
    ▶ Del 11 al 13 de…
    2021-05-04 09:48:37 @AlbertFabrega @iberdrola @AVIAenergias De momento qué tal la experiencia? Está claro que se puede viajar con los e… https://t.co/xGXqmZOedB
    2021-05-04 09:48:30 ¿Sabías que @iberdrola @tuSEAT  y el grupo @VW_es han firmado una alianza estratégica para dar un nuevo impulso a l… https://t.co/IK5FI21xi7
    2021-05-04 09:47:37 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 09:47:30 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 09:46:42 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 09:45:50 Eg of one person not asleep at the wheel- https://t.co/e7B802ZBNG
    2021-05-04 09:44:59 RT @Accountable2019: Que la factura de la luz suba un 46% respecto a abril de 2020 es un crimen de clara autoría: giratorios PSOE y PP, Ibe…
    2021-05-04 09:44:44 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 09:43:45 RT @APPA_Renovables: ☀️🔌 ¡¡Reserva YA los días 7⃣ y 8⃣ de #Julio!!🔌☀️
    
    Vuelve #CongresoAutoconsumo con más fuerza que nunca. Muchas gracias…
    2021-05-04 09:43:45 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:43:36 RT @addbreizhou: Tout y est expliqué. Un scandale financier, ecologique, économique.@barbarapompili arrêtez de vous entêter avec ce projet,…
    2021-05-04 09:41:38 Oye, CM de @iberdrola, a ver si tú me puedes ayudar a averiguar mi número CUPS, que me he quedado sin batería esper… https://t.co/Cip7firvVo
    2021-05-04 09:40:09 El CSCAE e Iberdrola sellan una alianza estratégica para impulsar la acción del Observatorio 2030  https://t.co/qy5rsIWGBF
    2021-05-04 09:39:07 @APPA_Renovables @Gesternova @iberdrola @NexusEnergia @AEDIVE @Arturopdelucia @jmgmoya @jgonzalezcortes @jmg_velez… https://t.co/EIuWEyxrSA
    2021-05-04 09:37:54 RT @iberdrola: Más proyectos VERDES para cambiar el mundo tal y como lo conocemos ➡️ Desarrollamos inversiones por 75.000 millones de euros…
    2021-05-04 09:37:51 @AlexisJR92 @robertoallende4 @AlbertFabrega @iberdrola @AVIAenergias Hoy cualquiera por 3000 euros tiene un coche p… https://t.co/TuZ14beaIK
    2021-05-04 09:37:34 @AlbertFabrega @iberdrola @AVIAenergias Uhh , "en ruta de nuevo" un gran tema musical de allá por los ochenta.
    2021-05-04 09:37:33 RT @adrianomones: Mañana tendrá lugar, de 11:00 a 13:00 horas, un webinario organizado por @AEDIVE para conocer en detalle el plan #MOVES3…
    2021-05-04 09:37:12 #ResultadosEmpresariales
    Avangrid (#Iberdrola) eleva un 39% su beneficio y mejora sus perspectivas para 2021 
    https://t.co/HJhj30nrpR
    2021-05-04 09:35:00 Más proyectos VERDES para cambiar el mundo tal y como lo conocemos ➡️ Desarrollamos inversiones por 75.000 millones… https://t.co/8SJNpmxSrQ
    2021-05-04 09:33:14 RT @CSCAE: El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo promoverá inic…
    2021-05-04 09:33:09 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:32:47 El CSCAE e @iberdrola sellan una alianza estratégica para impulsar la acción del #Observatorio2030 
    
    Este acuerdo p… https://t.co/pf7ylcchSK
    2021-05-04 09:32:10 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:31:40 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:31:31 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:31:30 RT @SantaBadajoz: JORNADA 29 | PRIMERA IBERDROLA 🔴⚪️
    
    🆚 @VCF_Femenino
    🗓 09/05
    🕛 12:00
    📍 IDM El Vivero | Aforo limitado 50%
    
    𝕄𝕒𝕤𝕥𝕖𝕣ℂ𝕙𝕖: la r…
    2021-05-04 09:29:56 @pacoespiga Si nos facilitas tus datos por mensaje privado (referencia o DNI+primer apellido+dirección) revisamos l… https://t.co/xFUhRbhDOG
    2021-05-04 09:29:56 RT @adrianomones: Mañana tendrá lugar, de 11:00 a 13:00 horas, un webinario organizado por @AEDIVE para conocer en detalle el plan #MOVES3…
    2021-05-04 09:28:43 🔴 HORARIOS I Consulta los horarios de los partidos del fin de semana en #RetoIberdrola
    
    ➡️ Los subgrupos Norte y Su… https://t.co/SiTwR0AYv1
    2021-05-04 09:28:02 ▶ dpa-AFX: Deutsche Bank belässt Iberdrola auf 'Buy': FRANKFURT (dpa-AFX Analyser) - Die Deutsche Bank hat die Eins… https://t.co/Rb75oOpZtV
    2021-05-04 09:25:07 RT @footters: Recta final de Reto Iberdrola, sólo quedan 6 plazas en la permanencia y 14 equipos luchando a muerte por ellas 🤯
    
    Muy atentos…
    2021-05-04 09:24:22 @iberdrola Hoy os habéis pasado Twitter 👍..... Pero esas facturas de la luz son muy del lado oscuro jejejej
    2021-05-04 09:23:48 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:22:26 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:21:57 A Bet 20 Years Ago Made It the Exxon of Green Power - The New York Times https://t.co/joTlI0pHCR
    2021-05-04 09:21:32 @AlbertFabrega @iberdrola @AVIAenergias Te vas a dar un buen desayuno, con esa potencia de carga viendo lo de ayer… https://t.co/8DNQakMavC
    2021-05-04 09:21:19 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:20:47 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:20:07 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:19:58 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:19:50 👉@iberdrola participando en el evento de lanzamiento del #MesEuropeoDeLaDiversidad de la @ComisionEuropea y… https://t.co/4itwtVlUZm
    2021-05-04 09:19:38 RT @Renewables4ever: The Exxon of GREEN power: an unstoppable Spanish company and its boss set sky-high goals for a better planet ♻️
    
    ➡️ Le…
    2021-05-04 09:19:29 Después de esta crisis, el sector energético se convirtió en un oligopolio formado por Endesa, Iberdrola, Unión Fen… https://t.co/9zxJuElkti
    

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
    
    ♻️ Una jornada a @Enginyeria posa sobre la taula el… https://t.co/6Yq9T05I0Z
    2021-05-05 08:14:08 RT @NaturgyClientEs: Pues el ganador de manera absoluta es: el baño.
    Limpiar el baño a fondo quema 100 Kcal cada 20 minutos… ¡cada 20 minut…
    2021-05-05 08:10:22 @Naturgy Porque el servicio de atención al cliente es tan deficiente? Notificamos cambio de cuenta para los recibos… https://t.co/QLuztKRnff
    2021-05-05 08:00:12 @agballester @Naturgy Buenos días😀, gracias por contactarnos por este canal. Lamentemos la molestia ocasionada. ¿No… https://t.co/5O4rpXFdwv
    2021-05-05 08:00:09 He fotografiat la seva nova seu corporativa a #Barcelona per a @Naturgy Petita ella com una flama de gas que encén… https://t.co/us3rsKtQkc
    2021-05-05 08:00:04 Pues el ganador de manera absoluta es: el baño.
    Limpiar el baño a fondo quema 100 Kcal cada 20 minutos… ¡cada 20 mi… https://t.co/RuL0VSAOfy
    2021-05-05 07:56:43 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:53:33 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:51:12 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:50:58 @betosalesrio @AleFlu39 @pcfilho Desculpe a ignorância, Naturgy é a antiga CEG?
    2021-05-05 07:47:36 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:46:25 Hay sinvergüenzas y luego están los de @Naturgy @NaturgyClientEs
    2021-05-05 07:45:03 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:44:13 RT @Naturgy: ¡Apostamos por las #renovables por tierra, mar y aire! Y nuestro desempeño ha obtenido recompensa 🙌 porque en los 3 primeros m…
    2021-05-05 07:42:34 @sonanthinks @NaturgyClientEs @Naturgy @consumogob Hola Álvaro. Si necesitas poner una reclamación https://t.co/2z3Yq3DaDf Saludos
    2021-05-05 07:42:31 RT @Renta4: 📈#BOLSA: El selectivo español abre la sesión del #miércoles al alza. 
    
    🇪🇸#Ibex35 🔼+0,79% (8.900,4 puntos)
    
    Mayores subidas
    #Ban…
    2021-05-05 07:41:26 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:35:19 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:35:07 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:35:03 Doge ya vale mas que:
    
    1. Santander + Caixabank
    2. Bbva+Caixabank+Telefonica
    3.Naturgy+Repsol+Grifols+RedElectrica+… https://t.co/7goIOYBuXd
    2021-05-05 07:34:21 RT @AnastasiaKnt: Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servicio SIN…
    2021-05-05 07:33:48 Hola @Naturgy tengo el móvil de una de vuestras operadoras que está estafando a la gente. Ha dado de alta un servic… https://t.co/R9tsx0Mfwc
    2021-05-05 07:31:54 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-05 07:30:00 ¡Apostamos por las #renovables por tierra, mar y aire! Y nuestro desempeño ha obtenido recompensa 🙌 porque en los 3… https://t.co/PHrI5ER3jk
    2021-05-05 07:21:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-05 07:18:58 @MaxDeLaBarrera1 @PaulBeaty2 @teddyboylocsin market actions to me. The electric company as a type of natural monopo… https://t.co/GDfDGju44X
    2021-05-05 07:11:21 [Actualización] Por si alguien lo dudaba, la factura vino errónea y lo mejor es que ahora me mandan otra para cobra… https://t.co/VbqqJq8nEs
    2021-05-05 07:06:12 RT @Renta4: 📈#BOLSA: El selectivo español abre la sesión del #miércoles al alza. 
    
    🇪🇸#Ibex35 🔼+0,79% (8.900,4 puntos)
    
    Mayores subidas
    #Ban…
    2021-05-05 07:05:58 📈#BOLSA: El selectivo español abre la sesión del #miércoles al alza. 
    
    🇪🇸#Ibex35 🔼+0,79% (8.900,4 puntos)
    
    Mayores… https://t.co/79crvRo0Zr
    2021-05-05 07:02:08 El ibex +0,8%
    8.906 puntos
    
    2 valores en rojo: Naturgy y REE, descensos muy suaves
    
    Sabadell +3%
    ArcelorMittal buen… https://t.co/NV9jpibd9u
    2021-05-05 06:46:16 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-05 06:46:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-05 06:31:00 🤝 | ANERR y Naturgy ponen en marcha el servicio 'RehabilitA'. Os contamos en qué consiste 👉 https://t.co/48dlDJB98T @Anerr_ @Naturgy
    2021-05-05 06:21:07 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-05 05:56:32 🔴Naturgy dinamita este jueves la caldera de la central térmica de Anllares 👇
    @Naturgy @ccooLeon @UGT_Leon… https://t.co/kXsZqPL9qZ
    2021-05-05 05:53:05 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-05 05:39:48 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-05 04:58:44 @Bryan01miranda @Bryan01miranda  Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través d… https://t.co/uyPuraZKCt
    2021-05-05 04:58:27 @narmata @panamapacifico @APP_GOB_PA @AsepPanama @narmata Estimado cliente, permítanos ayudarle. Es importante efec… https://t.co/V2TLSVyRpZ
    2021-05-05 04:30:40 @dasanjur @dasanjur Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de nuestra pag… https://t.co/oIjm5sJyPu
    2021-05-05 04:14:50 @Naturgy @NaturgyPa ya se ha ido la luz 3 veces continua y aun estamos esperando q llegue queman las cosas ajenas d… https://t.co/V7r6ICJyGF
    2021-05-05 04:00:00 El equipo de análisis de Mirabaud ha rebajado la recomendación de @Naturgy de Comprar a Vender con n P. O. de 22,4… https://t.co/J3G5S5UlV9
    2021-05-05 03:50:26 Los colegios Agora Lledó de Castellón y San José HFI de Valencia pasan a la semifinal del Certamen Tecnológico Efig… https://t.co/YpJu7dw4ky
    2021-05-05 03:44:11 Casi casi naturgy se gana una mandada a la verga de mi parte
    2021-05-05 03:35:46 @ArturoCPrez1 @ArturoCPrez1  Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de nu… https://t.co/CAaax2HzCA
    2021-05-05 03:34:18 @milagros2231 @milagros2231  Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de nu… https://t.co/k2B9ajgmLB
    2021-05-05 03:30:56 @atiempoparciall @atiempoparciall  Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través… https://t.co/WNLbwfBMCd
    2021-05-05 03:28:46 @aabarria @aabarria Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de nuestra pag… https://t.co/c1TRZ0rqx1
    2021-05-05 03:28:15 @luzibeth27 @tvnnoticias @AcodecoPma @AlvaroAlvaradoC @luzibeth27 Buenas noches, a continuación le compartimos el s… https://t.co/MQcL4bOizq
    2021-05-05 03:28:00 @RocioMAcevedo @RocioMAcevedo Buenas noches, a continuación le compartimos el siguiente enlace con el fin de garant… https://t.co/t3vhEiQgx8
    2021-05-05 03:26:55 @AvilaAngie_ @AvilaAngie_  Buenas noches, a continuación le compartimos el siguiente enlace con el fin de garantiza… https://t.co/ljM09mGIWb
    2021-05-05 03:26:34 @Radamesenrique @AsepPanama @arfuentes18 @denunciapmoeste @TraficologoO @Radamesenrique Buenas noches, a continuaci… https://t.co/b6rA6GC6tt
    2021-05-05 03:26:01 @CHEF_RAUL @AsepPanama @NitoCortizo @TReporta @tvnnoticias @nexnoticias @radiopanama @CHEF_RAUL Buenas noches, a co… https://t.co/E4x17MIUZP
    2021-05-05 03:25:38 @Jaipower1 @TeamCableOnda @tigobusinesspa @Naturgy @Etesatransmite @TReporta @AsepPanama @AcodecoPma @Jaipower1 Est… https://t.co/aj42RMAlG2
    2021-05-05 03:25:18 @Natha4 @OOrtega16 @AsepPanama @Natha4 Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a tr… https://t.co/edmncYhmh2
    2021-05-05 03:25:04 @Amariaq06 @NitoCortizo @tvntrafico @Amariaq06 Estimado cliente, permítanos ayudarle. Es importante efectúe su repo… https://t.co/Zpn0JMfplG
    2021-05-05 03:24:39 @Ricardo18238608 @Ricardo18238608 Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través… https://t.co/jYN3ffQbrp
    2021-05-05 03:24:23 @kbethancourtt02 @NaturgyClientEs @panama_asep @AsepPanama @kbethancourtt02 Buenas noches, a continuación le compar… https://t.co/Nvoc7tUy13
    2021-05-05 03:24:06 @NoxxSoporte @NoxxSoporte Buenas noches, a continuación le compartimos el siguiente enlace con el fin de garantizar… https://t.co/JMtAcRtPVR
    2021-05-05 03:23:55 @zu_0109 @311Panama @AsepPanama @zu_0109 Buenas noches, a continuación le compartimos el siguiente enlace con el fi… https://t.co/wdAHmcqhnx
    2021-05-05 03:23:42 @mariekathy @mariekathy Buenas noches, a continuación le compartimos el siguiente enlace con el fin de garantizar s… https://t.co/HOxOOeC79i
    2021-05-05 03:23:30 @MauBowen @OOrtega16 @AsepPanama @MauBowen Buenas noches, a continuación le compartimos el siguiente enlace con el… https://t.co/37W2QLfBgw
    2021-05-05 03:22:53 No entiendo a esa gente de naturgy, 11 horas sin luz y aún no llega🤬 HARTAAAA
    2021-05-05 03:13:43 Los millones de Panamá no sirven para lucirse en dubai pero si vienen los chinos o los japoneses ahi si chillan. Si… https://t.co/id8721avt7
    2021-05-05 03:12:24 Los millones de Panamá no sirven para lucirse en dubai pero si vienen los chinos o los japoneses ahi si chillan… https://t.co/wA3Npq7MrV
    2021-05-05 03:09:44 @NaturgyPa cuantas veces al año les hago reporte? Si llueve se va la luz y si no llueve también se va por q no llue… https://t.co/Nv4dUycg3g
    2021-05-05 02:57:36 @Naturgy sin luz en colinas de San Francisco, la mitra en la Chorrera. No puedo hacer reporte porque no me me de mi… https://t.co/5TnoJiXb8X
    2021-05-05 02:45:12 @OOrtega16 @AsepPanama @OOrtega16 Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través… https://t.co/8PykttEcaY
    2021-05-05 02:45:02 @MarcelleMonter2 @MarcelleMonter2 Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través… https://t.co/CqbXsQnHMA
    2021-05-05 02:44:54 @Gabopana @AsepPanama @AcodecoPma @Gabopana Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte… https://t.co/ZqZsY4YpE6
    2021-05-05 02:44:34 @Kinnyo @Kinnyo Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de nuestra pagina… https://t.co/hcMCRwx3a0
    2021-05-05 02:44:24 @Cesar_OsorioV @AsepPanama @rpc_radio @Cesar_OsorioV Estimado cliente, permítanos ayudarle. Es importante efectúe s… https://t.co/ij8lxUqt0k
    2021-05-05 02:42:31 Todo el clero católico, algunos pastores evangélicos, las promotoras, todos los estudiantes de la UP, UTP, UNACHI y… https://t.co/YdzX6AKdXY
    2021-05-05 02:25:47 RT @NaturgyPa: @kaja1483 @RuizSolangel @OndasCentrales @OriginalStereo @amoriginal1180 @kaja1483 Estimado cliente, permítanos ayudarle. Es…
    2021-05-05 02:22:13 @Asociacion_DEC @aegonseguros @eoi @ImqEuskadi @iDE_Iberdrola @la_Mutua @Solimat72 @BainAlerts @igslasalle… https://t.co/UwM2tqbkcC
    2021-05-05 02:10:08 @betosalesrio Eu. Era vazamento. No meu apartamento e,como os técnicos depois constataram,no prédio todo.
    Essa pequ… https://t.co/atAC6tlQCC
    2021-05-05 01:54:15 @NaturgyPa Muchas gracias por la colaboración de Naturgy.  Hoy me resolvieron el problema de fluctuación de voltaje… https://t.co/vI2NFNnK9A
    2021-05-05 01:41:02 @kaja1483 @RuizSolangel @OndasCentrales @OriginalStereo @amoriginal1180 @kaja1483 Estimado cliente, permítanos ayud… https://t.co/L5tVR9Q2Fj
    2021-05-05 01:25:52 RT @Alfons_ODG: Històric! Una victòria dels moviments socials representats per l'Aliança Contra la Pobresa Energètica (@APE_Cat). 
    
    5 anys…
    2021-05-05 01:23:55 RT @Luiscar38744354: @Naturgy de Arraiján solo saben mandar correos y mas nada desde octubre solicitando un medidor para poder mudarnos a n…
    2021-05-05 01:22:20 RT @betosalesrio: Desculpem por trazer um assunto pessoal. 
    Moro no Rio, sou cliente da monopolista Naturgy.
    Minha conta de gás quadruplico…
    2021-05-05 01:18:27 @Naturgy de Arraiján solo saben mandar correos y mas nada desde octubre solicitando un medidor para poder mudarnos… https://t.co/eqiprMmWxd
    2021-05-05 01:15:26 @Naturgy buenas no puede ser que esta compañia para los pagos hay q cumplir y para que te pongan un medidor Nvo par… https://t.co/SC98UbuMhn
    2021-05-04 23:27:54 @JoseICarrera1 @JoseICarrera1 Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de n… https://t.co/0XBak4WN0a
    2021-05-04 23:26:54 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 23:17:23 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 23:15:03 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 23:07:29 @AsepPanama buenas tardes se acaba de disparar fusibles dl poste frente a mi casa en la Omar Burunga atencion a Naturgy
    2021-05-04 22:53:57 https://t.co/LO83EgwuvE https://t.co/FiqPkZWsV5
    2021-05-04 22:49:11 O gás de rua está caríssimo e a Naturgy deita e rola em sua hegemonia.
    Lá em casa também. https://t.co/CsOQCsAHkp
    2021-05-04 22:41:27 RT @betosalesrio: Desculpem por trazer um assunto pessoal. 
    Moro no Rio, sou cliente da monopolista Naturgy.
    Minha conta de gás quadruplico…
    2021-05-04 22:40:00 @betosalesrio Aqui na região da Freguesia/Anil a Naturgy já trocou a tubulação que chega da rua em uns 3 condomínio… https://t.co/w2qNv3j4DU
    2021-05-04 22:26:55 @betosalesrio No meu prédio é GLP a granel. Muito mais barato que o gás da Naturgy (ex-CEG, ex-Fenosa). O monopólio… https://t.co/rLhZvRx4rW
    2021-05-04 22:20:09 @betosalesrio se for atípico, vai ser mais uma ação contra a Naturgy
    2021-05-04 22:17:37 Desculpem por trazer um assunto pessoal. 
    Moro no Rio, sou cliente da monopolista Naturgy.
    Minha conta de gás quadr… https://t.co/scxZMgTGiu
    2021-05-04 22:04:35 @drakkovich @Naturgy Agradecemos o seu retorno, Diogo. Esperamos que sua solicitação tenha sido atendida. Qualquer… https://t.co/QLA2xySoVE
    2021-05-04 21:48:42 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 21:47:17 @WillyTolerdoo Creo que en Naturgy ya le tienen un puesto
    2021-05-04 21:40:49 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 21:39:19 @rau_jere Programa propio patrocinado por Naturgy
    2021-05-04 21:26:38 RT @NSQPHOY: @XapulinSindical @ComdataGroup @CGT @cgttelem @CgtAtento @cgtunisono @CGTGrupoGSS @CGTTeleperSevil @CGTACoruna @CGTLince @Info…
    2021-05-04 21:26:05 RT @NSQPHOY: @XapulinSindical @ComdataGroup @CGT @cgttelem @CgtAtento @cgtunisono @CGTGrupoGSS @CGTTeleperSevil @CGTACoruna @CGTLince @Info…
    2021-05-04 21:23:03 @XapulinSindical @ComdataGroup @CGT @cgttelem @CgtAtento @cgtunisono @CGTGrupoGSS @CGTTeleperSevil @CGTACoruna… https://t.co/jtamZLedia
    2021-05-04 21:21:43 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 20:55:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 20:52:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 20:47:28 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 20:41:42 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 20:40:17 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 20:30:49 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 20:25:00 Renault y Naturgy firman un acuerdo por la #movilidadsostenible con su #cocheelectrico https://t.co/vrGm40BMmM
    2021-05-04 20:21:44 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 20:17:31 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 20:15:51 RT @NaturgyPa: @vielsy22 @RTPanama @tvnnoticias @TReporta @LaVozDeVeraguas @vielsy22 Buenas tardes, queremos ayudarle. Con el fin de garant…
    2021-05-04 20:05:36 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 20:01:59 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 19:59:39 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 19:53:10 @vielsy22 @RTPanama @tvnnoticias @TReporta @LaVozDeVeraguas @vielsy22 Buenas tardes, queremos ayudarle. Con el fin… https://t.co/EahULHUBkS
    2021-05-04 19:47:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 19:40:52 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 19:37:56 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 19:30:54 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 19:22:02 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 19:19:06 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 19:06:05 @GiselaTunon Tienes razón, lo que pasa es que tengo poco tiempo de mudado al actual apto. y la coordinación del aca… https://t.co/B6tXrqLS7u
    2021-05-04 19:05:26 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 19:02:27 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 19:00:01 EnerNews | @Naturgy 1T 2021: Por qué casi duplicó sus ganancias
    
    El beneficio neto global alcanzó los US$ 462 millo… https://t.co/Qs7XfuN1dr
    2021-05-04 18:58:48 @Observe50000564 @CarmeCatalana Abans tenia Naturgy i em vaig canviar a Catgas , estalvio en la factura i l´atenció… https://t.co/NOqO6L1gvv
    2021-05-04 18:52:21 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 18:33:32 RT @sonanthinks: @NaturgyClientEs última factura con vosotros de más de 500 euros con unos cálculos completamente falsos. Luego os preguntá…
    2021-05-04 18:32:39 RT @sonanthinks: @NaturgyClientEs última factura con vosotros de más de 500 euros con unos cálculos completamente falsos. Luego os preguntá…
    2021-05-04 18:12:00 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 18:11:12 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 18:10:19 Estic fart de les trucades en nom de @Naturgy per fer-me canviar de proveïdor.
    2021-05-04 18:08:39 @NaturgyClientEs última factura con vosotros de más de 500 euros con unos cálculos completamente falsos. Luego os p… https://t.co/LEKUamQpO2
    2021-05-04 17:59:17 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 17:53:34 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 17:45:00 .@iberdrola is the Spanish energy brand in the sector’s top 10, according to intangible valuation consultancy… https://t.co/g4zM4iJoWB
    2021-05-04 17:40:55 @Naturgy dejadme de llamar de una maldita vez. Además de acosadores, tirais el dinero.
    2021-05-04 17:30:33 ¿ Es normal que @Naturgy me MOLESTE tres veces en un día, a pesar que he tenido la deferencia de informar a un tele… https://t.co/5adhLBziBk
    2021-05-04 17:22:05 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 17:14:02 Hola @Naturgy y @NaturgyClientEs 👋:
    La compañía con la que tenéis subcontratadas las llamadas para captar nuevos cl… https://t.co/jqixAnawxl
    2021-05-04 17:04:56 @muscluman El pla ja està fet, només cal mirar la publicitat de les tv, Endesa, naturgy, telefònica.....
    2021-05-04 16:54:10 Renault y Naturgy por la movilidad sostenible https://t.co/P7qFxQfBy1 https://t.co/6C1aHzvoB8
    2021-05-04 16:42:25 @Cerni_report @Naturgy Y ta llaman a la hora de la siesta!!!!
    2021-05-04 16:40:45 @linx_99 Cedae, naturgy, light....
    2021-05-04 16:39:26 RT @Cerni_report: 🚨Tened cuidado con los de @Naturgy 🚨
    
    Te llaman diciendo que son tu proveedor de electricidad y que te ofrecen un descuen…
    2021-05-04 16:23:04 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 16:16:36 @aegonseguros | @eoi | @ImqEuskadi | @iDE_Iberdrola | @la_Mutua | @Solimat72 | @BainAlerts | @igslasalle |… https://t.co/iZNdJsZQBJ
    2021-05-04 16:04:00 Producción de hidrógeno verde a partir de #AguasResiduales
    https://t.co/2dOWp3iegk
    @CT_EnergyLab
    @Naturgy
    2021-05-04 15:58:30 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:57:49 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 15:57:14 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 15:56:14 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:55:42 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 15:46:02 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:40:41 RT @AltFuelsOK: New Naturgy’s public #naturalgas station becomes operational in Catalonia https://t.co/z8lNS2WdJd
    @Naturgy puso en funciona…
    2021-05-04 15:38:13 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:35:33 @Cerni_report @Naturgy No solo esa empresa. Es una vergüenza cómo tienen tus datos, y cuando ven que acaba tu contr… https://t.co/TZy4WrWEB2
    2021-05-04 15:30:20 @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mi madre también la engañaron los de Endesa y cuando me di cuent… https://t.co/Ne6eynI0xf
    2021-05-04 15:26:34 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:24:53 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 15:24:05 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 15:22:05 @MaguyMagan @Naturgy ¡Buenas tardes ! Te hemos contestado al mensaje privado, revisamos tu caso por allí. ¡Gracias! 😀
    2021-05-04 15:21:41 New Naturgy’s public #naturalgas station becomes operational in Catalonia https://t.co/z8lNS2WdJd
    @Naturgy puso en… https://t.co/aU9oiPqXnC
    2021-05-04 15:18:48 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 15:18:17 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:18:03 @Cerni_report @Naturgy Eso lo hacen todos.
    2021-05-04 15:12:34 🚨Tened cuidado con los de @Naturgy 🚨
    
    Te llaman diciendo que son tu proveedor de electricidad y que te ofrecen un d… https://t.co/mTLLmK26YA
    2021-05-04 15:12:25 RT @NaturgyClientEs: Donde haya una buena jornada de limpieza que se quite el gimnasio.
    ¿Qué respuesta dirías que es la correcta? Cuéntanos…
    2021-05-04 15:10:32 Me tienen hasta la madre #HDSPM @Naturgy @NaturgyMx #pesimoservicio @PesimaAtencion @AtencionProfeco
    2021-05-04 15:06:59 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 15:05:45 @Naturgy vergonzoso en trato recibido por atención al cliente. No se me ha generado ninguna factura desde el 4 de m… https://t.co/FU5ahIkF7U
    2021-05-04 15:01:40 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:00:35 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 15:00:05 Donde haya una buena jornada de limpieza que se quite el gimnasio.
    ¿Qué respuesta dirías que es la correcta? Cuénta… https://t.co/7qeSfHlWNO
    2021-05-04 14:56:20 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 14:54:31 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:53:27 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 14:52:16 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 14:51:45 RT @nuriaantoli: @Women360Congres @MGG_2012 @Naturgy MUY TOP! MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO @Women360Congres 
    Presenta y…
    2021-05-04 14:48:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:46:58 Naturgy la re concha de tu hermana
    2021-05-04 14:43:19 @sadupo @CarmeCatalana Gracias, yo también hice hace unos años el cambio de Endesa hacia Som Energía y no me arrepi… https://t.co/9hQs6DKa7a
    2021-05-04 14:39:34 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 14:36:41 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:31:22 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 14:30:13 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:20:47 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:16:54 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:14:08 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 14:00:15 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 13:58:44 @Rosa_Cusco @Women360Congres @Naturgy @UB_IL3 @ticsalut @tic @bcn_ajuntament @AMMDE_ @FundacioFidem @EIT_Health_es… https://t.co/wprGv14IQU
    2021-05-04 13:58:29 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 13:49:52 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 13:49:38 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 13:47:57 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 13:46:40 RT @alberto_nige: @NaturgyClientEs tampoco me soluciona la #estafa. Como estoy en el mercado regulado con #comercializadoraregulada del gru…
    2021-05-04 13:46:20 @Women360Congres @MGG_2012 @Naturgy MUY TOP! MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO @Women360Congres 
    Pres… https://t.co/rkcOTcIIVk
    2021-05-04 13:45:07 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 13:28:03 @0rllugoso @aldroenergia @Naturgy @Endesa Lo que yo hice fue cambiarme y luego devolver los recibos, tienes hasta 8 semanas.
    2021-05-04 13:25:44 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:25:28 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:15:06 RT @saz2000: Personal de Bomberos NL estación 5 así como @911SanPedro atienden reporte de probable fuga de gas en vía pública, calle Río Ba…
    2021-05-04 13:14:36 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:12:50 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 13:12:41 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:11:51 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 13:11:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 13:09:04 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:09:02 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:05:38 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 13:05:14 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 13:04:32 RT @gemma_g_fabrega: Forestalia, Naturgy, Enel, Ignis, Repsol i Statkraft controlen el 80% dels projectes eòlics i fotovoltaics de les Garr…
    2021-05-04 13:03:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 13:02:40 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 12:49:35 RT @CT_EnergyLab: La #UnidadMixta de #GasRenovable ya dispone de #website, podrás seguir todos los avances y actividades del proyecto que @…
    2021-05-04 12:47:36 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 12:44:58 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 12:41:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 12:41:09 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 12:39:59 RT @Mara19875: Yo quiero un país en el que @Naturgy @NaturgyClientEs no robe a los clientes, porque inventarse el consumo que aparece en la…
    2021-05-04 12:37:53 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 12:37:51 Yo quiero un país en el que @Naturgy @NaturgyClientEs no robe a los clientes, porque inventarse el consumo que apar… https://t.co/ZcNFckPges
    2021-05-04 12:36:42 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 12:32:17 @Naturgy @NaturgyClientEs sois unos timadores. Inventarse el consumo de las facturas es robar!!
    #ladrones… https://t.co/0juMww3JcH
    2021-05-04 12:23:44 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 12:23:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 12:20:11 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 12:18:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 12:18:01 RT @Women360Congres: MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbete a…
    2021-05-04 12:17:58 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 12:17:55 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 12:17:42 RT @Women360Congres: MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbete a…
    2021-05-04 12:17:38 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 12:17:35 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 12:17:20 RT @Women360Congres: MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbete a…
    2021-05-04 12:17:16 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 12:17:13 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 12:16:57 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 12:16:53 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 12:16:48 RT @Women360Congres: MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbete a…
    2021-05-04 12:16:18 RT @Women360Congres: MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbete a…
    2021-05-04 12:16:14 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 12:16:11 RT @Women360Congres: Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25 de…
    2021-05-04 12:15:15 Carmen Fernández Alvarez, Directora de Talento Directivo y Cultura de @Naturgy nos dará la bienvenida el próximo 25… https://t.co/MDFdLdBcsL
    2021-05-04 12:10:47 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 12:10:20 RT @el_gran_henry: @0rllugoso @aldroenergia @facua @Naturgy @Endesa Muy bien, Twitter haciendo su magia. Excelente, que todos se enteren de…
    2021-05-04 12:10:14 RT @Fben_Fben: @0rllugoso @aldroenergia @facua @Naturgy @Endesa Es increíble que te pidan datos a través de una red social, en lugar de fac…
    2021-05-04 12:09:54 RT @trastomina: @omy_rr @0rllugoso @facua @aldroenergia @Naturgy @Endesa @iberdrola @masmovil @Telefonica Es una práctica habitual y diaria…
    2021-05-04 12:09:33 RT @0rllugoso: @aldroenergia @facua @Naturgy @Endesa Os estoy contestando por DM, pero veo que actuáis igual que por teléfono. No hacéis ni…
    2021-05-04 12:08:48 RT @AnahRuiz1: @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mí me llamó Naturgy (supuestamente) al móvil de empresa que tengo desde h…
    2021-05-04 12:07:42 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 12:07:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 12:06:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 12:05:17 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 12:04:59 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 12:04:55 @XSpiderxGameX @Naturgy Buenas tardes👋. Te agradecemos el contacto a través de este canal. Lamentamos la molestia.… https://t.co/sBHv21RxdE
    2021-05-04 12:03:36 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 12:00:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 11:58:49 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 11:58:20 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 11:56:33 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:55:12 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:54:31 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 11:49:06 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 11:46:23 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:43:56 RT @WSjp_insight: 🇪🇸https://t.co/NiW06IPJBF
    caixabank bancosantander GrupoBPopular BBVA BancoSabadell Bankia Abengoa Abertis acerinox Amade…
    2021-05-04 11:42:17 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:39:35 @Naturgy la próxima vez que me llaméis para ofertarme algo os llevo a consumo. Os he pedido educadamente 100 veces… https://t.co/AM7btXFF5X
    2021-05-04 11:38:41 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:32:37 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 11:31:58 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:29:17 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:28:35 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 11:26:48 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:25:43 RT @CarladelCarme: @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA- De M…
    2021-05-04 11:22:55 @9Balas9 @francescd LLUM- D'Endesa m'he passat a @Catgas_Energia 
    GAS- De Naturgy a @Catgas_Energia 
    TELÈFON+FIBRA-… https://t.co/6LVyYC8A5s
    2021-05-04 11:20:19 @aldroenergia @0rllugoso @facua @Naturgy @Endesa @aldroenergia no se llama mala praxis se llama estafa. Y encima co… https://t.co/3QEoL2Up8l
    2021-05-04 11:15:07 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 11:13:30 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:13:09 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 11:12:49 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 11:12:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 11:11:48 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 11:11:36 RT @Women360Congres: MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbete a…
    2021-05-04 11:11:32 RT @Women360Congres: MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo puede…
    2021-05-04 11:11:22 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:11:05 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:10:43 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:08:12 @0rllugoso @aldroenergia @Naturgy @Endesa Que panda de HDP
    2021-05-04 11:06:08 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 11:04:11 En La Peña, Las Minas, Herrera van 33 horas sin energía eléctrica, increíble. @AsepPanama @Naturgy @AlvaroAlvaradoC… https://t.co/NTaoE7en92
    2021-05-04 11:01:07 Renault y Naturgy por la movilidad sostenible https://t.co/XrbvrAozKg https://t.co/zqTFYviMuP
    2021-05-04 11:01:05 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 11:00:05 Renault y Naturgy por la movilidad sostenible https://t.co/rZksC0qudP
    2021-05-04 11:00:02 El equipo de análisis de Mirabaud ha rebajado la recomendación de @Naturgy de Comprar a Vender con n P. O. de 22,4… https://t.co/wIiN4rJZm1
    2021-05-04 10:56:10 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 10:55:50 MESA DEBATE: LA ERA DEL #BLOCKCHAIN EN FEMENINO
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    con:
    No te lo pu… https://t.co/bDdP9ezYYL
    2021-05-04 10:54:46 MESA DEBATE: INGENIO, EL ARMA CONTRA EL #CORONAVIRUS
    Presenta y Modera. @MGG_2012 Montse Guardia Guell.
    
    ¡Inscríbet… https://t.co/vN4p11RzAO
    2021-05-04 10:53:06 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 10:51:54 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 10:45:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 10:44:09 Renault y Naturgy por la movilidad sostenible https://t.co/ePwo1zjKbF https://t.co/Opdofa82S9
    2021-05-04 10:44:06 Renault y Naturgy por la movilidad sostenible https://t.co/McRZvdkaTM
    2021-05-04 10:43:48 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 10:41:01 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 10:36:35 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 10:35:09 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 10:32:12 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 10:30:15 Naturgy no estudia ninguna gran adquisición o fusión y no descarta más desinversiones https://t.co/tOw3M9eODr
    2021-05-04 10:30:10 RT @0rllugoso: Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por @Naturgy y @e…
    2021-05-04 10:27:38 @0rllugoso @aldroenergia @Naturgy @Endesa Perdón? Vaya panda de hdlgpgp
    2021-05-04 10:26:52 Novedades en el caso de esta persona y la estafa de @aldroenergia y sus comercializadoras haciéndose pasar por… https://t.co/vkwgIc40eq
    2021-05-04 10:21:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 10:21:25 @ecatolicasclm os presenta esta noticia del Colegio Sagrado Corazón de Guadalajara. https://t.co/KAASlWLhHr
    2021-05-04 10:21:11 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 10:18:56 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 10:01:56 @Naturgy Los avances en tecnología están muy bien pero si las empresas como Naturgy dan un servicio comercial y de… https://t.co/jh3kiDYeaR
    2021-05-04 10:00:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 09:42:53 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:41:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 09:40:49 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:39:56 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 09:39:42 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 09:38:13 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 09:36:13 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:35:22 Empleos de @Naturgy . Ajustando despidos, y contratando becarios. Después de la OPA que os van a hacer.... vais a s… https://t.co/nPnGE7jMth
    2021-05-04 09:35:10 RT @CT_EnergyLab: La #UnidadMixta de #GasRenovable ya dispone de #website, podrás seguir todos los avances y actividades del proyecto que @…
    2021-05-04 09:32:43 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 09:27:01 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 09:26:00 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:25:11 RT @Eurogas_Eu: Our April issue is out! This time we reflect upon the #hydrogeneconomy, #blending, #gas quality &amp; #digital solutions togeth…
    2021-05-04 09:24:55 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:23:17 RT @Naturgy: La vulnerabilidad energética es nuestro enemigo número 1. Por este motivo, cada año, nuestra @NaturgyFnd desarrolla diversas i…
    2021-05-04 09:19:18 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:17:08 @EnricSans @vadortitus @CarmeCatalana Tots son comercialitzadors del monopoli Endesa i Naturgy.
    Si voleu uns que no… https://t.co/9fSjKOAdRE
    2021-05-04 09:14:15 @Observe50000564 @Jordi__PC @CarmeCatalana T'explico.. no es el mateix fibracat que movistar o vodafone, tampoc es… https://t.co/uuur35Rilo
    2021-05-04 09:11:50 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 09:08:52 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 09:08:40 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 09:08:25 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 09:08:15 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 09:08:05 RT @Women360Congres: Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOTW p…
    2021-05-04 09:07:31 Inscríbete al V Congreso de #Tecnología y #Salud del próximo 25 de #mayo ✅ con un descuento del 30%.
    Código 👉PROMOT… https://t.co/preHR4UXCf
    2021-05-04 09:05:58 Naturgy mejora su beneficio neto gracias a su capacidad de adaptación https://t.co/NPWV4GGiUb https://t.co/lkOILS8N15
    2021-05-04 09:03:54 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 09:01:19 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:59:44 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:57:17 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:55:44 Definitivamente, el plan del MOP es que se caigan los cinco puentes de Volcán a Cerro Punta. 
    Ayer tarde Naturgy tu… https://t.co/DHg5wrCIks
    2021-05-04 08:55:00 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:53:48 Naturgy inaugura una estación pública de GNL-GNC en Manresa https://t.co/vQoT9y8IlI #GNC #GNL #GNV #gasescombustibles #Manresa
    2021-05-04 08:44:53 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:44:01 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:42:44 La rehabilitación de los edificios apuesta por la eficiencia energética en el Proyecto “RehabilitA” @Anerr_… https://t.co/UWqd2GQYFV
    2021-05-04 08:40:50 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 08:40:08 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:39:01 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 08:38:42 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:38:32 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 08:38:22 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:38:22 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:38:01 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 08:37:33 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 08:35:01 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:34:45 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 08:34:34 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Que indefensión para esta señora
    Y que ineptitud de este gobierno… https://t.co/aMEZrZQJQ0
    2021-05-04 08:33:58 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Colla de lladres fastigosos!!
    2021-05-04 08:33:45 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:33:05 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:31:43 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:31:40 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:30:12 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:29:05 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:28:43 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:20:17 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:16:37 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:12:56 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 08:11:17 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:10:23 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 08:08:46 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:07:59 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:05:16 @0rllugoso @facua @aldroenergia @Naturgy @Endesa DESPRECIABLES
    2021-05-04 08:04:40 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 08:04:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:03:43 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:02:49 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 08:01:10 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:58:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:57:41 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:57:25 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:56:52 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:56:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:55:25 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:54:15 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:51:48 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:50:49 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:43:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:42:59 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:42:45 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:41:28 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:40:57 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:40:22 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:38:35 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 07:36:29 RT @Naturgy: La vulnerabilidad energética es nuestro enemigo número 1. Por este motivo, cada año, nuestra @NaturgyFnd desarrolla diversas i…
    2021-05-04 07:33:45 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:32:53 @Naturgy @NaturgyFnd Gracias a Florentino existe la vulnerabilidad energética
    2021-05-04 07:32:37 #Ciencia Renault y Naturgy por la movilidad sostenible https://t.co/4y0ZDM4dhh
    2021-05-04 07:31:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:30:00 La vulnerabilidad energética es nuestro enemigo número 1. Por este motivo, cada año, nuestra @NaturgyFnd desarrolla… https://t.co/vKKSlnB0AK
    2021-05-04 07:25:33 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:25:32 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:25:15 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:24:39 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:24:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:23:57 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:23:35 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:23:30 RT @NSQPHOY: @XapulinSindical @ComdataGroup @CGT @cgttelem @CgtAtento @cgtunisono @CGTGrupoGSS @CGTTeleperSevil @CGTACoruna @CGTLince @Info…
    2021-05-04 07:20:29 Compartimos el video del evento "Cybersecurity Trends" con los CISOs de #Iberia, #Naturgy, #Codere y #Sanitas.
    Si t… https://t.co/YbLkZcfxVg
    2021-05-04 07:20:08 RT @colegiosansu: Un colegio de Aragón, semifinalista del Certamen Tecnológico Efigy de @NaturgyFnd  https://t.co/BM5IYwTmZd vía @heraldoes…
    2021-05-04 07:19:49 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:18:45 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 07:18:43 RT @NSQPHOY: @XapulinSindical @ComdataGroup @CGT @cgttelem @CgtAtento @cgtunisono @CGTGrupoGSS @CGTTeleperSevil @CGTACoruna @CGTLince @Info…
    2021-05-04 07:18:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:15:08 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:14:41 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-04 07:13:11 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:11:44 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:08:48 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:08:32 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:07:47 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 07:07:31 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:02:25 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 07:02:18 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:58:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:58:32 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:58:05 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:55:32 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:52:26 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:51:12 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:50:32 @aldroenergia @0rllugoso @facua @Naturgy @Endesa @consumidores Me an visto cara de tonto???????
    2021-05-04 06:49:51 @aldroenergia @0rllugoso @facua @Naturgy @Endesa El 12 de abril 37 euros el 4 de mayo 83euros joder que negocio… https://t.co/1MypXvrRvB
    2021-05-04 06:48:30 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:47:45 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:47:07 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:46:55 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:44:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:44:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:43:12 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-04 06:42:49 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:42:42 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:42:12 @0rllugoso @facua @aldroenergia @Naturgy @Endesa @facua
    2021-05-04 06:41:42 @0rllugoso @facua @aldroenergia @Naturgy @Endesa 83 euros viviendo una persona sola??????? Pero esto que es????
    2021-05-04 06:41:14 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:40:59 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Espero que se ponga en contacto alguien conmigo a lo largo del dia
    2021-05-04 06:40:52 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 06:39:45 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Ademas consiguen tus datos bancarios sin el permiso del titular esto es robo denunciar
    2021-05-04 06:39:12 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-04 06:39:00 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Mira viviendo una persona solo https://t.co/Hp9K9lipF0
    2021-05-04 06:38:00 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:37:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:37:23 RT @trastomina: @omy_rr @0rllugoso @facua @aldroenergia @Naturgy @Endesa @iberdrola @masmovil @Telefonica Es una práctica habitual y diaria…
    2021-05-04 06:37:15 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 06:37:02 RT @omy_rr: @0rllugoso @facua @aldroenergia @Naturgy @Endesa Algo parecido están haciendo con @iberdrola y también @masmovil usando a @Tele…
    2021-05-04 06:36:56 RT @MichaelPoolSnc1: @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mi me llaman 2 veces a la semana para cambiar de empresa. No sé si…
    2021-05-04 06:36:48 RT @Evilotas: @RubenSanchezTW @VilloldoGarcia @0rllugoso @facua @aldroenergia @Naturgy @Endesa El viernes me llamaron de Repsol, para ofrec…
    2021-05-04 06:36:18 RT @LaGarraAmarilla: @cokita_anli @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Ves a la web https://t.co/OOdGJo7cn5 i segue…
    2021-05-04 06:36:09 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:36:09 RT @LaGarraAmarilla: @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Apunteu-vos a la llista Robinson p q no is facin tele màr…
    2021-05-04 06:35:58 RT @anagcerezo: @0rllugoso @facua @aldroenergia @Naturgy @Endesa Me consta que @Endesa también lo hace (a mi me acosan por teléfono) y @Nat…
    2021-05-04 06:35:53 RT @AnahRuiz1: @CatalanR2D2 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Yo todo esto lo sé porque un amigo fue "contratado" por una de…
    2021-05-04 06:35:50 @Naturgy y @ReformAnerr_com se unen para mejorar la #EficienciaEnergética de los edificios vía @CompromisoRSE https://t.co/aVD8KzDMAP
    2021-05-04 06:35:43 RT @CatalanR2D2: @AnahRuiz1 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Como a mí y vinieron a casa a mirar porque el contador marcaba…
    2021-05-04 06:35:35 RT @AnahRuiz1: @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mí me llamó Naturgy (supuestamente) al móvil de empresa que tengo desde h…
    2021-05-04 06:35:16 RT @SoyEsmeraldaSP: @0rllugoso @facua @aldroenergia @Naturgy @Endesa No sé si el afectado es familia tuya, pero lo sea o no, estamos orllug…
    2021-05-04 06:34:31 RT @facua: @0rllugoso @aldroenergia @Naturgy @Endesa Puedes presentar la denuncia ante la CNMC. Ya ha impuesto multas a varias eléctricas p…
    2021-05-04 06:34:24 RT @Fben_Fben: @0rllugoso @aldroenergia @facua @Naturgy @Endesa Es increíble que te pidan datos a través de una red social, en lugar de fac…
    2021-05-04 06:34:15 RT @0rllugoso: @aldroenergia @facua @Naturgy @Endesa Os estoy contestando por DM, pero veo que actuáis igual que por teléfono. No hacéis ni…
    2021-05-04 06:34:12 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:34:12 RT @aldroenergia: @0rllugoso @facua @Naturgy @Endesa Buenas tardes. Lamentamos toda mala praxis cometida en nuestro nombre. Nuestro dpto Ju…
    2021-05-04 06:33:28 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:33:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:32:59 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:31:51 @CarmeCatalana Jo també he canviat de Naturgy, i m'ha passa't el mateix
    2021-05-04 06:31:33 @0rllugoso @NuriMoBo @facua @aldroenergia @Naturgy @Endesa @consumcat
    https://t.co/uayeURdU8D
    2021-05-04 06:28:06 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:28:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:27:44 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:27:01 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 06:24:23 RT @Kikodelatorre2: @aldroenergia @Naturgy Me habeis vendido un descuento sin decirme que me cambiába de compañia! Como teneis mi numero de…
    2021-05-04 06:24:18 @Kikodelatorre2 @aldroenergia @Naturgy Otro más! No entiendo q la @CNMC_ES no intervenga YA y realice una inspecció… https://t.co/Vrm7Mfgz0W
    2021-05-04 06:23:19 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 06:22:37 @Dannyelsan1613 @NoaGresiva @nefeerr Pero deja de decir estupideces. ¿Qué mercado libre si hay un oligopolio en la… https://t.co/bwlL9hQDPo
    2021-05-04 06:21:49 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:20:59 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 06:20:32 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Aldro energia es un verdadero desastre y mejor estar lejos de ello… https://t.co/IHNvoq0sAt
    2021-05-04 06:19:19 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:19:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:17:44 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:15:12 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:10:14 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:10:07 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:09:42 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:07:57 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:06:25 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:06:22 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 06:00:39 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 06:00:38 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 05:59:35 @cdefreitas95 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Pues hay que denunciarlos
    2021-05-04 05:58:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:56:42 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:52:30 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 05:51:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:45:13 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:42:28 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 05:42:26 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Lo mismo le hicieron a mis padres unos comerciales de Iberdrola qu… https://t.co/fhtMzLfDWY
    2021-05-04 05:42:19 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:42:14 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:38:14 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:35:49 @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mi me hicieron lo mismo
    2021-05-04 05:35:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:34:40 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:34:16 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 05:33:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:32:03 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:31:47 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 05:31:34 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:29:28 @LaGarraAmarilla @cokita_anli @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Gràcies 🙂
    2021-05-04 05:29:12 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:29:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:27:16 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:26:58 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:25:24 RT @AnahRuiz1: @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mí me llamó Naturgy (supuestamente) al móvil de empresa que tengo desde h…
    2021-05-04 05:23:31 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-04 05:22:31 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:21:12 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:19:36 Renault y Naturgy por la movilidad sostenible https://t.co/nqyTQw9hLr
    2021-05-04 05:17:19 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:15:45 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:14:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:13:45 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:12:35 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:09:03 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:04:54 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 05:04:11 @GossipGirl1004 Yo tengo un seguro con Naturgy que me cubre la reparación y avería de todos los electrodomésticos d… https://t.co/WoHLEdGyhQ
    2021-05-04 05:02:54 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 05:00:58 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:59:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:56:36 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:56:08 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:53:58 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:49:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:38:59 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 04:34:16 RT @anagcerezo: @0rllugoso @facua @aldroenergia @Naturgy @Endesa Me consta que @Endesa también lo hace (a mi me acosan por teléfono) y @Nat…
    2021-05-04 04:33:55 RT @AnahRuiz1: @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mí me llamó Naturgy (supuestamente) al móvil de empresa que tengo desde h…
    2021-05-04 04:33:53 RT @CatalanR2D2: @AnahRuiz1 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Como a mí y vinieron a casa a mirar porque el contador marcaba…
    2021-05-04 04:33:49 RT @AnahRuiz1: @CatalanR2D2 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Yo todo esto lo sé porque un amigo fue "contratado" por una de…
    2021-05-04 04:32:54 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:30:46 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:27:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:27:02 @CarmeCatalana Ni un € a Naturgy!!!
    2021-05-04 04:26:52 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:26:51 @omy_rr @0rllugoso @facua @aldroenergia @Naturgy @Endesa @iberdrola @masmovil @Telefonica Y @ACNESP
    2021-05-04 04:23:28 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 04:21:55 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 04:19:56 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 04:14:20 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 04:10:59 En La Peña, Las Minas, Herrera 26 horas sin servicio eléctrico @AsepPanama @Naturgy @DenunciaChitre
    2021-05-04 04:10:42 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 04:05:23 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 04:05:23 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 04:02:26 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 04:00:21 RT @saz2000: Personal de Bomberos NL estación 5 así como @911SanPedro atienden reporte de probable fuga de gas en vía pública, calle Río Ba…
    2021-05-04 03:55:01 RT @Jessica62141399: #Naturgy me cambiaron el medidor sin consentimiento y sin que mi medidor estuviera en mal estado tengo todos los compr…
    2021-05-04 03:54:03 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 03:53:12 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 03:31:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 03:22:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 03:21:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 03:18:28 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 03:04:42 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 03:02:47 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 03:02:19 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 02:57:56 Naturgy y ANERR, la ‘gran alianza’ para mejorar la eficiencia energética de los edificios en España |… https://t.co/Npvr8h5aDe
    2021-05-04 02:57:41 25 horas sin energía eléctrica en La Peña, Las Minas, herrera. @AsepPanama @Naturgy
    2021-05-04 02:54:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 02:48:34 RT @MorenoG_Agustin: #ULTIMAHORA Naturgy pide autorización de Gobierno o Comunidad para restaurar la luz en Cañada Real y el alcalde de Riv…
    2021-05-04 02:32:14 RT @saz2000: Personal de Bomberos NL estación 5 así como @911SanPedro atienden reporte de probable fuga de gas en vía pública, calle Río Ba…
    2021-05-04 02:31:10 Personal de Bomberos NL estación 5 así como @911SanPedro atienden reporte de probable fuga de gas en vía pública, c… https://t.co/SiMhVGJ5vh
    2021-05-04 02:29:05 Por encima de los 22 euros #Naturgy activaría una potente señal de compra https://t.co/UZISJzr2JT https://t.co/WlAvbc2r8x
    2021-05-04 01:49:36 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 01:35:06 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 01:33:59 #Naturgy me cambiaron el medidor sin consentimiento y sin que mi medidor estuviera en mal estado tengo todos los co… https://t.co/hX73QDSrv4
    2021-05-04 01:32:04 Ya estoy cagadísima de frío y hacen 17 grados. Qué mal la voy a pasar este invierno, me cago en Naturgy.
    2021-05-04 01:27:40 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 01:27:34 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 01:18:52 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 01:17:10 RT @0rllugoso: @aldroenergia @facua @Naturgy @Endesa Os estoy contestando por DM, pero veo que actuáis igual que por teléfono. No hacéis ni…
    2021-05-04 01:15:39 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 01:12:46 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-04 01:07:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 01:06:26 @EydaGuardia @EydaGuardia Buenas noches, queremos ayudarle. Con el fin de garantizar su atención, le agradecemos re… https://t.co/A4psozSJN6
    2021-05-04 00:52:27 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 00:51:27 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 00:49:04 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 00:42:40 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 00:35:28 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 00:25:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 00:22:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 00:13:14 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-04 00:03:09 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-04 00:02:55 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:52:32 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 23:49:53 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:49:11 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 23:47:33 @NaturgyBrasil @Naturgy Serviço realizado hoje e concluído.
    Agradeço a atenção.
    O atendimento de SP foi ótimo. O do… https://t.co/7yRhx9FwMM
    2021-05-03 23:40:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:39:17 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:27:41 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:27:34 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Muy mala praxis sres de @aldroenergia Me pensaré seguir siendo cliente de uds.
    2021-05-03 23:27:01 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 23:26:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:26:20 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:26:10 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:25:57 @0rllugoso @facua @aldroenergia @Naturgy @Endesa A mi padre le hicieron exactamente lo mismo, reclamé, escucharon l… https://t.co/j0ReLtHo5F
    2021-05-03 23:19:30 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 23:15:36 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 23:15:33 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:14:55 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Lo mismo que @Endesa . Son todas una pandilla de ladrones. Endesa… https://t.co/qvoGOdDsK4
    2021-05-03 23:14:08 RT @facua: @0rllugoso @aldroenergia @Naturgy @Endesa Puedes presentar la denuncia ante la CNMC. Ya ha impuesto multas a varias eléctricas p…
    2021-05-03 23:13:03 hoy le he explicado amablemente a una muchacha de Naturgy que vivo alquilada y no puedo cambiar la compañía porque… https://t.co/xK7MwjhYsr
    2021-05-03 23:10:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:10:14 RT @Maximariana1: @Naturgy https://t.co/5mu7npW00e
    2021-05-03 23:08:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:07:48 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Tanto la luz como el gas contratar las tarifas del mercado regulad… https://t.co/gBjia3oADC
    2021-05-03 23:04:27 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 23:03:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 23:02:56 RT @Fben_Fben: @0rllugoso @aldroenergia @facua @Naturgy @Endesa Es increíble que te pidan datos a través de una red social, en lugar de fac…
    2021-05-03 23:00:33 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:57:10 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:54:29 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:49:30 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:47:16 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-03 22:46:26 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:46:10 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-03 22:45:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:44:13 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:41:01 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:40:56 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:40:01 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:39:29 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:38:58 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:38:16 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:37:57 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:37:21 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:36:14 RT @Rosa_Cusco: Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que hemos cr…
    2021-05-03 22:34:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:31:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:29:14 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:29:11 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:28:09 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:27:59 Gracias naturgy por mi día libre perdido 🙂🙂😡😡😡😡😡😡
    2021-05-03 22:27:29 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:27:13 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:26:29 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:25:22 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:24:52 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:24:36 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:24:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:22:02 RT @NestorRego: Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Bankia, 8.2…
    2021-05-03 22:21:38 Do ERTE ao ERE. Comeza a travesía: H&amp;M, 1.100; El Corte Inglés, 3.292; Santander, 1.823; BBVA, 3.450; CaixaBank-Ban… https://t.co/2RbRrYjewU
    2021-05-03 22:17:36 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:15:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:15:41 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:15:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:11:13 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:11:11 RT @gemma_g_fabrega: Forestalia, Naturgy, Enel, Ignis, Repsol i Statkraft controlen el 80% dels projectes eòlics i fotovoltaics de les Garr…
    2021-05-03 22:09:39 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:07:40 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:06:57 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:06:55 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:04:38 @jlbchabro1030 @jlbchabro1030 Buenas tardes, queremos ayudarle. Si en estos momentos no cuenta con el servicio, le… https://t.co/3ExnrkddM8
    2021-05-03 22:03:05 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:02:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 22:01:27 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 22:00:15 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:59:36 @drakkovich @Naturgy Oi, Diogo, tudo bem? Esperamos que possamos ajudar a solucionar o seu caso. Qualquer dúvida, e… https://t.co/lfMISfK7gw
    2021-05-03 21:54:00 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:53:58 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:52:50 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:52:40 @LaGarraAmarilla @cokita_anli @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Gràcies
    2021-05-03 21:52:35 Renault y Naturgy por la movilidad sostenible https://t.co/8bmkSnlNKo https://t.co/Q2fALq2E0K
    2021-05-03 21:50:30 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:49:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:48:43 @aldogdb @AsepPanama @aldogdb Buenas tardes, queremos ayudarle. Con el fin de garantizar su atención, le agradecemo… https://t.co/Amt5LHOXGt
    2021-05-03 21:48:29 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:46:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:45:26 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:43:56 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:42:12 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:40:18 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:39:54 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:38:06 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:37:50 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:36:10 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:35:56 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:35:00 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:34:35 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:32:49 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:32:16 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:31:57 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:31:21 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:31:12 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:27:26 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:27:11 RT @gemma_g_fabrega: Forestalia, Naturgy, Enel, Ignis, Repsol i Statkraft controlen el 80% dels projectes eòlics i fotovoltaics de les Garr…
    2021-05-03 21:27:09 RT @LaGarraAmarilla: @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Apunteu-vos a la llista Robinson p q no is facin tele màr…
    2021-05-03 21:26:32 @0rllugoso @Naturgy @Endesa @aldroenergia Algú de les dues companyies els hi deu passar les dades dels clients; sin… https://t.co/0H1HXgfa7D
    2021-05-03 21:24:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:24:19 RT @trastomina: @omy_rr @0rllugoso @facua @aldroenergia @Naturgy @Endesa @iberdrola @masmovil @Telefonica Es una práctica habitual y diaria…
    2021-05-03 21:23:12 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:22:41 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Que país y políticos de mierda en esta España 
    Las eléctricas hace… https://t.co/TBIPIqD0qv
    2021-05-03 21:22:24 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:22:02 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:21:29 @Cesar_OsorioV Buenas tardes, queremos ayudarle. Le agradecemos realizar su reporte a través del siguiente link:… https://t.co/FkC67ACCbV
    2021-05-03 21:20:09 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:19:01 RT @gemma_g_fabrega: Forestalia, Naturgy, Enel, Ignis, Repsol i Statkraft controlen el 80% dels projectes eòlics i fotovoltaics de les Garr…
    2021-05-03 21:17:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:17:36 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:17:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:16:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:16:35 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:13:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:11:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:11:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:10:59 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-03 21:10:20 Esta soy yo en #avatar!!#WomenTech21 el #25mayo en @Women360Congres estoy emocionada por el pedazo de programa que… https://t.co/qXln7vP8ZM
    2021-05-03 21:10:19 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:10:16 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:09:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:07:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:05:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:04:40 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-03 21:03:53 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 21:02:01 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:01:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:00:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 21:00:17 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-03 20:59:46 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:57:40 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:56:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:55:48 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:55:01 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:54:23 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:53:48 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:52:57 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:50:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:50:28 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:48:25 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:48:06 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:45:58 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:45:24 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:45:01 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:44:47 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:44:28 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:42:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:40:00 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:38:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    

    Rate limit reached. Sleeping for: 735
    

    2021-05-03 20:36:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:35:44 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:32:28 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:32:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:32:07 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Confio que ho resoldràs. Més gent com tu ens cal. 💪💪🔝🔝
    2021-05-03 20:31:50 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:31:41 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:31:01 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:30:14 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:29:20 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:28:06 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:27:32 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:27:16 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:26:51 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:26:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:24:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:23:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:20:42 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:20:31 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:17:48 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:16:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:15:41 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:14:58 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:14:54 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:14:54 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:14:21 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:12:59 @Naturgy buenas tardes, para la instalación de un boiler me puede indicar si puedo solicitar el servicio por este medio? Gracias
    2021-05-03 20:12:49 @aldroenergia @0rllugoso @facua @Naturgy @Endesa La mala praxis se la enseñáis vosotros mismos a los comerciales. L… https://t.co/hQay7ILo2q
    2021-05-03 20:12:20 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:12:14 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:12:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:11:15 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:11:06 @Naturgy @AsepPanama deseo saber por qué en mi factura del mes de abril no se me aplico el FET Extraordinario cuand… https://t.co/km4PWayXx2
    2021-05-03 20:10:58 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:10:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:08:17 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:07:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:05:55 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:05:53 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:05:28 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-03 20:03:34 @0rllugoso @facua @aldroenergia @Naturgy @Endesa No sé si el afectado es familia tuya, pero lo sea o no, estamos or… https://t.co/dnVO0cWwQq
    2021-05-03 20:03:10 RT @0rllugoso: @aldroenergia @facua @Naturgy @Endesa Os estoy contestando por DM, pero veo que actuáis igual que por teléfono. No hacéis ni…
    2021-05-03 20:02:28 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 20:01:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:00:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:00:36 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 20:00:00 .@iberdrola is the Spanish energy brand in the sector’s top 10, according to intangible valuation consultancy… https://t.co/jzMZCOUAQA
    2021-05-03 19:59:58 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:59:53 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:59:13 @Cesar_OsorioV @AsepPanama @Cesar_OsorioV Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a… https://t.co/hIGGNuW60m
    2021-05-03 19:58:58 @AnnLg18 @AnnLg18 Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de nuestra pagin… https://t.co/dZk7NH3Gdd
    2021-05-03 19:58:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:58:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:56:28 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:56:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:54:26 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:53:50 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:53:49 RT @LaGarraAmarilla: @cokita_anli @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Ves a la web https://t.co/OOdGJo7cn5 i segue…
    2021-05-03 19:52:06 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:52:01 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:50:43 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:49:30 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:49:30 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:48:08 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:48:04 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:47:55 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:47:00 @Naturgy deseo saber por qué en mi factura del mes se abril no se me aplico el FET Extraordinario cuando se supone… https://t.co/14GAx18BWN
    2021-05-03 19:46:54 RT @AnahRuiz1: @CatalanR2D2 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Yo todo esto lo sé porque un amigo fue "contratado" por una de…
    2021-05-03 19:46:34 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:46:05 RT @CatalanR2D2: @AnahRuiz1 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Como a mí y vinieron a casa a mirar porque el contador marcaba…
    2021-05-03 19:46:04 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:46:03 @CatalanR2D2 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Yo todo esto lo sé porque un amigo fue "contratado" p… https://t.co/tQY6HaUpLV
    2021-05-03 19:45:26 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:45:02 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:43:21 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:42:50 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:38:48 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:38:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:37:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:37:11 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:36:48 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:36:29 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:34:27 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:33:52 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:33:30 RT @NaturgyAr: Aboná tus facturas sin salir de tu hogar. Elegí tu medio de pago:
    Mercado Pago
    Botón de Pago por Oficina Virtual
    Débito auto…
    2021-05-03 19:33:17 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:32:28 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:32:21 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:32:01 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:31:55 RT @facua: @0rllugoso @aldroenergia @Naturgy @Endesa Puedes presentar la denuncia ante la CNMC. Ya ha impuesto multas a varias eléctricas p…
    2021-05-03 19:30:24 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:30:00 Aboná tus facturas sin salir de tu hogar. Elegí tu medio de pago:
    Mercado Pago
    Botón de Pago por Oficina Virtual
    Dé… https://t.co/ssymzDuLlJ
    2021-05-03 19:29:46 RT @Fben_Fben: @0rllugoso @aldroenergia @facua @Naturgy @Endesa Es increíble que te pidan datos a través de una red social, en lugar de fac…
    2021-05-03 19:24:23 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:23:17 @AnahRuiz1 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Como a mí y vinieron a casa a mirar porque el contador… https://t.co/jqeR3OQjhn
    2021-05-03 19:22:35 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:22:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:21:41 @0rllugoso @facua @aldroenergia @Naturgy @Endesa A nosaltres també ens va pasar, però ja vaig veure que era un frau… https://t.co/4eF0WNvEUI
    2021-05-03 19:20:49 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:19:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:19:38 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:19:06 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:18:54 @NaturgyPa Señores de Naturgy cómo es posible que en una residencia con 3  personas que laboran, la luz llegue tan… https://t.co/4AAXWCiEeq
    2021-05-03 19:17:53 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:14:43 RT @0rllugoso: Lo único que espero (a parte de esta gentuza sin remordimientos pague por engañar a personas mayores) es que tanto @Naturgy…
    2021-05-03 19:10:47 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:10:01 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:09:39 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:09:25 @LaGarraAmarilla @0rllugoso @facua @aldroenergia @Naturgy @Endesa Oooh! Muchas gracias, lo voy a hacer. Qué liberación! 😅
    2021-05-03 19:08:48 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:08:41 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:08:34 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:06:32 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:06:19 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:04:36 @LaGarraAmarilla @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Ja està 😋
    2021-05-03 19:03:31 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:02:37 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:02:14 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:02:13 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:01:04 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 19:00:57 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:00:56 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:00:15 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 19:00:10 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:59:26 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:59:04 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:58:40 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:58:09 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:57:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:57:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:57:05 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:55:55 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:55:23 @LaGarraAmarilla @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Mil gràcies 😉😉😉
    2021-05-03 18:55:14 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:55:03 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:54:51 @cokita_anli @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Fes-ho i ningú mes t’emprenyarà
    2021-05-03 18:54:33 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:54:26 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:53:26 RT @muyinteresante: Renault y Naturgy por la movilidad sostenible https://t.co/BmUVuH1SEG
    2021-05-03 18:53:17 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:52:46 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:52:29 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:52:21 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:52:10 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:51:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:51:08 @LaGarraAmarilla @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Gràcies 😉
    2021-05-03 18:49:45 RT @muyinteresante: Renault y Naturgy por la movilidad sostenible https://t.co/BmUVuH1SEG
    2021-05-03 18:49:34 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:47:22 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:45:17 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:45:11 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:43:35 RT @muyinteresante: Renault y Naturgy por la movilidad sostenible https://t.co/BmUVuH1SEG
    2021-05-03 18:42:50 RT @muyinteresante: Renault y Naturgy por la movilidad sostenible https://t.co/BmUVuH1SEG
    2021-05-03 18:42:40 Renault y Naturgy por la movilidad sostenible https://t.co/BmUVuH1SEG
    2021-05-03 18:42:12 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:42:02 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:41:33 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:40:18 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:39:12 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:39:08 Renault y Naturgy por la movilidad sostenible https://t.co/gzOFJzCyu8 https://t.co/UXYgLaxWuS
    2021-05-03 18:39:03 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:38:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:38:16 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:37:25 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:37:00 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:36:35 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:36:27 RT @gcapdevila: Surt petit en un peu de pàgina, però són pistes de com els Pirates del Carib es repartiran els fons europeus de recuperació…
    2021-05-03 18:35:15 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:35:12 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:34:29 @cokita_anli @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Ves a la web https://t.co/OOdGJo7cn5 i se… https://t.co/IbBimLgc9b
    2021-05-03 18:34:28 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:33:42 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:32:59 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:32:47 @LaGarraAmarilla @anagcerezo @0rllugoso @facua @aldroenergia @Naturgy @Endesa Com es fa?
    2021-05-03 18:31:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:31:07 @ZBer ¡Hola! gusto en saludarte, por el momento Naturgy no se encuentra disponible. Nos encontramos trabajando para restablecer el servicio.
    2021-05-03 18:28:08 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:27:45 @bancogeneral ya no se puede pagar @NaturgyPa desde banca en línea? Iba a pagar y se borró la opción para naturgy.
    2021-05-03 18:27:35 @0rllugoso @facua @aldroenergia @Naturgy @Endesa Quina vergonya!
    2021-05-03 18:27:23 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:25:54 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:23:55 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:23:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:23:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:22:37 @CarmeCatalana CatGas es una comercialitzadora de  Naturgy...
    2021-05-03 18:22:09 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:21:56 @Karenni49 Hola, por favor nos confirma que el servicio se ha normalizado. En caso de que no cuente con el servicio… https://t.co/SRtEc5Joqm
    2021-05-03 18:21:02 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:20:53 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:20:47 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:20:24 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:19:59 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:19:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:19:51 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:18:22 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:17:29 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:17:19 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:17:10 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:16:52 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:16:51 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:16:38 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:16:31 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:16:07 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:15:31 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:14:34 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:14:27 @aldogdb @AsepPanama @aldogdb Estimado cliente, permítanos ayudarle. Es importante efectúe su reporte a través de n… https://t.co/Xkmy5S6DDd
    2021-05-03 18:13:55 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    2021-05-03 18:13:45 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:12:25 RT @0rllugoso: Hola @facua. Quisiera denunciar a la compañía @aldroenergia y sus comercializadoras por engañar a una persona de 80 años, ha…
    2021-05-03 18:11:07 RT @CarmeCatalana: Dijous passat vam causar baixa del gas de casa de Naturgy i hem canviat a Catgas. 
    
    Avui ja m’han trucat per saber-ne el…
    


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


    
![png](output_59_0.png)
    



```python
#grafico de Naturgy
plt.hist(x=result["Naturgy"], bins=5, color='darkorange', rwidth=0.95)
plt.title('Histograma de valoración de Tweets de Naturgy')
plt.xlabel('Valoración del Tweet del 1 al 5')
plt.ylabel('Número de Tweets')
plt.show()
```


    
![png](output_60_0.png)
    



```python
boxplot = result.boxplot(rot=90)

```


    
![png](output_61_0.png)
    


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
    

Con una confinza del 95% podemos decir que las medias son iguales


```python

```


```python

```
