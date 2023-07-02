from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.window import Window
import os
import pickle
import pandas as pd
import load_data
import gmplot
import webbrowser
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

GMAP_PATH = r"C:\Users\1997i\PycharmProjects\pythonProject\gmap"
IMG_PATH = r"C:\Users\1997i\PycharmProjects\frontend\static\img"

apikeys = "AIzaSyBAfHsJ7miedYS1yoU66VVoPA9fdAuv9DA"

spark_home = os.getenv("SPARK_HOME")
sc = SparkContext(master="local[*]", appName="SimpleApp")
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
sc.getConf().set('spark.sql.caseSensitive', 'true')
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")


def start(c=600):
    path = load_data.loadjson(c)[1]
    df = pickle.load(open(path, 'rb'))
    ##USO APACHE ARROW IN PYSPARK TRAMITE PYARROW PER CARICARE UN DATAFRAME PANDAS SULLA JVM OTTENEDO UN DATAFRAME SPARK
    sdf = spark.createDataFrame(df).cache()
    return sdf


def plot_in_gmap(sparkdf, elements=100):
    ##MI HA PERMESSO DI SCOPRIRE CHE I POST SONO TUTTI LOCALIZZATI A ROMA
    count = sparkdf.count()
    if elements > count:
        elements = count
    lat_lon = sparkdf.select('latitude', 'longitude')
    lat_lon = lat_lon.toPandas()
    lat = pd.Series.to_list(lat_lon['latitude'])
    lon = pd.Series.to_list(lat_lon['longitude'])
    gmap = gmplot.GoogleMapPlotter.from_geocode("Rome", apikey=apikeys)
    gmap.scatter(lat[0:elements], lon[0:elements], color='r', marker=True, size=12)
    filename = "map.html"
    path_to_file = os.path.join(GMAP_PATH, filename)
    gmap.draw(path_to_file)
    webbrowser.open('file://' + path_to_file)


def show_placeid_on_map(sdf, number=4):
    latlonplaceMax = sdf.groupby('placeId').max('latitude', 'longitude')
    latlonplaceMin = sdf.groupby('placeId').min('latitude', 'longitude')
    latlonplace = latlonplaceMax.join(latlonplaceMin, 'placeId', 'inner')
    count = latlonplace.count()
    if number <= count:
        count = number
    i = 0
    gmap = gmplot.GoogleMapPlotter.from_geocode("Rome", apikey=apikeys)
    elem = latlonplace.head(count)
    while i < count:
        dictrow = elem.pop().asDict()
        latitude = [dictrow['max(latitude)'], dictrow['max(latitude)'], dictrow['min(latitude)'],
                    dictrow['min(latitude)']]
        longitude = [dictrow['max(longitude)'], dictrow['min(longitude)'], dictrow['min(longitude)'],
                     dictrow['max(longitude)']]
        gmap.scatter(latitude, longitude, color='r', marker=True, size=5)
        gmap.polygon(latitude, longitude, face_color='pink', edge_color='cornflowerblue', edge_width=2)
        i += 1
    filename = "map.html"
    path_to_file = os.path.join(GMAP_PATH, filename)
    gmap.draw(path_to_file)
    webbrowser.open('file://' + path_to_file)


def top_n_post_per_views(sdf, n=10, photo=True, video=True):
    gmap = gmplot.GoogleMapPlotter.from_geocode("Rome", apikey=apikeys)
    data = sdf
    count = sdf.count()
    if not (photo and video):
        if photo and not video:
            ##SELEZIONO SOLO I POST RELATIVI A PHOTO
            data = sdf.filter(sdf['media'] == 'photo')
            count = data.count()
        if not photo and video:
            ##SELEZIONO SOLO I POST RELATIVI A VIDEO
            data = sdf.filter(sdf['media'] == 'video')
            count = data.count()
        ##SE SI VOGLIONO ENTRAMBE LE TIPOLOGIE DI POST NON EFFETTUO ALCUN FILTRAGGIO
        data = data.orderBy(desc('views'))
        ##SE L'UTENTE CHIEDE UN NUMERO DI ELEMENTI NELLA TOP SUPERIORE A QUELLI CHE CI SONO LI MOSTRO TUTTI
        numElem = n
        if n > count:
            numElem = count
        i = 0
        elem = data.head(numElem)
        while i < numElem:
            dictrow = elem.pop().asDict()
            latitude = dictrow['latitude']
            longitude = dictrow['longitude']
            views = dictrow['views']
            media = dictrow['media']
            owner = ''.join(e for e in dictrow['owner'] if e.isalnum())
            if media == 'photo':
                label = 'Photo #'+str(numElem-i)
                color = 'r'
                info = "photo by "+owner+" views: "+str(views)
            else:
                label = 'Video #'+str(numElem-i)
                color = 'b'
                info = "video by '"+owner+"' views: "+str(views)
            gmap.marker(latitude, longitude, label=label, color=color, info_window=info)
            i += 1
        filename = "map.html"
        path_to_file = os.path.join(GMAP_PATH, filename)
        gmap.draw(path_to_file)
        webbrowser.open('file://' + path_to_file)
        return
    else:
        photo = sdf.filter(sdf['media'] == 'photo')
        photo_count = photo.count()
        video = sdf.filter(sdf['media'] == 'video')
        video_count = video.count()
        photo = photo.orderBy(desc('views'))
        video = video.orderBy(desc('views'))
        numElemP = n
        numElemV = n
        if n > photo_count:
            numElemP = photo_count
        if n > video_count:
            numElemV = video_count
        i = 0
        elemP = photo.head(numElemP)
        elemV = video.head(numElemV)
        while i < numElemP:
            dictrow = elemP.pop().asDict()
            latitude = dictrow['latitude']
            longitude = dictrow['longitude']
            views = dictrow['views']
            owner = ''.join(e for e in dictrow['owner'] if e.isalnum())
            label = 'Photo #'+str(numElemP-i)
            color = 'r'
            info = "photo by "+owner+" views: "+str(views)
            gmap.marker(latitude, longitude, label=label, color=color, info_window=info)
            i += 1
        i = 0
        while i < numElemV:
            dictrow = elemV.pop().asDict()
            latitude = dictrow['latitude']
            longitude = dictrow['longitude']
            views = dictrow['views']
            owner = ''.join(e for e in dictrow['owner'] if e.isalnum())
            label = 'Video #'+str(numElemV-i)
            color = 'b'
            info = "video by "+owner+" views: "+str(views)
            gmap.marker(latitude, longitude, label=label, color=color, info_window=info)
            i += 1
        filename = "map.html"
        path_to_file = os.path.join(GMAP_PATH, filename)
        gmap.draw(path_to_file)
        webbrowser.open('file://' + path_to_file)
    return


def top_n_directions(sdf, n=3, np=10):
    ##ITINERARI DEI POST DEGLI N UTENTI PIU POPOLARI IN ORDINE CRONOLOGICO
    palette = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'white', 'gray',
             'lime', 'bisque', 'gold', 'azure', 'indigo', 'navy', 'aquamarine', 'palegreen', 'wheat', 'olive']
    gmap = gmplot.GoogleMapPlotter.from_geocode("Rome", apikey=apikeys)
    owner = sdf.groupby('owner').agg(sum('views').alias('views'))
    num = owner.count()
    if num >= n:
        num = n
    ##ORDINO IN MANIERA DECRESCENTE PER VIEWS
    owner = owner.orderBy(desc('views'))
    top = owner.head(num)
    i = 0
    for elem in top:
        row = elem.asDict()
        username = ''.join(e for e in row['owner'] if e.isalnum())
        ppowner = sdf.filter(sdf['owner'] == row['owner'])
        ppowner = ppowner.drop_duplicates(['latitude', 'longitude'])
        ppowner = ppowner.orderBy(desc('dateTaken'))
        nump = ppowner.count()
        if nump >= np:
            nump = np
        post = ppowner.head(nump)
        lat = []
        lon = []
        label = nump
        for p in post:
            d = p.asDict()
            latitude = d['latitude']
            longitude = d['longitude']
            lat.append(latitude)
            lon.append(longitude)
            info = "Post by "+username+" "+str(d['dateTaken'])+" views: "+str(d['views'])
            color = palette[i % len(palette)]
            gmap.marker(lat=latitude, lng=longitude, color=color, info_window=info, label=label)
            label -= 1
        i += 1
        route = list(zip(lat, lon))
        #gmap.directions(origin=route[nump-1], destination=route[0], waypoint=route[-2:0])
    filename = "map.html"
    path_to_file = os.path.join(GMAP_PATH, filename)
    gmap.draw(path_to_file)
    webbrowser.open('file://' + path_to_file)


def kmeans_heatmap(sdf, k=5, np=100):
    color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'white', 'gray',
             'lime', 'bisque', 'gold', 'azure', 'indigo', 'navy', 'aquamarine', 'palegreen', 'wheat', 'olive']
    gmap = gmplot.GoogleMapPlotter.from_geocode("Rome", apikey=apikeys)
    ##COME PRIMA COSA ELIMINO I POST LOCALIZZATI NELLO STESSO LUOG0
    data = sdf.drop_duplicates(['latitude', 'longitude'])
    ##COSTRUISCO UN NUOVO DATASET IN CUI AGGIUNGO LA COLONNA 'FEATURES' CHE CORRISPONDE
    ##ALLA LISTA DI FEATURES NECESSARIE ALL'ADDESTRAMENTO DEL MODELLO KMEANS
    vecAssembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features")
    ##L'EVALUATOR IMPIEGA IL COEFFICCIENTE DI SILHOUETTE OSSIA UNA MISURA DI VALUTAZIONE DI QUANTO
    ##GLI ELEMENTI ALL'INTERNO DI UNO STESSO CLUSTER SONO SIMILI RISPETTO A CLUSTED DIFFERENTI, IMPIEGA
    ##PER TALE CALCOLO LA DISTANZA QUADRATICA MEDIA
    evaluator = ClusteringEvaluator()
    new_sdf = vecAssembler.transform(data)
    kmeans = KMeans().setK(k).setSeed(42).setInitMode('k-means||')
    ##ISTRUISCO IL MODELLO ANDANDO A SELEZIONARE LE FEATURES D'INTERESSE
    model = kmeans.fit(new_sdf.select('features'))
    ##TRANSFORMED E' IL DATASET RISULTATO DELL'APPLICAZIONE DEL MODELLO IN CUI LA
    ##COLONNA 'PREDICTIONS' AMMETTE VALORI INTERI DA 0 A K-1 CHE CORRISPONDONO AL
    ##CLUSTER DI APPARTENENZA DEL RISPETTIVO ELEMENTO
    transformed = model.transform(new_sdf)
    evaluation = evaluator.evaluate(transformed)
    ##VADO AD ESTRARRE I CENTROIDI CHE SARANNO RAPPRESENTATI COME ARRAY PANDAS CONTENTI 2 SOLI ELEMENTI:
    ##LATITUDE E LONGITUDE, DEVO QUINDI ESTRARRE TALI FEATURES SINGOLARMENTE PER POTER PLOTTARE I PUNTI CON GMAP
    centers = model.clusterCenters()
    latitude = [e[0] for e in centers]
    longitude = [e[1] for e in centers]
    l = []
    for i in range(k):
        cluster = transformed.filter(transformed['prediction'] == i)
        num_elem = cluster.count()
        gmap.marker(lat=latitude[i], lng=longitude[i], color='cyan', label="C"+str(i),
                    info_window=("Centroid of cluster "+str(i)+" with "+str(num_elem)+" elements inside"))
        l.append(cluster)
    i = 0
    for elem in l:
        elem = elem.orderBy(desc('views'))
        views = elem.select('views').head(np)
        vs = [e.asDict()['views'] for e in views]
        latitude = elem.select('latitude').head(np)
        lats = [e.asDict()['latitude'] for e in latitude]
        longitude = elem.select('longitude').head(np)
        lngs = [e.asDict()['longitude'] for e in longitude]
        gmap.heatmap(lats=lats, lngs=lngs, radius=30)

        gmap.scatter(lats=lats[0:int(np/2)], lngs=lngs[0:int(np/2)], color=color[i % len(color)],
                     label=str(i), title=vs[0:int(np/2)])
        i += 1
    filename = "map.html"
    path_to_file = os.path.join(GMAP_PATH, filename)
    gmap.draw(path_to_file)
    webbrowser.open('file://' + path_to_file)
    return evaluation


def evaluate_kmeans(sdf):
    k = [3, 5, 7, 10, 15, 35]
    data = sdf.drop_duplicates(['latitude', 'longitude'])
    vecAssembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features")
    new_sdf = vecAssembler.transform(data)
    evaluations = np.zeros(len(k))
    evaluator = ClusteringEvaluator()
    j = 0
    for i in k:
        kmeans = KMeans().setK(i).setSeed(42).setInitMode('k-means||')
        model = kmeans.fit(new_sdf.select('features'))
        predictions = model.transform(new_sdf.select("features"))
        evaluations[j] = evaluator.evaluate(predictions)
        j += 1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(k, evaluations)
    ax.set_xlabel('k')
    ax.set_ylabel('coefficiente di silhouette')
    img_name = "evaluations.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.ioff()
    plt.savefig(img_path)


def flattened_tags_count_mean(sdf, num_esclusi, n=20, order_by_mean=False, escludi=True):
    ##QUERY CHE CALCOLA IL NUMERO DI VOLTE CHE COMPARE UN TAG IN UN POST
    ##E LA MEDIA DI VISUALIZZAZIONI CHE OTTENGONO LE FOTO IN CUI COMPAIONO I VARI TAG
    ##RESTITUISCE INOLTRE GLI N MIGLIORI TAG IN BASE ALL'ORDINAMENTO SCELTO:
    ##L'ORDINAMENTO DI DEFAULT E' SUL NUMERO DI VOLTA CHE UN TAG VIENE USATO, MENTRE
    ##SI PUO' ANCHE SCEGLIERE DI OTTENERE I MIGLORI TAG PER MEDIA VISUALIZZAZIONI
    data = sdf.groupby('tags').agg(count('tags').alias('count'),
                                   mean('views').alias('mean_views'))
    tag = data.select(explode('tags').alias('tag'), 'count', 'mean_views')
    tag = tag.groupBy('tag').agg(sum('count').alias('count'),
                                 mean('mean_views').alias('mean_views'))
    ##ELIMINO I TAG CON 1 SOLA OCCORRENZA POICHE' POCO SIGNIFICATIVI
    ordered_data = tag
    if escludi:
        ordered_data = tag.where(tag['count'] > num_esclusi)
    if not order_by_mean:
        ##APPLICO L'ORDINAMENTO SULLA COLONNA 'COUNT'
        ordered_data = ordered_data.orderBy(desc('count'))
    else:
        ##APPLICO L'ORDINAMENTO SULLA COLONNA 'MEAN_VIEWS'
        ordered_data = ordered_data.orderBy(desc('mean_views'))
    num_elem = ordered_data.count()
    if num_elem < n:
        n = num_elem
    ordered_data = ordered_data.toPandas()
    ordered_data = ordered_data.iloc[0:n]
    ##VISUALIZZO I RISULTATI
    ##1) NUMERO DI OCCORRENZE
    plt.ioff()
    plt.figure(figsize=(8, 10), tight_layout=True)
    p = sns.barplot(x='count', y='tag', data=ordered_data)
    plt.xlabel("numero di occorrenze")
    plt.ylabel("tag")
    img_name = "occurrence.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.savefig(img_path)
    ##2) VISUALIZZAZIONI MEDIE
    plt.figure(figsize=(8, 10), tight_layout=True)
    p = sns.barplot(x='mean_views', y='tag', data=ordered_data)
    plt.xlabel("visualizzazioni medie")
    plt.ylabel("tag")
    img_name = "mean_views.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.savefig(img_path)


##FUNZIONE CHE MI SERVIRA' PER UDF SPARK
@f.udf
def udf_month(dateTaken):
    calendar = {1: 'January',
                2: 'February',
                3: 'March',
                4: 'April',
                5: 'May',
                6: 'June',
                7: 'July',
                8: 'August',
                9: 'September',
                10: 'October',
                11: 'November',
                12: 'December'
                }
    return calendar[dateTaken.month]


##FUNZIONE CHE MI SERVIRA' PER UDF SPARK
@f.udf
def udf_year(dateTaken):
    return dateTaken.year


def year_stats(sdf, year):
    ##METODO CHE, UNA VOLTA SELEZIONATO UN DETERMINATO ANNO
    ##RESTITUISCE STATISTICHE PER OGNI MESE DI TALE ANNO RELATIVE AL NUMERO DI POST PER MESE
    ##ED ALLE VISUALIZZAZIONI MEDIE DEI POST IN UN DATO MESE
    data = sdf.withColumn('year', udf_year('dateTaken'))
    data = data.filter(data['year'] == year)
    data = data.withColumn('month', udf_month('dateTaken'))
    months = data.groupBy('month').agg(count('month').alias('numero_post'), mean('views').alias('mean_views'))
    x = np.array(['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'])
    y1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    d1 = months.select('month').collect()
    d1 = [e['month'] for e in d1]
    numero_post = months.select('numero_post').collect()
    numero_post = [e['numero_post'] for e in numero_post]
    mean_views = months.select('mean_views').collect()
    mean_views = [e['mean_views'] for e in mean_views]
    i = 0
    for m in d1:
        index = np.where(x == m)
        y1[index] = numero_post[i]
        y2[index] = mean_views[i]
        i += 1
    ##VISUALIZZO I RISULTATI
    ##1) PLOT: MESI-POST
    plt.ioff()
    plt.figure(figsize=(8, 12), tight_layout=True)
    plt.plot(x, y1)
    plt.xlabel("mesi")
    plt.ylabel("numero post anno: "+str(year))
    plt.xticks(rotation=90)
    img_name = "year_stats_1.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.savefig(img_path)
    ##2) PLOT: MESI-VIEWS
    plt.ioff()
    plt.figure(figsize=(8, 12), tight_layout=True)
    plt.bar(x, y2)
    plt.xlabel("mesi")
    plt.ylabel("visualizzazioni medie post anno: "+str(year))
    plt.xticks(rotation=90)
    img_name = "year_stats_2.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.savefig(img_path)


def total_years(sdf):
    data = sdf.withColumn('year', udf_year('dateTaken'))
    data = data.select('year').distinct().collect()
    data = [int(e['year']) for e in data]
    data.sort()
    return data


def owner_overview(sdf, n=20):
    ##RESTITUISCE VARIE INFO SUGLI UTENTI PIU' ATTIVI
    user = sdf.groupby('owner').agg(count('owner').alias('num_post'), mean('views').alias('mean_views'),
                                    max('datePosted').alias('lastPost'))
    user = user.orderBy(desc('lastPost'))
    ##AGGIUNO UNA COLONNA PER L'ID DEGLI UTENTI
    user = user.select("*").withColumn('id', row_number().over(Window.orderBy(desc('lastPost'))))
    num_elem = user.count()
    if num_elem < n:
        n = num_elem
    utenti = user.toPandas()
    utenti['id'] = utenti['id'].astype(str)
    utenti['id'] = "U" + utenti['id']
    utenti = utenti.iloc[0:n]
    ##VISUALIZZO I RISULTATI
    ##1) NUMERO DI POST PER UTENTE
    plt.ioff()
    plt.figure(figsize=(7, 10), tight_layout=True)
    sns.barplot(x='num_post', y='id', data=utenti)
    plt.xlabel("numero di post")
    plt.ylabel("id utente")
    img_name = "owner_overview_1.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.savefig(img_path)
    ##2) VISUALIZZAZIONI MEDIE PER UTENTE
    plt.ioff()
    plt.figure(figsize=(7, 10), tight_layout=True)
    sns.barplot(x='mean_views', y='id', data=utenti)
    plt.xlabel("visualizzazioni medie")
    plt.ylabel("id utente")
    img_name = "owner_overview_2.jpg"
    img_path = os.path.join(IMG_PATH, img_name)
    plt.savefig(img_path)
    ##COSTRUISCO LA TABELLA
    utenti = spark.createDataFrame(utenti)
    user = utenti.join(sdf, utenti.owner == sdf.owner, "inner")
    tag_per_user = user.groupBy(utenti.owner, 'tags').agg(count('tags').alias('utilizzi'))
    w = Window.partitionBy('owner').orderBy(col('utilizzi').desc())
    tgu = tag_per_user.withColumn('row', row_number().over(w)).where(col("row") == 1)
    utenti = utenti.join(tgu, 'owner', 'inner')
    utenti = utenti.orderBy(desc('lastPost'))
    utenti = utenti.toPandas()
    return utenti













































