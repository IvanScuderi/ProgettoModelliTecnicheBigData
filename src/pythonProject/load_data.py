import pandas as pd
import pickle
import os
from datetime import datetime

##SCRIPT CHE CARICA IN MEMORIA SOTTO FORMA DI OGGETTO DATAFRAME LA PORZIONE DI DATASET FLICKR2X SPECIFICANDONE I CHUNKS
##EFFETTUA DELLE PRE ELABORAZIONI SUI DATI E SALVA I RISULTATI OTTENUTI SU UN FILE .CSV NELLA DIRECTORY DI PROGETTO
def loadjson(chunks=600):
    ##DATASET CONTENENTE INFORMAZIONI RIGUARDO POST INERENTI AL SOCIALWEB FLICKR IN FORMATO JSON
    data = pd.read_json(r"C:\Users\1997i\PycharmProjects\pythonProject\data\flickr2x.json",
                        lines=True, chunksize=100, orient="records")
    ##UTILIZZO IL FILE PROVA SOLO PER ESTRAPOLARE I NOMI DELLE COLONNE PER POTER CREARE UN DF VUOTO E FARE LE APPEND
    d = pd.read_json(r"C:\Users\1997i\PycharmProjects\pythonProject\data\prova.json", lines=True)

    columns = d.columns

    ##NUMERO DI CHUNK MASSIMI CHE SI VOGLIONO ESTRARRE DAL FILE FLICKR2X.JSON
    stop = chunks

    PICKLE_PATH = "C:\\Users\\1997i\\PycharmProjects\\pythonProject\\pickle"
    namePickle = "dataframe_"+str(stop)+".sav"
    namePickleClear = "dataframe_"+str(stop)+"_clear.sav"
    nameCsv = "dataframe_"+str(stop)+".csv"
    pathPickle = os.path.join(PICKLE_PATH, namePickle)
    pathCsv = os.path.join(PICKLE_PATH, nameCsv)
    pathPickleClear = os.path.join(PICKLE_PATH, namePickleClear)

    if os.path.exists(pathCsv):
        print("Il file csv del dataset formato dai chunks indicati esiste già: ", pathCsv)
        return (pathCsv, pathPickleClear)

    if os.path.exists(pathPickle):
        df = pickle.load(open(pathPickle, 'rb'))
    else:
        df = pd.DataFrame(columns=columns)
        c = 0
        ##AD OGNI ITERAZIONE PRENDO 100 ENTRY DAL FILE JSON
        print("Inizio il caricamento dei chunks del dataset specificato...")
        for d in data:
            df = df.append(d)
            c = c + 1
            print("CHUNKS CARICATI > ", c, "/", stop)
            if c == stop:
                print("Caricamento COMPLETATO!")
                break
        pickle.dump(df, open(pathPickle, 'wb'))

    print("Stampo il dataset:...")
    print(df)
    print("Stampo informazioni relativamente alle colonne del dataset:...")
    print(df.info())
    print("Stampo la lista di colonne del dataset:...")
    print(df.columns)
    print("Stampo un esempio di entry del dataset:...")
    print(df.iloc[2])
    print()

    ##ANALIZZO IL DATAFRAME OTTENUTO ED I SUOI ATTRIBUTI:

    ##---------COMMENTS-------------
    print()
    commentsNotZero = (df['comments'] != 0).sum()
    print("Numero di elementi della colonna 'COMMENTS' con valore != da zero > ", commentsNotZero)
    if commentsNotZero == 0:
        print("*COMMENTS: siccome la colonna non contiene alcuna informazione"
              " perchè i valori sono tutti posti a 0 si è deciso di dropparla")
        df.drop(columns=['comments'], inplace=True)
    print(df.columns)

    ##---------URLS-------------
    print()
    print("*URLS: si è deciso di droppare tale colonna poichè non mi servirà per le analisi che si andranno"
          " a condurre, specifica infatti eventuali urls accessori inseriti all'interno del post flickr")
    df.drop(columns=['urls'], inplace=True)
    print(df.columns)

    ##---------DESCRIPTION-------------
    print()
    print("Percentuale di elementi nulli nella colonna 'DESCRIPTION' in rapporto sul totale > ",
          df['description'].isna().sum()/df.shape[0])
    print("*DESCRIPTION: considerando che questa colonna può"
          " essere rilevante nel corso della nostra analisi, si è deciso di sostituire i "
          "valori nulli con la stringa 'No description'")
    df['description'].replace({None: "No description"}, inplace=True)
    print("Percentuale di elementi nulli nella colonna 'DESCRIPTION' in rapporto sul totale > ",
          df['description'].isna().sum()/df.shape[0])

    ##---------FAMILYFLAG, FRIENDFLAG, PUBLICFLAG, FAVORITE, PRIMARY, HASPEOPLE-------------
    print()
    print("Le colonne: 'FAMILYFLAG, FRIENDFLAG, PUBLICFLAG, FAVORITE, PRIMARY' contengono solo"
          " valori booleani, si convertono quindi al tipo bool")
    favorite = (df['favorite'] == True).sum()
    dropFavorite = False
    primary = (df['primary'] == True).sum()
    dropPrimary = False
    print("Numero di elementi posti a 'True' nella colonna 'FAVORITE' > ", favorite)
    print("Numero di elementi posti a 'True' nella colonna 'PRIMARY' > ", primary)
    if favorite == 0:
        print("Si decide avendo tutti gli elementi posti a 'False', di droppare la colonna 'FAVORITE'")
        df.drop(columns=['favorite'], inplace=True)
        dropFavorite = True
    if primary == 0:
        print("Si decide avendo tutti gli elementi posti a 'False', di droppare la colonna 'PRIMARY'")
        df.drop(columns=['primary'], inplace=True)
        dropPrimary = True
    df[['familyFlag', 'friendFlag', 'publicFlag', 'hasPeople']] = df[['familyFlag', 'friendFlag', 'publicFlag', 'hasPeople']].astype(bool)
    if not dropFavorite:
        df[['favorite']] = df[['favorite']].astype(bool)
    if not dropPrimary:
        df[['primary']] = df[['primary']].astype(bool)
    print(df.info())

    ##---------FARM, ICONFARM, ICONSERVER, ID, LICENSE, SECRET, SERVER-------------
    print()
    print("Per quanto riguarda le colonne 'FARM, ID, SECRET, SERVER', tali valori vengono impiegati per effettuare query"
          " alle FlickrAPI in modo da riuscire ad ottenere uno specifico post; si decide quindi di droppare tali colonne"
          " poichè non contengono informazioni utili per le future analisi")
    print("Si decide inoltre di eliminare anche le colonne 'ICONFARM, LICENSE, ICONSERVER' "
          "poichè non utili per le ulteriori analisi")
    df.drop(columns=['iconFarm', 'license', 'iconServer', 'farm', 'id', 'secret', 'server'], inplace=True)
    print(df.info())

    ##---------NOTES, ORIGINALFORMAT, ORIGINALHEIGHT, ORIGINALSECRET, ORIGINALWIDTH, PATHALIAS, ROTATION-------------
    print()
    print("Tutte le colonne con prefisso 'ORIGINAL***' contengono informazioni riguardo al dato multimediale originale,"
          " ossia quello prodotto dall'utente, informazioni quali: formato, altezza in pxl, larghezza in pxl, etc..."
          " Non traendo alcun beneficio rilevante da queste colonne, si decide di dropparle")
    df.drop(columns=['originalFormat', 'originalHeight', 'originalSecret', 'originalWidth'], inplace=True)
    notes = (df['notes'].apply(lambda x: len(x) != 0)).sum()
    print("Per quanto riguarda la colonne 'NOTES', contiene note aggiuntive al post, tale colonna è formate da oggetti list")
    print("Elementi del dataframe con colonne 'NOTES' contenente lista non vuota > ", notes)
    if notes == 0:
        print("Si decide quindi di eliminare tale colonna")
        df.drop(columns=['notes'], inplace=True)
    print("La colona 'ROTATION' ha anche essa informazioni tecniche riguardo al dato multimediale relativo al post\n"
          , df['rotation'],
          "\nSi decide quindi di eliminare anche questa")
    df.drop(columns=['rotation'], inplace=True)
    pathAlias = (df['pathAlias'] != "").sum()
    print("Numero di elementi nella colonna 'PATHALIAS' >", pathAlias)
    if pathAlias == 0:
        print("Si decide dunque di droppare anche tale colonna")
        df.drop(columns=['pathAlias'], inplace=True)
    print(df.info())

    ##---------MEDIASTATUS-------------
    print()
    mediaStatus = len(df['mediaStatus'].unique())
    print("Elementi della colonna 'MEDIASTATUS' > ", df['mediaStatus'].unique())
    if mediaStatus == 1:
        print("La colonna 'MEDIASTATUS' contiene al suo interno solo valori 'ready' e rappresenta"
              " lo stato del file multimediale che fa riferimento al post (foto, video). Non avendo informazioni rilevanti"
              " si è deciso di dropparla")
        df.drop(columns=['mediaStatus'], inplace=True)
    print(df.info())

    ##---------GEODATA-------------
    print()
    print("La colonna 'GEODATA' contiene oggetti dict in cui sono contenute info"
          " riguarda: latitudine, longitudine e accuratezza; di ogni rispettivo post.")
    print("Come prima cosa eliminiamo le entry nel dataset in cui il relativo valore di "
          "[geoData] è posto a NaN")
    df = df.dropna(subset=['geoData'])
    print("Valori NaN nelle relative colonne: ")
    print(df.isna().sum())
    ##estraggo in un dataframe la colonna geoData andando a convertire ogni elemento
    ##dei dict in una Series pandas
    geo = df['geoData'].apply(pd.Series)
    df.drop(columns=['geoData'], inplace=True)
    df['latitude'] = geo['latitude']
    df['longitude'] = geo['longitude']
    df['accuracy'] = geo['accuracy']
    print(df.info())

    ##---------VIEWS-------------
    print()
    print("*VIEWS: si converte il tipo di tale colonna da object ad int")
    df['views'] = df['views'].astype(int)
    print(df.info())

    ##---------OWNER-------------
    print()
    print("*OWNER: la colonna contiene elementi di tipo dict il cui unico campo di interesse"
          " risulta essere 'username' ossial il nome utente di chi ha generato il post")
    owner = df['owner'].apply(pd.Series)
    df['owner'] = owner['username']
    print(df.info())

    ##---------TAGS-------------
    print()
    print("*TAGS: la colonna in questione contiene per ogni entry una lista di dict in cui"
          " in ognuno di questi dict si ha informazione su un singolo tag impiegato nel post, "
          "si è deciso quindi di cambiare formato passando ad una lista di stringhe che rappresentano tag.")
    print("Esempio di elemento in 'TAGS' > ", df['tags'].iloc[0])
    tags = df.tags.apply(lambda x: [x[i]['value'] for i in range(len(x))])
    df['tags'] = tags
    print("Esempio di elemento in 'TAGS' dopo la modifica > ", df['tags'].iloc[0])

    ##---------DATEPOSTED-------------
    print()
    print("*DATEPOSTED: si converte tale colonna al tipo Datetime e si effettuano "
          "eventuali eliminazioni di righe che contengono date incongruenti")
    datePosted = pd.to_datetime(df['datePosted'])
    df['datePosted'] = datePosted
    t = datetime.now()
    ##non ci possono essere post che sono stati caricati in momenti successivi a datetime.now()
    errors = (df['datePosted'] > t).sum()
    index = df[df['datePosted'] > t].index
    if errors > 0:
        df = df.drop(index=index)
        df.reset_index(drop=True, inplace=True)
    print(df.info())

    ##---------DATETAKEN-------------
    print()
    print("*DATETAKEN: si converte tale colonna al tipo Datetime e si effettuano "
          "eventuali eliminazioni di righe che contengono date incongruenti")
    ##siccome ci sono diverse date errate in cui l'anno 2000 è segnato come 1000 e 2001 come 0001
    ##si procede a sostituire tali valori errati con quelli corretti
    df['dateTaken'] = pd.Series(df['dateTaken']).str.replace('1000', '2000')
    df['dateTaken'] = pd.Series(df['dateTaken']).str.replace('0001', '2001')
    dateTaken = pd.to_datetime(df['dateTaken'])
    df['dateTaken'] = dateTaken
    ##non ci possono essere post che sono stati caricati in momenti precedenti rispetto al tempo in cui è stata scattata la foto
    errors = (df['datePosted'] < df['dateTaken']).sum()
    index = df[df['datePosted'] < df['dateTaken']].index
    if errors > 0:
        df = df.drop(index=index)
        df.reset_index(drop=True, inplace=True)
    print(df.info())

    ##---------LASTUPDATE-------------
    print()
    print("*LASTUPDATE: si converte tale colonna al tipo Datetime")
    lastUpdate = pd.to_datetime(df['lastUpdate'])
    df['lastUpdate'] = lastUpdate
    print(df.info())

    pickle.dump(df, open(pathPickleClear, 'wb'))
    df.to_csv(pathCsv, index=False)

    return (pathCsv, pathPickleClear)
