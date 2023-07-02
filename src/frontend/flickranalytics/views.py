from django.http import HttpResponseRedirect
from django.shortcuts import render
import sys
import pandas
import os
sys.path.insert(0, 'C:\\Users\\1997i\\PycharmProjects\\pythonProject')
import main

sdf = None
valutato = False


def redirect(request):
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\evaluations.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\evaluations.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\occurrence.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\occurrence.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\mean_views.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\mean_views.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_1.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_1.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_2.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_2.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_2.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_2.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_1.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_1.jpg')
    return HttpResponseRedirect('/flickranalytics/')


def home(request):
    global sdf
    model = {'caricato': False,
             'home': True
             }
    if request.method == 'POST':
        c = request.POST.get('chunks')
        try:
            c = int(c)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova ad inserire un valore numerico come chunk..."
            return render(request, 'error.html', model)
        if c < 0:
            model['titolo'] = "Ops! Si è verificato un errore, prova ad inserire un valore numerico come chunk..."
            return render(request, 'error.html', model)
        sdf = main.start(c)
        df = sdf.toPandas()
        model['righe'] = df.shape[0]
        model['colonne'] = df.shape[1]
        model['caricato'] = True
        model['iter'] = df.iloc[0:20].iterrows()
    if sdf is None:
        model['caricato'] = False
    else:
        df = sdf.toPandas()
        model['caricato'] = True
        model['righe'] = df.shape[0]
        model['colonne'] = df.shape[1]
        model['iter'] = df.iloc[0:20].iterrows()
    return render(request, 'home.html', model)


def clear(request):
    global sdf
    global valutato
    valutato = False
    sdf = None
    model = {'caricato': False,
             'home': True
             }
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\evaluations.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\evaluations.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\occurrence.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\occurrence.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\mean_views.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\mean_views.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_1.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_1.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_2.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_2.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_2.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_2.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_1.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_1.jpg')
    return render(request, 'home.html', model)


def error(request):
    model = {}
    model['titolo'] = "Ops! Si è verificato un errore, " \
                      "devi caricare il Dataset prima di poter effettuare delle query..."
    return render(request, 'error.html', model)


def plot(request):
    global sdf
    df = sdf.toPandas()
    model = {'caricato': True,
             'completato': False,
             'nElem': df.shape[0]
             }
    if request.method == 'POST':
        elem = request.POST.get('elem')
        try:
            elem = int(elem)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if elem < 0:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        main.plot_in_gmap(sdf, elements=elem)
        model['completato'] = True
        model['p'] = elem
    return render(request, 'plot.html', model)


def placeid(request):
    global sdf
    df = sdf.toPandas()
    n_place = len(df['placeId'].unique())
    model = {'caricato': True,
             'completato': False,
             'nPlace': n_place
             }
    if request.method == 'POST':
        elem = request.POST.get('nP')
        try:
            elem = int(elem)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if elem < 0:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        main.show_placeid_on_map(sdf, number=elem)
        model['completato'] = True
        model['pId'] = elem
    return render(request, 'placeid.html', model)


def postperviews(request):
    global sdf
    df = sdf.toPandas()
    model = {'caricato': True,
             'completato': False,
             'foto': (df['media'] == 'photo').sum(),
             'video': (df['media'] == 'video').sum()
             }
    photo = True
    video = True
    if request.method == 'POST':
        post = request.POST.get('post')
        media = request.POST.get('media')
        if media == "foto":
            video = False
        if media == "video":
            photo = False
        try:
            post = int(post)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if post < 0:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        main.top_n_post_per_views(sdf, n=post, photo=photo, video=video)
        model['completato'] = True
        model['post'] = post
        model['media'] = media
    return render(request, 'postperviews.html', model)


def directions(request):
    global sdf
    df = sdf.toPandas()
    model = {'caricato': True,
             'completato': False,
             'users': len(df['owner'].unique())
             }
    if request.method == 'POST':
        utenti = request.POST.get('utenti')
        npost = request.POST.get('npost')
        try:
            utenti = int(utenti)
            npost = int(npost)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if (utenti < 0) or (npost < 0):
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        main.top_n_directions(sdf, n=utenti, np=npost)
        model['completato'] = True
        model['utenti'] = utenti
        model['post'] = npost
    return render(request, 'directions.html', model)


def kmeans(request):
    global sdf
    global valutato
    df = sdf.toPandas()
    model = {'caricato': True,
             'completato': False,
             'valutato': valutato
             }
    if request.method == 'POST':
        k = request.POST.get('k')
        hm = request.POST.get('hm')
        try:
            k = int(k)
            hm = int(hm)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if (k < 0) or (hm < 0):
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        sc = main.kmeans_heatmap(sdf, k=k, np=hm)
        model['sc'] = sc
        model['k'] = k
        model['cluster'] = True
        model['completato'] = True
        model['p'] = hm
    return render(request, 'kmeans.html', model)


def valutaK(request):
    global sdf
    global valutato
    model = {'caricato': True,
             'completato': False,
             'valutato': valutato
             }
    if not valutato:
        main.evaluate_kmeans(sdf)
        valutato = True
        model['valutato'] = valutato
        return render(request, 'kmeans.html', model)
    return render(request, 'kmeans.html', model)


def tag(request):
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\occurrence.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\occurrence.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\mean_views.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\mean_views.jpg')
    global sdf
    model = {'caricato': True,
             'completato': False,
             'mostra_tag': False
             }
    if request.method == 'POST':
        e = request.POST.get('escludi')
        escludi = False
        num_esclusi = 0
        ntag = request.POST.get('tag')
        ordinamento = request.POST.get('ordinamento')
        order = False
        O = "Occorrenze"
        E = "No"
        try:
            ntag = int(ntag)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if ntag < 0:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if e == "m1":
            escludi = True
            num_esclusi = 1
            E = "Si [ < 1 ]"
        if e == "m2":
            escludi = True
            num_esclusi = 2
            E = "Si [ < 2 ]"
        if e == "m3":
            escludi = True
            num_esclusi = 3
            E = "Si [ < 3 ]"
        if ordinamento == "views":
            order = True
            O = "Visualizzazioni"
        main.flattened_tags_count_mean(sdf, num_esclusi=num_esclusi, n=ntag, order_by_mean=order, escludi=escludi)
        model['completato'] = True
        model['mostra_tag'] = True
        model['N'] = ntag
        model['O'] = O
        model['E'] = E
    return render(request, 'tag.html', model)


def year(request):
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_1.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_1.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_2.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\year_stats_2.jpg')
    global sdf
    model = {'caricato': True,
             'completato': False,
             'mostra_foto': False
             }
    anni = main.total_years(sdf)
    model['year'] = anni
    if request.method == 'POST':
        anno = request.POST.get('year')
        try:
            anno = int(anno)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if anno not in anni:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        main.year_stats(sdf, anno)
        model['mostra_foto'] = True
        model['completato'] = True
        model['info'] = anno
    return render(request, 'year.html', model)


def active_users(request):
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_2.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_2.jpg')
    if os.path.exists(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_1.jpg'):
        os.remove(r'C:\Users\1997i\PycharmProjects\frontend\static\img\owner_overview_1.jpg')
    global sdf
    d = sdf.toPandas()
    model = {'caricato': True,
             'completato': False,
             'users': len(d['owner'].unique())
             }
    if request.method == "POST":
        utenti = request.POST.get('utenti')
        try:
            utenti = int(utenti)
        except ValueError:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        if utenti < 0:
            model['titolo'] = "Ops! Si è verificato un errore, prova a controllare i valori in input alla query..."
            return render(request, 'error.html', model)
        data = main.owner_overview(sdf, n=utenti)
        model['iter1'] = data.iterrows()
        model['iter2'] = data.iterrows()
        model['completato'] = True
    return render(request, 'active_users.html', model)







