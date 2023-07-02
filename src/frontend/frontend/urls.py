from django.contrib import admin
from django.urls import path, include
from flickranalytics import views

urlpatterns = [
    path('', views.redirect),
    path('flickranalytics/', views.home),
    path('plot/', views.plot),
    path('placeid/', views.placeid),
    path('postperviews/', views.postperviews),
    path('directions/', views.directions),
    path('kmeans/', views.kmeans),
    path('valutaK/', views.valutaK),
    path('tag/', views.tag),
    path('year/', views.year),
    path('active_users/', views.active_users),
    path('clear/', views.clear),
    path('error/', views.error),
    path('admin/', admin.site.urls),
]
