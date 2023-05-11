from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="main"),
    path("about-us", views.about, name="info1"),
    path("modelstepa", views.modelkozhcon, name="modelstepa"),
    path("modelfridman1", views.modelfridman1, name="modelfridman1"),
    path("modelfridman2", views.modelfridman2, name="modelfridman2"),
]
