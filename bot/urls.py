from django.urls import path
from . import views

urlpatterns = [
    # frontend URLs
    path('train', views.train_model, name='train_model'),
    path('generate_response', views.generate_response, name='generate_response'),
]
