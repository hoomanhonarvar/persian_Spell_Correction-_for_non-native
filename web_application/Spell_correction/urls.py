from django.urls import path
from . import views

urlpatterns=[
path( '' , views.sentence_corrector, name="index_page" )
]