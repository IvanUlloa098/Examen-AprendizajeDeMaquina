from django.contrib import admin
from apiCNN.models import Libro
from apiCNN.models import Persona
from apiCNN.models import Image
# Register your models here.
admin.site.register(Libro)
admin.site.register(Persona)
admin.site.register(Image)