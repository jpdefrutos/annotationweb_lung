from django.urls import path

from . import views

app_name = 'subsequence_classification'
urlpatterns = [
    path('label-subsequence/<int:task_id>/', views.label_next_image, name='label_subsequence'),
    path('label-subsequence/<int:task_id>/<int:image_id>/', views.label_subsequence, name='label_subsequence'),
    path('save/', views.save_labels, name='save')
]
