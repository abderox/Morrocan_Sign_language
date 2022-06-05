from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from msl.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('test_agile', test_agile , name="test_agile"),
    path('video_feed_', video_feed_ , name="video_feed_"),
    path('video_feed_2/<str:name_>', video_feed_2 , name="video_feed_"),
    


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


