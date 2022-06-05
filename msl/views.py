from django.shortcuts import render, redirect
import numpy as np
import cv2
from django.http.response import StreamingHttpResponse

from .classVideo_agile import *

# Create your views here.


def gen_(camera):
    while True:
        frame = camera.get_frame()
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed_(request):
    return StreamingHttpResponse(gen_(VideoCamera_()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
def video_feed_2(request,name_):
    cv2.VideoCapture(0,cv2.CAP_DSHOW).release()
    return render(request, 'test_agile.html')
def test_agile(request):
    print("fff")
    return render(request, 'test_agile.html')
    