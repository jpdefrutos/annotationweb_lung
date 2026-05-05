import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from annotationweb.models import Task
from common.task import setup_task_context, save_annotation, NoMoreImages
from .models import BronchoscopyBoundingBox


def process_next_image(request, task_id):
    return process_image(request, task_id, None)


def process_image(request, task_id, image_id):
    try:
        context = setup_task_context(request, task_id, Task.BRONCHOSCOPY_BOUNDING_BOX, image_id)
        context['javascript_files'] = ['bronchoscopy_bounding_box/bronchoscopy_bounding_box.js']
        context['boxes'] = BronchoscopyBoundingBox.objects.filter(image__in=context['frames'])
        return render(request, 'bronchoscopy_bounding_box/process_image.html', context)
    except NoMoreImages:
        return redirect('index')
    except RuntimeError:
        return redirect(request.META.get('HTTP_REFERER', '/'))


def save_boxes(request):
    try:
        annotations = save_annotation(request)
        boxes_data = json.loads(request.POST['boxes'])
        for annotation in annotations:
            frame_nr = str(annotation.frame_nr)
            for box in boxes_data[frame_nr]:
                BronchoscopyBoundingBox.objects.create(
                    image=annotation,
                    x=int(box['x']),
                    y=int(box['y']),
                    width=int(box['width']),
                    height=int(box['height']),
                    label=box.get('label', ''),
                    color=box.get('color', '#e6194b'),
                )
        return JsonResponse({'success': 'true', 'message': 'Completed'})
    except Exception as e:
        return JsonResponse({'success': 'false', 'message': str(e)})
