from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import predict_sign
import os
import threading

# Global variable to store prediction result
prediction_result = None

def predict_async(image_path):
    global prediction_result
    try:
        prediction_result = predict_sign(image_path)
    except Exception as e:
        prediction_result = f"Error: {str(e)}"

@csrf_exempt
def index(request):
    global prediction_result
    
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image uploaded'}, status=400)
        
        image_file = request.FILES['image']

        # Save the uploaded file to media directory
        file_name = default_storage.save('uploads/' + image_file.name, ContentFile(image_file.read()))
        file_path = default_storage.path(file_name)
        image_url = default_storage.url(file_name)
        
        try:
            # Start prediction in a separate thread for better responsiveness
            prediction_thread = threading.Thread(target=predict_async, args=(file_path,))
            prediction_thread.start()
            
            # Wait for prediction to complete (with timeout)
            prediction_thread.join(timeout=10)  # 10 second timeout
            
            if prediction_thread.is_alive():
                # If still running, return processing status
                return JsonResponse({'status': 'processing', 'message': 'Prediction in progress...'})
            
            sign = prediction_result
            prediction_result = None  # Reset for next request

            return JsonResponse({'sign': sign, 'image_url': image_url})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        finally:
            # Note: Keeping the uploaded file in media directory for display
            pass
    
    return render(request, 'index.html')
