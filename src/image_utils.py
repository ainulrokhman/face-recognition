import cv2
import numpy as np
import storage


def parse_image_from_request(request):
    """Parse image from either multipart/form-data or JSON base64 request.
    
    Supports dual input:
    - multipart/form-data: binary file in 'image' field (saves ~33% bandwidth)
    - application/json: base64 string in 'frame' field (backward compatible)
    
    Returns:
        tuple: (image_ndarray, extra_fields_dict, error_message)
        - On success: (np.ndarray, dict, None)
        - On error: (None, None, str)
    """
    content_type = request.content_type or ""
    
    if "multipart/form-data" in content_type:
        # Binary file upload path — no base64 overhead
        if "image" not in request.files:
            return None, None, "Missing 'image' file in multipart upload"
        
        file = request.files["image"]
        if file.filename == "":
            return None, None, "Empty filename in uploaded file"
        
        try:
            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return None, None, "Failed to decode uploaded image file"
        except Exception as e:
            return None, None, f"Error reading uploaded file: {e}"
        
        # Collect extra form fields for register endpoint
        extra_fields = {}
        for key in ("label_id", "username", "target_pose"):
            if key in request.form:
                extra_fields[key] = request.form[key]
        
        return img, extra_fields, None
    
    else:
        # JSON base64 path — backward compatible
        data = request.get_json()
        if not data or "frame" not in data:
            return None, None, "Missing 'frame' parameter in JSON body"
        
        img = storage.decode_base64_image(data["frame"])
        if img is None:
            return None, None, "Failed to decode base64 frame"
        
        # Pass through all JSON fields as extra_fields
        extra_fields = {k: v for k, v in data.items() if k != "frame"}
        
        return img, extra_fields, None
