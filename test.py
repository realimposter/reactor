from modules.processing import StableDiffusionProcessingImg2Img
from scripts.reactor_faceswap import FaceSwapScript
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from PIL import Image
import base64, io

def execute_face_swap(input_image_path, face_image_path, output_path):
    # Load image using PIL
    input_image = [Image.open(input_image_path)]
    face_image = Image.open(face_image_path)
    
    # Convert face_image to base64 for processing
    img_bytes = io.BytesIO()
    face_image.save(img_bytes, format='PNG')
    face_img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    # Face Swap
    script = FaceSwapScript()
    p = StableDiffusionProcessingImg2Img(input_image)
    
    script.process(
        p=p, 
        img=face_img_base64,  # Pass the face image as base64
        enable=True,
        swap_in_source=True,
        swap_in_generated=True,
        source_faces_index="0,1,2,3",
        faces_index="0,1,2,3",
        model="inswapper_128.onnx",  # Replace with your model's name
        gender_source="no",
        gender_target="no"
    )

    # Save Result
    result = p.init_images[0]
    # show result
    result.show()
    # save image
    result.save(output_path)

# Usage Example
# input image, face, output
execute_face_swap("man.jpg", "rock.jpg", "output.jpg")
