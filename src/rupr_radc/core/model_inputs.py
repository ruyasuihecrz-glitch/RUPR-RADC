from PIL import Image

def messages(text, image_path=None):
    content = []
    if image_path is not None:
        with Image.open(image_path) as image:
            content.append({'type': 'image', 'image': image.convert('RGB')})
    content.append({'type': 'text', 'text': text})
    return [{'role': 'user', 'content': content}]
