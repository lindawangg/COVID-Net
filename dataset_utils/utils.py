import requests
import os
import shutil
import ntpath




def download_file_from_google_drive(id, destination,name):
    URL = "https://docs.google.com/uc?export=download"
    # file_id = 'TAKE ID FROM SHAREABLE LINK'
    # destination = 'DESTINATION FILE ON YOUR DISK'
    # Example:
    # file_id = '1aZnIMIUr8dsUjVFzDueA8mqfXWEkHJvn'
    # destination = '/home/hossein/data_hoss/images.zip'

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, os.path.join(destination,name))
    return os.path.join(destination,name)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def remove_sub_dir(destination):
    source=os.path.join(destination,next(os.walk(destination))[1][0])
    files_list = os.listdir(source)
    for files in files_list:
        image_name=ntpath.basename(os.path.join(source,files))
        shutil.move(os.path.join(source,files), os.path.join(destination,image_name))
    os.rmdir(source)
