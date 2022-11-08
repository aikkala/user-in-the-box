import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen

def check_study_dataset_dir(DIRNAME_STUDY):
    if not os.path.exists(DIRNAME_STUDY):
        download_datasets = input(
            "Could not find reference to the ISO-VR-Pointing Dataset. Do you want to download it (~3.9GB after unpacking)? (y/N) ")
        if download_datasets.lower().startswith("y"):
            print(f"Will download and unzip to '{DIRNAME_STUDY}'.")
            print("Downloading archive... ", end='', flush=True)
            resp = urlopen("https://zenodo.org/record/7300062/files/ISO_VR_Pointing_Dataset.zip?download=1")
            zipfile = ZipFile(BytesIO(resp.read()))
            print("unzip archive... ", end='', flush=True)
            for file in zipfile.namelist():
                if file.startswith('study/'):
                    zipfile.extract(file, path=os.path.dirname(os.path.normpath(DIRNAME_STUDY)) if file.split("/")[0] == os.path.basename(os.path.normpath(DIRNAME_STUDY)) else DIRNAME_STUDY)
            print("done.")
            assert os.path.exists(DIRNAME_STUDY), "Internal Error during unpacking of ISO-VR-Pointing Dataset."
        else:
            raise FileNotFoundError("Please ensure that 'DIRNAME_STUDY' points to a valid directory containing the ISO-VR-Pointing Dataset.")
