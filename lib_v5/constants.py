import os
import sys
import subprocess
from pathlib import Path
from gui_data.constants import *
from __version__ import VERSION, PATCH, PATCH_MAC, PATCH_LINUX


if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(Path(os.path.abspath(__file__)).parent.parent)

os.chdir(BASE_PATH)  # Change the current working directory to the base path

if OPERATING_SYSTEM=="Darwin":
    OPEN_FILE_func = lambda input_string:subprocess.Popen(["open", input_string])
    dnd_path_check = MAC_DND_CHECK
    banner_placement = -8
    current_patch = PATCH_MAC
    is_windows = False
    is_macos = True
    right_click_button = '<Button-2>'
    application_extension = ".dmg"
elif OPERATING_SYSTEM=="Linux":
    OPEN_FILE_func = lambda input_string:subprocess.Popen(["xdg-open", input_string])
    dnd_path_check = LINUX_DND_CHECK
    current_patch = PATCH_LINUX
    is_windows = False
    is_macos = False
    right_click_button = '<Button-3>'
    application_extension = ".zip"
elif OPERATING_SYSTEM=="Windows":
    OPEN_FILE_func = lambda input_string:os.startfile(input_string)
    dnd_path_check = WINDOWS_DND_CHECK
    current_patch = PATCH
    is_windows = True
    is_macos = False
    right_click_button = '<Button-3>'
    application_extension = ".exe"
    
#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, 'gui_data', 'model_manual_download.json')

#CR Text
CR_TEXT = os.path.join(BASE_PATH, 'gui_data', 'cr_text.txt')

#Style
ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.ico')
if not is_windows:
    MAIN_ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.png')

OWN_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'own_font.json')

MAIN_FONT_NAME = 'Montserrat'
SEC_FONT_NAME = 'Century Gothic'
FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'Montserrat', 'Montserrat.ttf')#
SEC_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'centurygothic', 'GOTHIC.ttf')#
OTHER_FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'other')#

FONT_MAPPER = {MAIN_FONT_NAME:FONT_PATH,
               SEC_FONT_NAME:SEC_FONT_PATH}

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')

DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

MODEL_DATA_URLS = [VR_MODEL_DATA_LINK, MDX_MODEL_DATA_LINK, MDX_MODEL_NAME_DATA_LINK, DEMUCS_MODEL_NAME_DATA_LINK]
MODEL_DATA_FILES = [VR_HASH_JSON, MDX_HASH_JSON, MDX_MODEL_NAME_SELECT, DEMUCS_MODEL_NAME_SELECT]