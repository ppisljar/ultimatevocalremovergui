import time
import os
import shutil
from pathlib  import Path
from datetime import datetime
from gui_data.constants import *
from gui_data.app_size_values import *
from gui_data.error_handling import error_text, error_dialouge

from UVR import Ensembler

from separate import (
    SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR,  # Model-related
    save_format, clear_gpu_cache,  # Utility functions
    cuda_available, mps_available, #directml_available,
)

def extract_stems(audio_file_base, export_path):
    
    filenames = [file for file in os.listdir(export_path) if file.startswith(audio_file_base)]

    pattern = r'\(([^()]+)\)(?=[^()]*\.wav)'
    stem_list = []

    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            stem_list.append(match.group(1))
            
    counter = Counter(stem_list)
    filtered_lst = [item for item in stem_list if counter[item] > 1]

    return list(set(filtered_lst))

class Processor():
    true_model_count = 0
    iteration = 0
    inputPaths = []
    all_models = []
    export_path_var = "./"
    model_sample_mode_var = False
    chosen_process_method_var = ENSEMBLE_MODE
    is_secondary_stem_only_var = False
    is_primary_stem_only_var = False
    is_add_model_name_var = True
    is_testing_audio_var = False
    is_create_model_folder_var = True
    ensemble_main_stem_var = ?
    demucs_model_var = ?
    mdx_net_model_var = ?
    vr_model_var = ?
    process_iteration = 0

    def assemble_model_data(self, model=None, arch_type=ENSEMBLE_MODE, is_dry_check=False, is_change_def=False, is_get_hash_dir_only=False):

        if arch_type == ENSEMBLE_STEM_CHECK:
            
            model_data = self.model_data_table
            missing_models = [model.model_status for model in model_data if not model.model_status]
            
            if missing_models or not model_data:
                model_data: List[ModelData] = [ModelData(model_name, is_dry_check=is_dry_check) for model_name in self.ensemble_model_list]
                self.model_data_table = model_data

        if arch_type == KARAOKEE_CHECK:
            model_list = []
            model_data: List[ModelData] = [ModelData(model_name, is_dry_check=is_dry_check) for model_name in self.default_change_model_list]
            for model in model_data:
                if model.model_status and model.is_karaoke or model.is_bv_model:
                    model_list.append(model.model_and_process_tag)
            
            return model_list

        if arch_type == ENSEMBLE_MODE:
            model_data: List[ModelData] = [ModelData(model_name) for model_name in self.ensemble_listbox_get_all_selected_models()]
        if arch_type == ENSEMBLE_CHECK:
            model_data: List[ModelData] = [ModelData(model, is_change_def=is_change_def, is_get_hash_dir_only=is_get_hash_dir_only)]
        if arch_type == VR_ARCH_TYPE or arch_type == VR_ARCH_PM:
            model_data: List[ModelData] = [ModelData(model, VR_ARCH_TYPE)]
        if arch_type == MDX_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, MDX_ARCH_TYPE)]
        if arch_type == DEMUCS_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)]#

        return model_data
    
    def cached_source_model_list_check(self, model_list: List[ModelData]):

        model: ModelData
        primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
        secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

        self.vr_primary_model_names = primary_model_names(VR_ARCH_TYPE)
        self.mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
        self.demucs_primary_model_names = primary_model_names(DEMUCS_ARCH_TYPE)
        self.vr_secondary_model_names = secondary_model_names(VR_ARCH_TYPE)
        self.mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)
        self.demucs_secondary_model_names = [model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == DEMUCS_ARCH_TYPE and not model.secondary_model is None else None for model in model_list]
        self.demucs_pre_proc_model_name = [model.pre_proc_model.model_basename if model.pre_proc_model else None for model in model_list]#list(dict.fromkeys())
        
        for model in model_list:
            if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
                if not model.is_4_stem_ensemble:
                    self.demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                    break
                else:
                    for i in model.secondary_model_4_stem_model_names_list:
                        self.demucs_secondary_model_names.append(i)
        
        self.all_models = self.vr_primary_model_names + self.mdx_primary_model_names + self.demucs_primary_model_names + self.vr_secondary_model_names + self.mdx_secondary_model_names + self.demucs_secondary_model_names + self.demucs_pre_proc_model_name
      
    def verify_audio(self, audio_file, is_process=True, sample_path=None):
        is_good = False
        error_data = ''
        
        if not type(audio_file) is tuple:
            audio_file = [audio_file]

        for i in audio_file:
            if os.path.isfile(i):
                try:
                    librosa.load(i, duration=3, mono=False, sr=44100) if not type(sample_path) is str else self.create_sample(i, sample_path)
                    is_good = True
                except Exception as e:
                    error_name = f'{type(e).__name__}'
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    message = f'{error_name}: "{e}"\n{traceback_text}"'
                    if is_process:
                        audio_base_name = os.path.basename(i)
                        self.error_log_var.set(f'{ERROR_LOADING_FILE_TEXT[0]}:\n\n\"{audio_base_name}\"\n\n{ERROR_LOADING_FILE_TEXT[1]}:\n\n{message}')
                    else:
                        error_data = AUDIO_VERIFICATION_CHECK(i, message)

        if is_process:
            return is_good
        else:
            return is_good, error_data
        
    def process_get_baseText(self, total_files, file_num, is_dual=False):
        """Create the base text for the command widget"""
        
        init_text = 'Files' if is_dual else 'File'
        
        text = '{init_text} {file_num}/{total_files} '.format(init_text=init_text,
                                                              file_num=file_num,
                                                              total_files=total_files)
        
        return text
    
    def cached_sources_clear(self):

        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}

    def cached_source_callback(self, process_method, model_name=None):
        
        model, sources = None, None
        
        if process_method == VR_ARCH_TYPE:
            mapper = self.vr_cache_source_mapper
        if process_method == MDX_ARCH_TYPE:
            mapper = self.mdx_cache_source_mapper
        if process_method == DEMUCS_ARCH_TYPE:
            mapper = self.demucs_cache_source_mapper
        
        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value
        
        return model, sources

    def cached_model_source_holder(self, process_method, sources, model_name=None):
        
        if process_method == VR_ARCH_TYPE:
            self.vr_cache_source_mapper = {**self.vr_cache_source_mapper, **{model_name: sources}}
        if process_method == MDX_ARCH_TYPE:
            self.mdx_cache_source_mapper = {**self.mdx_cache_source_mapper, **{model_name: sources}}
        if process_method == DEMUCS_ARCH_TYPE:
            self.demucs_cache_source_mapper = {**self.demucs_cache_source_mapper, **{model_name: sources}}

    def create_sample(self, audio_file, sample_path=SAMPLE_CLIP_PATH):
        try:
            with audioread.audio_open(audio_file) as f:
                track_length = int(f.duration)
        except Exception as e:
            print('Audioread failed to get duration. Trying Librosa...')
            y, sr = librosa.load(audio_file, mono=False, sr=44100)
            track_length = int(librosa.get_duration(y=y, sr=sr))
        
        clip_duration = int(self.model_sample_mode_duration_var.get())
        
        if track_length >= clip_duration:
            offset_cut = track_length//3
            off_cut = offset_cut + track_length
            if not off_cut >= clip_duration:
                offset_cut = 0
            name_apped = f'{clip_duration}_second_'
        else:
            offset_cut, clip_duration = 0, track_length
            name_apped = ''

        sample = librosa.load(audio_file, offset=offset_cut, duration=clip_duration, mono=False, sr=44100)[0].T
        audio_sample = os.path.join(sample_path, f'{os.path.splitext(os.path.basename(audio_file))[0]}_{name_apped}sample.wav')
        sf.write(audio_sample, sample, 44100)
        
        return audio_sample

    def process_start(self):
        """Start the conversion for all the given mp3 and wav files"""
        
        stime = time.perf_counter()
        time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'
        export_path = self.export_path_var
        is_ensemble = False
        self.true_model_count = 0
        self.iteration = 0
        is_verified_audio = True
        inputPaths = self.inputPaths
        inputPath_total_len = len(inputPaths)
        is_model_sample_mode = self.model_sample_mode_var
        
        
        
        
        try:
            if self.chosen_process_method_var.get() == ENSEMBLE_MODE:
                model, ensemble = self.assemble_model_data(), Ensembler()
                export_path, is_ensemble = ensemble.ensemble_folder_name, True
            if self.chosen_process_method_var.get() == VR_ARCH_PM:
                model = self.assemble_model_data(self.vr_model_var.get(), VR_ARCH_TYPE)
            if self.chosen_process_method_var.get() == MDX_ARCH_TYPE:
                model = self.assemble_model_data(self.mdx_net_model_var.get(), MDX_ARCH_TYPE)
            if self.chosen_process_method_var.get() == DEMUCS_ARCH_TYPE:
                model = self.assemble_model_data(self.demucs_model_var.get(), DEMUCS_ARCH_TYPE)

            self.cached_source_model_list_check(model)
            
            true_model_4_stem_count = sum(m.demucs_4_stem_added_count if m.process_method == DEMUCS_ARCH_TYPE else 0 for m in model)
            true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
            self.true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count + self.determine_voc_split(model)

            #print("self.true_model_count", self.true_model_count)

            for file_num, audio_file in enumerate(inputPaths, start=1):
                self.cached_sources_clear()
                base_text = self.process_get_baseText(total_files=inputPath_total_len, file_num=file_num)

                if self.verify_audio(audio_file):
                    audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file
                    print(f'{NEW_LINE if not file_num ==1 else NO_LINE}{base_text}"{os.path.basename(audio_file)}\".{NEW_LINES}')
                    is_verified_audio = True
                else:
                    error_text_console = f'{base_text}"{os.path.basename(audio_file)}\" {MISSING_MESS_TEXT}\n'
                    print(f'\n{error_text_console}') if inputPath_total_len >= 2 else None
                    self.iteration += self.true_model_count
                    is_verified_audio = False
                    continue

                for current_model_num, current_model in enumerate(model, start=1):
                    self.iteration += 1

                    if is_ensemble:
                        print(f'Ensemble Mode - {current_model.model_basename} - Model {current_model_num}/{len(model)}{NEW_LINES}')

                    model_name_text = f'({current_model.model_basename})' if not is_ensemble else ''
                    print(base_text + f'{LOADING_MODEL_TEXT} {model_name_text}...')

                    set_progress_bar = lambda step, inference_iterations=0:self.process_update_progress(total_files=inputPath_total_len, step=(step + (inference_iterations)))
                    write_to_console = lambda progress_text, base_text=base_text:print(base_text + progress_text)

                    audio_file_base = f"{file_num}_{os.path.splitext(os.path.basename(audio_file))[0]}"
                    audio_file_base = audio_file_base if not self.is_testing_audio_var or is_ensemble else f"{round(time.time())}_{audio_file_base}"
                    audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{current_model.model_basename}"
                    if not is_ensemble:
                        audio_file_base = audio_file_base if not self.is_add_model_name_var else f"{audio_file_base}_{current_model.model_basename}"

                    if self.is_create_model_folder_var and not is_ensemble:
                        export_path = os.path.join(Path(self.export_path_var), current_model.model_basename, os.path.splitext(os.path.basename(audio_file))[0])
                        if not os.path.isdir(export_path):os.makedirs(export_path) 

                    process_data = {
                                    'model_data': current_model, 
                                    'export_path': export_path,
                                    'audio_file_base': audio_file_base,
                                    'audio_file': audio_file,
                                    'set_progress_bar': set_progress_bar,
                                    'write_to_console': write_to_console,
                                    'process_iteration': self.process_iteration,
                                    'cached_source_callback': self.cached_source_callback,
                                    'cached_model_source_holder': self.cached_model_source_holder,
                                    'list_all_models': self.all_models,
                                    'is_ensemble_master': is_ensemble,
                                    'is_4_stem_ensemble': True if self.ensemble_main_stem_var in [FOUR_STEM_ENSEMBLE, MULTI_STEM_ENSEMBLE] and is_ensemble else False}
                    
                    if current_model.process_method == VR_ARCH_TYPE:
                        seperator = SeperateVR(current_model, process_data)
                    if current_model.process_method == MDX_ARCH_TYPE:
                        seperator = SeperateMDXC(current_model, process_data) if current_model.is_mdx_c else SeperateMDX(current_model, process_data)
                    if current_model.process_method == DEMUCS_ARCH_TYPE:
                        seperator = SeperateDemucs(current_model, process_data)
                        
                    seperator.seperate()
                    
                    if is_ensemble:
                        print('\n')

                if is_ensemble:
                    
                    audio_file_base = audio_file_base.replace(f"_{current_model.model_basename}","")
                    print(base_text + ENSEMBLING_OUTPUTS)
                    
                    if self.ensemble_main_stem_var in [FOUR_STEM_ENSEMBLE, MULTI_STEM_ENSEMBLE]:
                        stem_list = extract_stems(audio_file_base, export_path)
                        for output_stem in stem_list:
                            ensemble.ensemble_outputs(audio_file_base, export_path, output_stem, is_4_stem=True)
                    else:
                        if not self.is_secondary_stem_only_var:
                            ensemble.ensemble_outputs(audio_file_base, export_path, PRIMARY_STEM)
                        if not self.is_primary_stem_only_var:
                            ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM)
                            ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM, is_inst_mix=True)

                    print(DONE)
                    
                if is_model_sample_mode:
                    if os.path.isfile(audio_file):
                        os.remove(audio_file)
                    
                clear_gpu_cache()
                
            shutil.rmtree(export_path) if is_ensemble and len(os.listdir(export_path)) == 0 else None

            if inputPath_total_len == 1 and not is_verified_audio:
                print(f'{error_text_console}\n{PROCESS_FAILED}')
                print(time_elapsed())
                print("done")
            else:
                set_progress_bar(1.0)
                print(PROCESS_COMPLETE)
                print(time_elapsed())
                print("done")
                
                        
        except Exception as e:
            print(f'\n\n{PROCESS_FAILED}')
            print(time_elapsed())
            print("failed")
