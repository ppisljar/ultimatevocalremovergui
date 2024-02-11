import time
import os
import shutil
import yaml
import json
from re import match
from lib_v5.utils import load_model_hash_data
import librosa
import traceback
import audioread
import soundfile as sf
import hashlib
from collections import Counter
from ml_collections import ConfigDict

from typing import List
from pathlib  import Path
from datetime import datetime
from gui_data.constants import *
from gui_data.app_size_values import *
from gui_data.error_handling import error_text, error_dialouge

from lib_v5.constants import *
from lib_v5 import spec_utils
from lib_v5.utils import *
from lib_v5.vr_network.model_param_init import ModelParameters

from separate import (
    SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR,  # Model-related
    save_format, clear_gpu_cache,  # Utility functions
    cuda_available, mps_available, #directml_available,
)

model_hash_table = {}

class Ensembler():
    def __init__(self, is_manual_ensemble=False, is_save_all_outputs_ensemble=False, chosen_ensemble='Ensembled',
                 ensemble_type='Type/Algo', ensemble_main_stem='MainStem/SubStem', export_path='export/path',
                 is_append_ensemble_name=False, is_testing_audio=False, is_normalization=False,
                 is_wav_ensemble=False, wav_type_set='default', mp3_bit_set=128, save_format='wav',
                 choose_algorithm='default'):
        self.is_save_all_outputs_ensemble = is_save_all_outputs_ensemble
        chosen_ensemble_name = '{}'.format(chosen_ensemble.replace(" ", "_")) if chosen_ensemble != 'CHOOSE_ENSEMBLE_OPTION' else 'Ensembled'
        ensemble_algorithm = ensemble_type.partition("/")
        self.ensemble_type = ensemble_type
        ensemble_main_stem_pair = ensemble_main_stem.partition("/")
        time_stamp = round(time.time())
        self.audio_tool = 'MANUAL_ENSEMBLE'
        self.main_export_path = Path(export_path)
        self.chosen_ensemble = f"_{chosen_ensemble_name}" if is_append_ensemble_name else ''
        ensemble_folder_name = self.main_export_path if self.is_save_all_outputs_ensemble else 'ENSEMBLE_TEMP_PATH'
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, '{}_Outputs_{}'.format(chosen_ensemble_name, time_stamp))
        self.is_testing_audio = f"{time_stamp}_" if is_testing_audio else ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[2]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[2]
        self.is_normalization = is_normalization
        self.is_wav_ensemble = is_wav_ensemble
        self.wav_type_set = wav_type_set
        self.mp3_bit_set = mp3_bit_set
        self.save_format = save_format
        self.choose_algorithm = choose_algorithm
        if not is_manual_ensemble:
            os.mkdir(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        if is_4_stem:
            algorithm = self.ensemble_type
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} {INST_STEM}"
            else:
                algorithm = self.primary_algorithm if stem == PRIMARY_STEM else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == PRIMARY_STEM else self.ensemble_secondary_stem

        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}_({stem_tag})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}.wav'.format(audio_file_output))
        
        #print("get_files_to_ensemble: ", stem_outputs)
        
        if len(stem_outputs) > 1:
            spec_utils.ensemble_inputs(stem_outputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path, is_wave=self.is_wav_ensemble)
            save_format(stem_save_path, self.save_format, self.mp3_bit_set)
        
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                save_format(i, self.save_format, self.mp3_bit_set)
        else:
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(e)

    def ensemble_manual(self, audio_inputs, audio_file_base, is_bulk=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        is_mv_sep = True
        
        if is_bulk:
            number_list = list(set([os.path.basename(i).split("_")[0] for i in audio_inputs]))
            for n in number_list:
                current_list = [i for i in audio_inputs if os.path.basename(i).startswith(n)]
                audio_file_base = os.path.basename(current_list[0]).split('.wav')[0]
                stem_testing = "instrum" if "Instrumental" in audio_file_base else "vocals"
                if is_mv_sep:
                    audio_file_base = audio_file_base.split("_")
                    audio_file_base = f"{audio_file_base[1]}_{audio_file_base[2]}_{stem_testing}"
                self.ensemble_manual_process(current_list, audio_file_base, is_bulk)
        else:
            self.ensemble_manual_process(audio_inputs, audio_file_base, is_bulk)
            
    def ensemble_manual_process(self, audio_inputs, audio_file_base, is_bulk):
        
        algorithm = self.choose_algorithm
        algorithm_text = "" if is_bulk else f"_({self.choose_algorithm})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}{}{}.wav'.format(self.is_testing_audio, audio_file_base, algorithm_text))
        spec_utils.ensemble_inputs(audio_inputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path, is_wave=self.is_wav_ensemble)
        save_format(stem_save_path, self.save_format, self.mp3_bit_set)

    def get_files_to_ensemble(self, folder="", prefix="", suffix=""):
        """Grab all the files to be ensembled"""
        
        return [os.path.join(folder, i) for i in os.listdir(folder) if i.startswith(prefix) and i.endswith(suffix)]

    def combine_audio(self, audio_inputs, audio_file_base):
        save_format_ = lambda save_path:save_format(save_path, self.save_format, self.mp3_bit_set)
        spec_utils.combine_audio(audio_inputs, 
                                 os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}"), 
                                 self.wav_type_set,
                                 save_format=save_format_)

class ModelData():
    def __init__(self, model_name: str, 
                 config: dict,
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False, 
                 ):
        # Basic configuration
        self.model_name = config.get('model_name')
        self.selected_process_method = config.get('selected_process_method')
        self.is_secondary_model = config.get('is_secondary_model', False)
        self.primary_model_primary_stem = config.get('primary_model_primary_stem')
        self.is_primary_model_primary_stem_only = config.get('is_primary_model_primary_stem_only', False)
        self.is_primary_model_secondary_stem_only = config.get('is_primary_model_secondary_stem_only', False)
        self.is_pre_proc_model = config.get('is_pre_proc_model', False)
        self.is_dry_check = config.get('is_dry_check', False)
        self.is_change_def = config.get('is_change_def', False)
        self.is_get_hash_dir_only = config.get('is_get_hash_dir_only', False)
        self.is_vocal_split_model = config.get('is_vocal_split_model', False)

        # Device and model paths
        self.device_set = config.get('device_set_var', 'cpu')
        self.DENOISER_MODEL = config.get('DENOISER_MODEL_PATH')
        self.DEVERBER_MODEL = config.get('DEVERBER_MODEL_PATH')

        # Feature toggles
        self.is_deverb_vocals = config.get('is_deverb_vocals_var', False)
        self.deverb_vocal_opt = config.get('deverb_vocal_opt_var', 'default')
        self.is_denoise_model = config.get('is_denoise_model', False)
        self.is_gpu_conversion = config.get('is_gpu_conversion_var', 0)
        self.is_normalization = config.get('is_normalization_var', False)
        self.is_use_opencl = config.get('is_use_opencl_var', False)

        # Processing parameters
        self.is_primary_stem_only = config.get('is_primary_stem_only_var', False)
        self.is_secondary_stem_only = config.get('is_secondary_stem_only_var', False)
        self.mdx_batch_size = config.get('mdx_batch_size_var', 1)
        self.overlap = config.get('overlap_var', 0.25)
        self.semitone_shift = config.get('semitone_shift_var', 0.0)
        self.is_match_frequency_pitch = config.get('is_match_frequency_pitch_var', False)
        self.wav_type_set = config.get('wav_type_set', 'default')
        self.mp3_bit_set = config.get('mp3_bit_set_var', 128)
        self.save_format = config.get('save_format_var', 'wav')

        # Advanced model configuration
        self.mdxnet_stem_select = config.get('mdxnet_stems_var', 'default')
        self.overlap_mdx = config.get('overlap_mdx_var', 0.25)
        self.overlap_mdx23 = config.get('overlap_mdx23_var', 0)
        self.is_pitch_change = self.semitone_shift != 0

        # Ensemble and output settings
        self.is_mdx_c_seg_def = config.get('is_mdx_c_seg_def_var', False)
        self.is_mdx_ckpt = config.get('is_mdx_ckpt', False)
        self.is_mdx_c = config.get('is_mdx_c', False)
        self.is_mdx_combine_stems = config.get('is_mdx23_combine_stems_var', False)
        self.is_invert_spec = config.get('is_invert_spec_var', False)
        self.is_mixer_mode = config.get('is_mixer_mode', False)

        # Additional configurations not explicitly listed in your initial method but implied by the usage of `root` variables
        # For example, handling of mappings, model paths, or specific processing flags not detailed in the provided snippet.
        # self.custom_config = config.get('custom_config_key', 'default_value')

        # Placeholder for methods or functionalities implied by variables like mappings or model-specific settings
        # self.setup_custom_behavior()

        if selected_process_method == ENSEMBLE_MODE:
            self.process_method, _, self.model_name = model_name.partition(ENSEMBLE_PARTITION)
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = self.return_ensemble_stems()
            
            is_not_secondary_or_pre_proc = not is_secondary_model and not is_pre_proc_model
            self.is_ensemble_mode = is_not_secondary_or_pre_proc
            
            if self.ensemble_main_stem == FOUR_STEM_ENSEMBLE:
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif self.ensemble_main_stem == MULTI_STEM_ENSEMBLE and self.chosen_process_method == ENSEMBLE_MODE:
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != VOCAL_STEM
            self.pre_proc_model_activated = self.is_demucs_pre_proc_model_activate if is_not_vocal_stem else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = self.vr_is_secondary_model_activate if not is_secondary_model else False
            self.aggression_setting = float(int(self.aggression_setting)/100)
            self.is_tta = self.is_tta
            self.is_post_process = self.is_post_process
            self.window_size = int(self.window_size)
            self.batch_size = 1 if self.batch_size == DEF_OPT else int(self.batch_size)
            self.crop_size = int(self.crop_size)
            self.is_high_end_process = 'mirroring' if self.is_high_end_process else 'None'
            self.post_process_threshold = float(self.post_process_threshold)
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(VR_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(VR_HASH_DIR, self.vr_hash_MAPPER) if not self.model_hash == WOOD_INST_MODEL_HASH else WOOD_INST_PARAMS
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = secondary_stem(self.primary_stem)
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    self.primary_stem_native = self.primary_stem
                    if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                    self.check_if_karaokee_model()
   
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = self.mdx_is_secondary_model_activate if not is_secondary_model else False
            self.margin = int(self.margin)
            self.chunks = 0
            self.mdx_segment_size = int(self.mdx_segment_size)
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR, self.mdx_hash_MAPPER)
                if self.model_data:
                    
                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

                            self.mdx_c_configs = config
                                
                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem and set 4-stem ensemble to False
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments in the training config
                                self.mdx_model_stems = self.mdx_c_configs.training.instruments
                                self.mdx_stem_count = len(self.mdx_model_stems)
                                
                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select
                                
                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"] if self.compensate == AUTO_SELECT else float(self.compensate)
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()
                        
                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = self.demucs_is_secondary_model_activate if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = self.is_demucs_pre_proc_model_activate if not self.demucs_stems in [VOCAL_STEM, INST_STEM] else False
            self.margin_demucs = int(self.margin_demucs)
            self.chunks_demucs = 0
            self.shifts = int(self.shifts)
            self.is_split_mode = self.is_split_mode
            self.segment = self.segment
            self.is_chunk_demucs = self.is_chunk_demucs
            self.is_primary_stem_only = self.is_primary_stem_only if self.is_ensemble_mode else self.is_primary_stem_only_Demucs
            self.is_secondary_stem_only = self.is_secondary_stem_only if self.is_ensemble_mode else self.is_secondary_stem_only_Demucs
            self.get_demucs_model_data()
            self.get_demucs_model_path()
            
        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None
            
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = self.demucs_stems == ALL_STEMS
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0
                
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model = self.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = self.is_demucs_pre_proc_model_inst_mix if self.pre_proc_model else False

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = BV_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else LEAD_VOCAL_STEM
            else:
                primary = LEAD_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else BV_VOCAL_STEM
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)
            
        self.vocal_splitter_model_data()

    def process_determine_demucs_pre_proc_model(self, primary_stem=None):
        """Obtains the correct pre-process secondary model data for conversion."""
        
        # Check if a pre-process model is set and it's not the 'NO_MODEL' value
        if self.demucs_pre_proc_model != NO_MODEL and self.is_demucs_pre_proc_model_activate:
            pre_proc_model = ModelData(self.demucs_pre_proc_model, 
                                        primary_model_primary_stem=primary_stem, 
                                        is_pre_proc_model=True)
            
            # Return the model if it's valid
            if pre_proc_model.model_status:
                return pre_proc_model
                
        return None
            
    def return_ensemble_stems(self, is_primary=False): 
        """Grabs and returns the chosen ensemble stems."""
        
        ensemble_stem = self.ensemble_main_stem.partition("/")
        
        if is_primary:
            return ensemble_stem[0]
        else:
            return ensemble_stem[0], ensemble_stem[2]
        
    def process_determine_vocal_split_model(self):
        """Obtains the correct vocal splitter secondary model data for conversion."""
        
        # Check if a vocal splitter model is set and if it's not the 'NO_MODEL' value
        if self.set_vocal_splitter != NO_MODEL and self.is_set_vocal_splitter:
            vocal_splitter_model = ModelData(self.set_vocal_splitter, is_vocal_split_model=True)
            
            # Return the model if it's valid
            if vocal_splitter_model.model_status:
                return vocal_splitter_model
                
        return None
    
    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = self.process_determine_vocal_split_model()
            self.is_vocal_split_model_activated = True if self.vocal_split_model else False
            
            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True

    def process_determine_secondary_model(self, process_method, main_model_primary_stem, is_primary_stem_only=False, is_secondary_stem_only=False):
        """Obtains the correct secondary model data for conversion."""
        
        secondary_model_scale = None
        secondary_model = NO_MODEL
        
        if process_method == VR_ARCH_TYPE:
            secondary_model_vars = self.vr_secondary_model_vars
        if process_method == MDX_ARCH_TYPE:
            secondary_model_vars = self.mdx_secondary_model_vars
        if process_method == DEMUCS_ARCH_TYPE:
            secondary_model_vars = self.demucs_secondary_model_vars

        if main_model_primary_stem in [VOCAL_STEM, INST_STEM]:
            secondary_model = secondary_model_vars["voc_inst_secondary_model"]
            secondary_model_scale = secondary_model_vars["voc_inst_secondary_model_scale"].get()
        if main_model_primary_stem in [OTHER_STEM, NO_OTHER_STEM]:
            secondary_model = secondary_model_vars["other_secondary_model"]
            secondary_model_scale = secondary_model_vars["other_secondary_model_scale"].get()
        if main_model_primary_stem in [DRUM_STEM, NO_DRUM_STEM]:
            secondary_model = secondary_model_vars["drums_secondary_model"]
            secondary_model_scale = secondary_model_vars["drums_secondary_model_scale"].get()
        if main_model_primary_stem in [BASS_STEM, NO_BASS_STEM]:
            secondary_model = secondary_model_vars["bass_secondary_model"]
            secondary_model_scale = secondary_model_vars["bass_secondary_model_scale"].get()

        if secondary_model_scale:
           secondary_model_scale = float(secondary_model_scale)

        if not secondary_model.get() == NO_MODEL:
            secondary_model = ModelData(secondary_model.get(), 
                                        is_secondary_model=True, 
                                        primary_model_primary_stem=main_model_primary_stem, 
                                        is_primary_model_primary_stem_only=is_primary_stem_only, 
                                        is_primary_model_secondary_stem_only=is_secondary_stem_only)
            if not secondary_model.model_status:
                secondary_model = None
        else:
            secondary_model = None
            
        return secondary_model, secondary_model_scale
            
    def secondary_model_data(self, primary_stem):
        secondary_model_data = self.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
            
        #print("self.is_secondary_model_activated: ", self.is_secondary_model_activated)
              
    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]#
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]#
   
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in self.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = self.demucs_version in {DEMUCS_V3, DEMUCS_V4}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in self.demucs_name_select_MAPPER.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        if DEMUCS_UVR_MODEL in self.model_name:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
        else:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4

        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = secondary_stem(self.primary_stem)
            
    def get_model_data(self, model_hash_dir, hash_mapper:dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

            return self.get_model_data_from_popup()

    def change_model_data(self):
        if self.is_get_hash_dir_only:
            return None
        else:
            return self.get_model_data_from_popup()

    # def get_model_data_from_popup(self):
    #     if self.is_dry_check:
    #         return None
            
    #     if not self.is_change_def:
    #         confirm = messagebox.askyesno(
    #             title=UNRECOGNIZED_MODEL[0],
    #             message=f'"{self.model_name}"{UNRECOGNIZED_MODEL[1]}',
    #             parent=root
    #         )
    #         if not confirm:
    #             return None
        
    #     if self.process_method == VR_ARCH_TYPE:
    #         root.pop_up_vr_param(self.model_hash)
    #         return root.vr_model_params
    #     elif self.process_method == MDX_ARCH_TYPE:
    #         root.pop_up_mdx_model(self.model_hash, self.model_path)
    #         return root.mdx_model_params

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)
                
        #print(self.model_name," - ", self.model_hash)
                
class AudioTools():
    def __init__(self, audio_tool, export_path, wav_type_set, is_normalization, is_testing_audio,
                 save_format, mp3_bit_set, time_window, intro_analysis, db_analysis,
                 is_save_align, is_match_silence, is_spec_match, pitch_rate, is_time_correction,
                 time_stretch_rate, phase_option, phase_shifts):
        time_stamp = round(time.time())
        self.audio_tool = audio_tool
        self.main_export_path = Path(export_path)
        self.wav_type_set = wav_type_set
        self.is_normalization = is_normalization
        self.is_testing_audio = f"{time_stamp}_" if is_testing_audio else ''
        # Adjust the lambda function to directly use parameters
        self.save_format = lambda save_path:save_format(save_path, save_format, mp3_bit_set)
        # Use direct mapping from parameters instead of root
        self.align_window = TIME_WINDOW_MAPPER[time_window]
        self.align_intro_val = INTRO_MAPPER[intro_analysis]
        self.db_analysis_val = VOLUME_MAPPER[db_analysis]
        self.is_save_align = is_save_align
        self.is_match_silence = is_match_silence
        self.is_spec_match = is_spec_match
        self.pitch_rate = pitch_rate
        self.is_time_correction = is_time_correction
        self.time_stretch_rate = time_stretch_rate
        
        self.phase_option = phase_option
        self.phase_shifts = PHASE_SHIFTS_OPT[phase_shifts]
        
    def align_inputs(self, audio_inputs, audio_file_base, audio_file_2_base, command_Text, set_progress_bar):
        audio_file_base = f"{self.is_testing_audio}{audio_file_base}"
        audio_file_2_base = f"{self.is_testing_audio}{audio_file_2_base}"
        
        aligned_path = os.path.join('{}'.format(self.main_export_path),'{}_(Aligned).wav'.format(audio_file_2_base))
        inverted_path = os.path.join('{}'.format(self.main_export_path),'{}_(Inverted).wav'.format(audio_file_base))

        spec_utils.align_audio(audio_inputs[0], 
                               audio_inputs[1], 
                               aligned_path, 
                               inverted_path, 
                               self.wav_type_set, 
                               self.is_save_align, 
                               command_Text, 
                               self.save_format,
                               align_window=self.align_window,
                               align_intro_val=self.align_intro_val,
                               db_analysis=self.db_analysis_val,
                               set_progress_bar=set_progress_bar, 
                               phase_option=self.phase_option,
                               phase_shifts=self.phase_shifts,
                               is_match_silence=self.is_match_silence,
                               is_spec_match=self.is_spec_match)
        
    def match_inputs(self, audio_inputs, audio_file_base, command_Text):
        
        target = audio_inputs[0]
        reference = audio_inputs[1]
        
        command_Text(f"Processing... ")
        
        save_path = os.path.join('{}'.format(self.main_export_path),'{}_(Matched).wav'.format(f"{self.is_testing_audio}{audio_file_base}"))
        
        match.process(
            target=target,
            reference=reference,
            results=[match.save_audiofile(save_path, wav_set=self.wav_type_set),
            ],
        )
        
        self.save_format(save_path)
        
    def combine_audio(self, audio_inputs, audio_file_base):
        spec_utils.combine_audio(audio_inputs, 
                                 os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}"), 
                                 self.wav_type_set,
                                 save_format=self.save_format)
        
    def pitch_or_time_shift(self, audio_file, audio_file_base):
        is_time_correction = True
        rate = float(self.time_stretch_rate) if self.audio_tool == TIME_STRETCH else float(self.pitch_rate)
        is_pitch = False if self.audio_tool == TIME_STRETCH else True
        if is_pitch:
            is_time_correction = True if self.is_time_correction else False
        file_text = TIME_TEXT if self.audio_tool == TIME_STRETCH else PITCH_TEXT
        save_path = os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}{file_text}.wav")
        spec_utils.augment_audio(save_path, audio_file, rate, self.is_normalization, self.wav_type_set, self.save_format, is_pitch=is_pitch, is_time_correction=is_time_correction)

class Processor(ModelData):
    true_model_count = 0
    iteration = 0
    inputPaths = []
    all_models = []

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
            model_data: List[ModelData] = [ModelData(model_name) for model_name in self.model]
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
