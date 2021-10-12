# TO-DO: Wrap into dataloader (and wav2lps?). Save LTS for training/testing.

# Import lts_maker
from soundscape_IR.soundscape_viewer import lts_maker

# Define parameters and collect all the audio files from the designated folder
LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=1024, window_overlap=0, initial_skip=0)
LTS_run.collect_folder(path=r"D:\Data")

# Define the format of time stamps and run the lts_maker
LTS_run.filename_check(dateformat='yymmddHHMMSS',initial='5738.',year_initial=2000)
LTS_run.run(save_filename=save_filename,folder_id=[], file_begin= 0, num_file = 20)
