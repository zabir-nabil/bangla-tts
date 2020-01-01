# bangla-tts
Multilingual (Bangla, English) real-time ([almost]in a GPU) speech synthesis API

### Installation

 * Install Anaconda
 * `conda create -n new_virtual_env python==3.6.8`
 * `conda activate new_virtual_env`
 * `pip install -r requirements.txt`
 * While running for the first time, keep your internet connection on to download the weights of the speech synthesis models (>500 MB)

### Usage

```python

'''
function: generate(text_arr = [""], save_path = None)
arguments: 
text_arr (array) : an array of strings
save_path (string, optional) : location where generated wav files will be stored if save_path is not None, if the path is not valid, the wav files will be saved in current directory
returns:
if save_path is None, instead of saving an array of tuples containing geenrated speech signals and the sampling rate will be returned
if save_path is not None, then a list containing the file paths (relative) will be returned
'''

from bangla_tts import generate

# usage 1 (saving to path)

file_names = generate(["আমার সোনার বাংলা আমি তোমাকে ভালোবাসি"], save_path = "static") # will be saved to static folder
print(file_names)

# usage 2 (getting numpy arrays for the signals)

gen_wavs = generate(["আমার সোনার বাংলা আমি তোমাকে ভালোবাসি"]) # will return an array containing the speech and sampling rate
print(gen_wavs[0])
print(f"signal length: {gen_wavs[0][0].shape}")
print(f"samplign rate: {gen_wavs[0][1]}")

```

### Generated unseen speech samples


[Sample 1 (আমার সোনার বাংলা আমি তোমাকে ভালোবাসি)](static/0_56258.wav)


[Sample 2 (আমার নাম জাবির আল নাজি নাবিল)](static/1_283811.wav)


[Sample 3 (I am still not a great speaker)](static/2_235924.wav)


[Sample 4 (This is just a test)](static/3_256189.wav)


### To-dos

- [ ] PyPI
- [ ] More training
- [ ] Light model
