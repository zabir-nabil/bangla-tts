# bangla-tts
Multilingual (Bangla, English) real-time (in a GPU) speech synthesis API

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

file_names = generate(["আমার সোনার বাংলা আমি তোমাকে ভালোবাসি"], save_path = "static")
print(file_names)

# usage 2 (getting numpy arrays for the signals)

gen_wavs = generate(["আমার সোনার বাংলা আমি তোমাকে ভালোবাসি"])
print(gen_wavs[0])
print(f"signal length: {gen_wavs[0][0].shape}")
print(f"samplign rate: {gen_wavs[0][1]}")

```


### To-dos

- [ ] PyPI
- [ ] More training
- [ ] Light model
