# bangla text to speech
Multilingual (Bangla, English) real-time ([almost] in a GPU) speech synthesis library

### Installation

 * Install Anaconda
 * `conda create -n new_virtual_env python==3.6.8`
 * `conda activate new_virtual_env`
 * `pip install -r requirements.txt`
 * While running for the first time, keep your internet connection on to download the weights of the speech synthesis models (>500 MB)
 * For fast inference, you must install tensorflow-gpu and have a NVidia GPU.

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

### Update (18th April)

- [x] Synthesize longer sentences
- [x] Phonetic representation for English, Bangla numeric segments

 * Added a simple parser which will translate numeric keys to corresponding phonetic representation.

 Example: *১৯৯৭ সালের ২১ জানুয়ারী তে আমার জন্ম হয়* will be converted to *['ঊনিশশ সাতানব্বই সালের একুশ জানুয়ারী তে আমার জন্ম হয় ']* by the parser.

 * Added a simple batch mechanism for translating longer sentences. As the attention window was fixed during training, the model previously failed to generate long sentences (n_characters > 200). So, added a simple segmenting scheme to break the sentences into multiple parts, synthesize in batch, and finally merge them into a single audio file.


 **New examples:**

 [১৯৯৭ সালের ২১ জানুয়ারী তে আমার জন্ম হয়](birthdate.wav)


 [আমার ফোন নাম্বার ০১৭১৩৩৫৩৪৩, তবে আমাকে সকাল ১০ টার আগে পাবেন না](phone_number.wav)


 [বাংলাদেশে গত ২৪ ঘণ্টায় ৩০৬ জন কোভিড-১৯ আক্রান্ত হয়েছেন। এই সময়ের মধ্যে মৃত্যু হয়েছে ৯ জনের। এ নিয়ে দেশটিতে মোট আক্রান্ত হলেন ২১৪৪। আর করোনা ভাইরাসে আক্রান্ত হয়ে মৃত্যু হয়েছে ৮৪ জনের। নতুন করে ৮ জনের পরীক্ষা করার পর করোনা ভাইরাসের উপস্থিতি পাওয়া যায়নি। এনিয়ে মোট ৬৬ জন সুস্থ হলেন।](covid19.wav) - BBC Bangla


### To-dos

- [ ] PyPI
- [ ] More training
- [ ] Light model
- [ ] Publish the restful API
- [ ] Publish the flask app


> Usage granted only for educational/non-commerial purposes so far, ** GPL License **

### If this repository helps you in anyway, show your love :heart: by putting a :star: on this project :v:

