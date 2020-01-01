# python 3.6
# github/zabir-nabil

# signal processing
sr = 22050 # more data, test # 48000 # 22050  # Sampling rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr * frame_shift)  # samples. =276.
win_length = int(sr * frame_length)  # samples. =1102.
n_mels = 80  # Number of Mel banks to generate
power = 1.5  # Exponent for amplifying the predicted magnitude
n_iter = 50  # Number of inversion iterations
preemphasis = .97
max_db = 100
ref_db = 20

# Model
r = 4 # Reduction factor. Do not change this.
dropout_rate = 0.05
e = 128 # == embedding
d = 256 # == hidden units of Text2Mel
c = 512 # 512 # == hidden units of SSRN
attention_win_size = 3

# synthesis
vocab = "PE &্0123456789অoোyয়ওঐৌৈঔüuুwউঊূaাআàâeèéেêএিইঈীiংঙঞঃশষসsহhণনnটtঠডdঢৎতথদধপpফfবbভvমmলlযঝজjzছচcxকkqখগgঘঋৃড়rর" # P: Padding, E: EOS.
max_N = 180 # Maximum number of characters.
max_T = 210 # Maximum number of mel frames.

