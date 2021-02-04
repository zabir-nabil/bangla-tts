# author: github/zabir-nabil
# a parser for bangla, for converting numeric character sets to phonetic representation
# if length is > len, split it, pass in a single batch and reconstruct

num_phonetic_map = {'০': 'শূন্য',
                     '১': 'এক',
                     '২': 'দুই',
                     '৩': 'তিন',
                     '৪': 'চার',
                     '৫': 'পাঁচ',
                     '৬': 'ছয়',
                     '৭': 'সাত',
                     '৮': 'আট',
                     '৯': 'নয়',
                     '১০': 'দশ',
                     '১১': 'এগার',
                     '১২': 'বার',
                     '১৩': 'তের',
                     '১৪': 'চৌদ্দ',
                     '১৫': 'পনের',
                     '১৬': 'ষোল',
                     '১৭': 'সতের',
                     '১৮': 'আঠার',
                     '১৯': 'ঊনিশ',
                     '২০': 'বিশ',
                     '২১': 'একুশ',
                     '২২': 'বাইশ',
                     '২৩': 'তেইশ',
                     '২৪': 'চব্বিশ',
                     '২৫': 'পঁচিশ',
                     '২৬': 'ছাব্বিশ',
                     '২৭': 'সাতাশ',
                     '২৮': 'আঠাশ',
                     '২৯': 'ঊনত্রিশ',
                     '৩০': 'ত্রিশ',
                     '৩১': 'একত্রিশ',
                     '৩২': 'বত্রিশ',
                     '৩৩': 'তেত্রিশ',
                     '৩৪': 'চৌত্রিশ',
                     '৩৫': 'পঁয়ত্রিশ',
                     '৩৬': 'ছত্রিশ',
                     '৩৭': 'সাঁইত্রিশ',
                     '৩৮': 'আটত্রিশ',
                     '৩৯': 'ঊনচল্লিশ',
                     '৪০': 'চল্লিশ',
                     '৪১': 'একচল্লিশ',
                     '৪২': 'বিয়াল্লিশ',
                     '৪৩': 'তেতাল্লিশ',
                     '৪৪': 'চুয়াল্লিশ',
                     '৪৫': 'পঁয়তাল্লিশ',
                     '৪৬': 'ছেচল্লিশ',
                     '৪৭': 'সাতচল্লিশ',
                     '৪৮': 'আটচল্লিশ',
                     '৪৯': 'ঊনপঞ্চাশ',
                     '৫০': 'পঞ্চাশ',
                     '৫১': 'একান্ন',
                     '৫২': 'বায়ান্ন',
                     '৫৩': 'তিপ্পান্ন',
                     '৫৪': 'চুয়ান্ন',
                     '৫৫': 'পঞ্চান্ন',
                     '৫৬': 'ছাপ্পান্ন',
                     '৫৭': 'সাতান্ন',
                     '৫৮': 'আটান্ন',
                     '৫৯': 'ঊনষাট',
                     '৬০': 'ষাট',
                     '৬১': 'একষট্টি',
                     '৬২': 'বাষট্টি',
                     '৬৩': 'তেষট্টি',
                     '৬৪': 'চৌষট্টি',
                     '৬৫': 'পঁয়ষট্টি',
                     '৬৬': 'ছেষট্টি',
                     '৬৭': 'সাতষট্টি',
                     '৬৮': 'আটষট্টি',
                     '৬৯': 'ঊনসত্তর',
                     '৭০': 'সত্তর',
                     '৭১': 'একাত্তর',
                     '৭২': 'বাহাত্তর',
                     '৭৩': 'তিয়াত্তর',
                     '৭৪': 'চুয়াত্তর',
                     '৭৫': 'পঁচাত্তর',
                     '৭৬': 'ছিয়াত্তর',
                     '৭৭': 'সাতাত্তর',
                     '৭৮': 'আটাত্তর',
                     '৭৯': 'ঊনআশি',
                     '৮০': 'আশি',
                     '৮১': 'একাশি',
                     '৮২': 'বিরাশি',
                     '৮৩': 'তিরাশি',
                     '৮৪': 'চুরাশি',
                     '৮৫': 'পঁচাশি',
                     '৮৬': 'ছিয়াশি',
                     '৮৭': 'সাতাশি',
                     '৮৮': 'আটাশি',
                     '৮৯': 'ঊননব্বই',
                     '৯০': 'নব্বই',
                     '৯১': 'একানব্বই',
                     '৯২': 'বিরানব্বই',
                     '৯৩': 'তিরানব্বই',
                     '৯৪': 'চুরানব্বই',
                     '৯৫': 'পঁচানব্বই',
                     '৯৬': 'ছিয়ানব্বই',
                     '৯৭': 'সাতানব্বই',
                     '৯৮': 'আটানব্বই',
                     '৯৯': 'নিরানব্বই',
                     '-': ' ',
                     ' ': ' '}

eng_num_map = {
    '1': 'ওয়ান',
    '2': 'টু',
    '3': 'থ্রি',
    '4': 'ফৌর',
    '5': 'ফাইভ',
    '6': 'সিক্স',
    '7': 'সেভেন',
    '8': 'এইট',
    '9': 'নাইন',
    ' ': ' ',
    '-': ''
}

hundred = 'শ'
thousand = 'হাজার'
lakh = 'লক্ষ'
crore = 'কোটি'


def en(num_k):
    for c in num_k:
        if c not in eng_num_map.keys():
            return False
        
    return True



def bn(num_k):
    for c in num_k:
        if c not in num_phonetic_map.keys():
            return False
        
    return True

def num_process(num_k):
    out = ''
    
    for c in num_k:
        if c in num_phonetic_map.keys():
            out += num_phonetic_map[c] + ' '
        elif c in eng_num_map.keys():
            out += eng_num_map[c] + ' '
        else:
            out += c
            
    return out
            
def enum_process(num_k): # only pass if characters have english numeric
    out = ''.join(eng_num_map[a] + ' ' if a in eng_num_map.keys() else a for a in num_k)
    return out
    
def bnum_process(num_k): # only pass if every character is bangla numeric
    num_k = num_k.replace(' ', '')
    out = ''
    if len(num_k) > 9 or ''.join(a for a in num_k if a not in num_phonetic_map.keys()):
        # separate pronunciation
        out = ''.join(num_phonetic_map[a] + ' ' if a in num_phonetic_map.keys() else a + ' ' for a in num_k)

    elif len(num_k) == 4 and num_k[0] != '০': # most probably an year
        out = num_phonetic_map[num_k[:2]] + hundred + num_phonetic_map[num_k[2:]]
        
        return out
    else:

        while num_k.startswith('০'):
            num_k = num_k[1:]
        if len(num_k) >= 8:
            out += num_phonetic_map[ num_k[:len(num_k)-7] ] + ' ' + crore + ' '
            num_k = num_k[len(num_k)-7:]
            while num_k.startswith('০'):
                num_k = num_k[1:]
        if len(num_k) >= 6:
            out += num_phonetic_map[ num_k[:len(num_k)-5] ] + ' ' + lakh + ' '
            num_k = num_k[len(num_k)-5:]  
            while num_k.startswith('০'):
                num_k = num_k[1:]
        if len(num_k) >= 4:
            out += num_phonetic_map[ num_k[:len(num_k)-3] ] + ' ' + thousand + ' '
            num_k = num_k[len(num_k)-3:]    
            while num_k.startswith('০'):
                num_k = num_k[1:]
        if len(num_k) >= 3:
            out += num_phonetic_map[ num_k[:len(num_k)-2] ] + ' ' + hundred + ' '
            num_k = num_k[len(num_k)-2:]
            while num_k.startswith('০'):
                num_k = num_k[1:]
        if len(num_k) >= 1:
            out += num_phonetic_map[ num_k[:len(num_k)-0] ] + ' '
            num_k = num_k[len(num_k)-0:]     
        
        
    return out.strip()

def process(in_str):
    in_str = in_str.replace('-', ' ').replace('।', ' ').replace(',', ' ')
    
    ls_str = []
    
    split_str_c = in_str.split(' ')
    
    split_str = []
    
    for ck in split_str_c:
        if en(ck): # eng numeric, other bangla chars
            split_str.append(enum_process(ck))
        elif bn(ck): # only bangla numeric
            split_str.append(bnum_process(ck))
        else: # mix
            #print(ck)
            split_str.append(num_process(ck))
            
    #print(split_str)
    
    cchln = 0
    cw = ''
    
    for w in split_str:
        cw += w + ' '
        cchln += len(w)
        if cchln >= 50:
            ls_str.append(cw)
            cw = ''
            cchln = 0
            
    if cchln != 0:
        ls_str.append(cw)
        
    return ls_str 
