"""
This is the definition of the whole characters used in the input of the model.
I excluded all the digits since they will be converted into spoken words accordingly.
"""

_pad            = '_'
_eos            = '~'
# from \u1000 to \u1021
_consonants     = 'ကခဂဃငစဆဇဈဉညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဿဟဠအ'
# from \u1023 to \u102A (except \u1028)
_vowels         = 'ဣဤဥဦဧဩဪ'
# from \u102B to \u103E (except \u1033,34,35,39)
_signs          = '\u102B\u102C\u102D\u102E\u102F\u1030\u1031\u1032\u1036\u1037\u1038\u103A\u103B\u103C\u103D\u103E'
# from \u104C to \u104F
_other_signs    = '၌၍၎၏'
# from \u104A to \u104B
_punctuation    = '၊။'
# special characters
_other_chars    = '!\'(),-.:;? '

# export all of them
characters = [_pad, _eos] + list(_consonants + _vowels + _signs + _other_signs + _punctuation + _other_chars)
