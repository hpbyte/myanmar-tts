import re

from text.character_set import characters
from text.mm_num2word import extract_num, mm_num2word

# mappings of each character and its id
char_to_id = {s : i for i, s in enumerate(characters)}
id_to_char = {i : s for i, s in enumerate(characters)}


def _should_keep_char(c):
  """
  Determines whether the input character is defined in the character set

  @type   c   str
  @param  c   char
  
  @rtype      bool
  @return     result of the check
  """
  return c in char_to_id and c is not '_' and c is not '~'


def collapse_whitespace(text):
  """
  Combine a series of whitespaces into a single whitespace

  @type   text    str
  @param  text    input string of text
  
  @rtype          str
  @return         collapsed text string
  """
  rgx_whitespace = re.compile(r'\s+')
  return re.sub(rgx_whitespace, ' ', text)


def numbers_to_words(text):
  """
  Convert numbers into corresponding spoken words

  @type   text    str
  @param  text    input string of text

  @rtype          str
  @return         converted spoken words
  """
  nums = extract_num(text)
  for n in nums:
    text = text.replace(n, mm_num2word(n))

  return text


def normalize(text):
  """
  Normalize text string for numbers and whitespaces

  @type   text    str
  @param  text    input string of text

  @rtype          str
  @return         normalized string
  """
  text = collapse_whitespace(text)
  text = numbers_to_words(text)

  return text


def text_to_sequence(text):
  """
  Convert an input text into a sequence of ids

  @type   text    str
  @param  text    input string of text

  @rtype          list
  @return         list of IDs corresponding to the characters
  """
  text = normalize(text)
  seq = [char_to_id[c] for c in text if _should_keep_char(c)]

  seq.append(char_to_id['~'])
  return seq


def sequence_to_text(seq):
  """
  Convert a sequence of ids into the corresponding characters

  @type   seq   list
  @param  seq   list of ids

  @rtype        str
  @return       a string of text
  """
  text = ''
  for char_id in seq:
    if char_id in id_to_char:
      text += id_to_char[char_id]

  return text

