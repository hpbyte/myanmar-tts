"""
This module has a separate git repository https://github.com/hpbyte/Myanmar_Number_to_Words.git
"""

import re

mm_digit = {
  '၀': 'သုည',
  '၁': 'တစ်',
  '၂': 'နှစ်',
  '၃': 'သုံ:',
  '၄': 'လေ:',
  '၅': 'ငါ:',
  '၆': 'ခြောက်',
  '၇': 'ခုနှစ်',
  '၈': 'ရှစ်',
  '၉': 'ကို:'
}

# regular expressions
rgxPh = '^(၀၁|၀၉)'
rgxDate = '[၀-၉]{1,2}-[၀-၉]{1,2}-[၀-၉]{4}|[၀-၉]{1,2}\/[၀-၉]{1,2}\/[၀-၉]{4}'
rgxTime = '[၀-၉]{1,2}:[၀-၉]{1,2}'
rgxDec = '[၀-၉]*\.[၀-၉]*'
rgxAmt = '[,၀-၉]+'


def convert_digit(num):
  """
  @type     num   str
  @param    num   Myanmar number
  @rtype          str
  @return         converted Myanmar spoken words
  """

  converted = ''
  nb_digits = len(num)

  def check_if_zero(pos):
    return not num[-pos] == '၀'

  def hundred_thousandth_val():
    n = num[:-5]
    return ('သိန်: ' + mm_num2word(n)) if (n[-2:] == '၀၀') else (mm_num2word(n) + 'သိန်: ')

  def thousandth_val():
    return mm_digit[num[-4]] + ('ထောင် ' if (num[-3:] == '၀၀၀') else 'ထောင့် ')

  def hundredth_val():
    return mm_digit[num[-3]] + ('ရာ့ ' if (
      (num[-2] == '၀' and re.match(r'[၁-၉]', num[-1])) or (re.match(r'[၁-၉]', num[-2]) and num[-1] == '၀')
    ) else 'ရာ ')

  def tenth_val():
    return ('' if (num[-2] == '၁') else mm_digit[num[-2]]) + ('ဆယ် ' if (num[-1] == '၀') else 'ဆယ့် ')

  if ((nb_digits > 5)):
    converted += hundred_thousandth_val()
  if ((nb_digits > 4) and check_if_zero(5)):
    converted += mm_digit[num[-5]] + 'သောင်: '
  if ((nb_digits > 3) and check_if_zero(4)):
    converted += thousandth_val()
  if ((nb_digits > 2) and check_if_zero(3)):
    converted += hundredth_val()
  if ((nb_digits > 1) and check_if_zero(2)):
    converted += tenth_val()
  if ((nb_digits > 0) and check_if_zero(1)):
    converted += mm_digit[num[-1]]

  return converted


def mm_num2word(num):
  """
  Detect type of number and convert accordingly

  @type     num   str
  @param    num   Myanmar number
  @rtype          str
  @return         converted Myanmar spoken words
  """
  
  word = ''

  # phone number
  if (re.match(r'' + rgxPh, num[:2])):
    word = ' '.join([(mm_digit[d] if not d == '၇' else 'ခွန်') for d in num])
  # date
  elif (re.match(r'' + rgxDate, num)):
    n = re.split(r'-|/', num)
    word = convert_digit(n[-1]) + ' ခုနှစ် ' + convert_digit(n[1]) + ' လပိုင်: ' + convert_digit(n[0]) + ' ရက်'
  # time
  elif (re.match(r'' + rgxTime, num)):
    n = re.split(r':', num)
    word = (convert_digit(n[0]) + ' နာရီ ') + ('ခွဲ' if (n[1] == '၃၀') else (convert_digit(n[1]) + ' မိနစ်'))
  # decimal
  elif (re.match(r'' + rgxDec, num)):
    n = re.split(r'\.', num)
    word = convert_digit(n[0]) + ' ဒဿမ ' + ' '.join([mm_digit[d] for d in n[1]])
  # amount
  elif (re.match(r'' + rgxAmt, num)):
    word = convert_digit(num.replace(',', ''))
  # default
  else:
    raise Exception('Cannot convert the provided number format!')

  return word


def extract_num(S):
  """
  Extract numbers from the input string

  @type     S   str
  @param    S   Myanmar sentence
  @rtype        list
  @return       a list of Myanmar numbers
  """
  matchedNums = re.compile('%s|%s|%s|%s' % (rgxDate, rgxTime, rgxDec, rgxAmt)).findall(S)

  return matchedNums
