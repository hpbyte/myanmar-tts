from datetime import datetime

_log_file = None
_date_time_format = '%Y-%m-%d %H:%M:%S.%f'


def init(filename):
  global _log_file
  close_logging()
  _log_file = open(filename, 'a', encoding='utf-8')
  _log_file.write('\n------------------------------------------')
  _log_file.write('\nStarting new training---------------------')
  _log_file.write('\n------------------------------------------')


def log(msg):
  print(msg)
  if _log_file is not None:
    _log_file.write('[%s]  %s\n' % (datetime.now().strftime(_date_time_format)[:-3], msg))


def close_logging():
  global _log_file
  if _log_file is not None:
    _log_file.close()
    _log_file = None
