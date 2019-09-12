import os
import re
import argparse

from signal_proc.synthesizer import Synthesizer
from constants.hparams import Hyperparams as hparams


sentences = [
  'အခုပဲ ဝယ်လာလို့ စမ်းဖွင့်ကြည့်တယ်။ နာရီဝက်ပဲ ဖွင့်ရသေးတယ်၊',
  'သူစစ်ပေးထားတဲ့ ရေက မနည်းဘူး ရနေပြီ။ လေထဲမှာ ရေငွေ့ပါဝင်မှု တော်တော် များနေတာပဲ။',
  'ကိုယ်တွေလည်း အသက်ရှုရင်း ရေနစ်နေကြတာ',
  'မန်ယူ - ချယ်ဆီး ပွဲ ကြည့်ဖြစ်တယ်။',
  '"ဘောလုံးဆိုတာအလုံးကြီး" ဆိုတဲ့ စကားလိုပဲ၊ ဘာမဆို ဖြစ်သွားနိုင်တဲ့ ပွဲတွေဆိုပေမယ့်',
  'ဒိုမိန်းတွေကို မြန်မာလိုပေးပြီး စမ်းထားကြတာတစ်ချို့ တွေ့ဖူးတယ်။',
  'ဒါပေမယ့် သိပ် စိတ်မဝင်စားမိဘူး။'
]


def get_output_base_path(checkpoint_path, out_dir):
  base_dir = os.path.abspath(out_dir)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def test(args):
  synthesizer = Synthesizer()
  synthesizer.init(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint, args.out_dir)

  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synthesizer.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--out_dir', default=os.path.expanduser('~/mm-tts'))
  args = parser.parse_args()

  test(args)


if __name__ == "__main__":
  main()
