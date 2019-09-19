import os
import argparse

import falcon

from signal_proc.synthesizer import Synthesizer
from constants.hparams import Hyperparams as hparams


html_body = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <title>MM-TTS</title>
  <style>
    .wrapper {
      display: flex;
    }
  </style>
  <!-- Compiled and minified CSS -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
</head>
<body>
<div class="row" style="height: 100vh">
  <div class="col s12 light-blue darken-4" style="height: 40%">
    <h2 class="center-align white-text">Myanmar TTS</h2>
  </div>
  <div class="col s12" style="margin-top: -10%">
    <div class="container">
      <div class="card large z-depth-4">
        <form class="col s12">
          <div class="card-content">
            <span class="card-title"><h4>End-to-End Speech Synthesis</h4></span>
            <br>
            <p>You can type any Burmese sentences in Unicode and the model will try to
              synthesize the speech based on your inputs. <br/>
              Synthesizing process may take a few seconds.
            </p>
            <div class="row" style="margin-top: 2rem;">
              <div class="input-field col s12">
                <input placeholder="Type Here ..." id="text" type="text">
                <label for="text">Input Text</label>
              </div>
            </div>
            <div class="row">
              <div class="col s6">
                <h6 id="intxt" class="teal-text"></h6>
                <p id="inputtxt" style="margin: 1rem 0"></p>
              </div>
              <div class="col s6">
                <h6 id="message" class="teal-text"></h6>
                <audio src="/" id="audio" controls autoplay hidden style="margin-top: 0.5rem; margin-left: 3rem"></audio>
              </div>
            </div>
          </div>
          <div class="card-action">
            <button id="button" name="synthesize" class="waves-effect btn waves-light">Speak</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</div>
  <script>
    function q(selector) {return document.querySelector(selector)}
    q('#text').focus()
    q('#button').addEventListener('click', function(e) {
      text = q('#text').value.trim()
      if (text) {
        q('#message').textContent = 'Synthesizing...'
        q('#intxt').textContent = 'Input'
        q('#inputtxt').textContent = text
        q('#button').disabled = true
        q('#audio').hidden = true
        synthesize(text)
      }
      e.preventDefault()
      return false
    })
    function synthesize(text) {
      fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
        .then(function(res) {
          if (!res.ok) throw Error(res.statusText)
          return res.blob()
        }).then(function(blob) {
          q('#message').textContent = 'Synthesized!'
          q('#button').disabled = false
          q('#audio').src = URL.createObjectURL(blob)
          q('#audio').hidden = false
        }).catch(function(err) {
          q('#message').textContent = 'Error: ' + err.message
          q('#button').disabled = false
        })
    }
  </script>
  <!-- Compiled and minified JavaScript -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
</body>
</html>
'''


class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


class SynthesisResource:
  def on_get(self, req, res):
    inputTxt = req.params.get('text')
    if not inputTxt:
      raise falcon.HTTPBadRequest()
    res.data = synthesizer.synthesize(inputTxt)
    res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())


if __name__ == '__main__':
  from wsgiref import simple_server
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=4000)
  args = parser.parse_args()
  
  synthesizer.init(args.checkpoint)

  print('Serving on port %d' % args.port)
  simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
