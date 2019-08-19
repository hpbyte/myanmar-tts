# Myanmar End-to-End Text-to-Speech

This is the development of a Myanmar Text-to-Speech system with the famous End-to-End Speech Synthesis Model, Tacotron. This is a part of a thesis research that I've been doing and the implementation of the Tacotron model is **heavily** influenced by the following implementation:

* https://github.com/keithito/tacotron

## Corpus

**[Base Technology, Expa.Ai (Myanmar)](https://expa.ai)** kindly provided Myanmar text corpus and their amazing tool for creating speech corpus.

Speech corpus (mmSpeech as I call it) is created solely on my own with a recorder tool (as previously mentioned) and it currently contains over 4,000 recorded `<text, audio>` samples.

## Instructions

### Installing dependencies

1.  Install Python 3
2.  Install the latest version of [TensorFlow](https://www.tensorflow.org/install/)
3.  Install a number of modules
    ```
    pip install -r requirements.txt
    ```


### Preparing Text and Audio Dataset

1.  First of all, the corpus should reside in `~/Documents/myanmar-tts`, although it is not a **must** and can easily be changed by a command line argument.
    ```
    myanmar-tts
      | mmSpeech
        | metadata.csv
        | wavs
    ```

2.  **Preprocess the data**
    ```
      python3 preprocess.py
    ```
    After it is done, you should see the outputs in `~/Documents/myanmar-tts/training/`


### Training

```
python3 train.py
```

If you want to restore the step from a checkpoint
```
python train.py --restore_step=Number
```


### Testing

There are some sentences defined in test.py, you may test them out with the trained model to see how good the current model is.
```
python3 test.py --checkpoint /path/to/checkpoint
```


### Realtime Text-to-Speech

There is a simple app implemented to try out the trained models for their performance.
```
python3 app.py --checkpoint /path/to/checkpoint
```
This will create a simple web app listening at port 9000 unless you specify.
Open up your browser and go to `http://localhost:9000`, you should see a simple interface with a text input to get the text from the user.

