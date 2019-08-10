# Myanmar End-to-End Text-to-Speech

This is the development of a Myanmar Text-to-Speech system with the famous End-to-End Speech Synthesis Model, Tacotron. This is a part of a thesis research that I've been doing and the implementation of the Tacotron model is **heavily** influenced by the following implementation:

* https://github.com/keithito/tacotron

## Text and Speech Corpus

**[Base Technology, Expa.Ai (Myanmar)](https://expa.ai)** kindly provided Myanmar text corpus and their amazing tool for creating speech corpus.

Text corpus contains about 19K lines of Myanmar text in Unicode.

Speech corpus (mmSpeech as I call it) is created solely on my own with a recorder tool (as previously mentioned) and it currently contains over 3,000 recorded audio samples and still growing.

## Instructions

### Installing dependencies

1.  Install Python 3
2.  Install the latest version of [TensorFlow](https://www.tensorflow.org/install/)
3.  Install a number of modules
    ```
    pip install -r requirements.txt
    ```

### Preparing Text and Audio Dataset

1.  First of all, both Text and Speech corpus should reside in `~/Documents/myanmar-tts`, although it is not a **must** and can easily be changed by a command line argument.
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
