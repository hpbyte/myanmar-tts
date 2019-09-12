# Myanmar End-to-End Text-to-Speech

This is the development of a Myanmar Text-to-Speech system with the famous End-to-End Speech Synthesis Model, Tacotron. It is a part of a thesis for B.E. Degree that I've been assigned at Yangon Technological University.

## Corpus

**[Base Technology, Expa.Ai (Myanmar)](https://expa.ai)** kindly provided Myanmar text corpus and their amazing tool for creating speech corpus.

Speech corpus (mmSpeech as I call it) is created solely on my own with a recorder tool (as previously mentioned) and it currently contains over 5,000 recorded `<text, audio>` pairs. I intend to upload the created corpus on some channel in future.

## Instructions

### Installing dependencies

1.  Install Python 3
2.  Install [TensorFlow](https://www.tensorflow.org/install/)
3.  Install a number of modules
    ```
    pip install -r requirements.txt
    ```


### Preparing Text and Audio Dataset

1.  First of all, the corpus should reside in `~/mm-tts`, although it is not a **must** and can easily be changed by a command line argument.
    ```
    mm-tts
      | mmSpeech
        | metadata.csv
        | wavs
    ```

2.  **Preprocess the data**
    ```
      python3 preprocess.py
    ```
    After it is done, you should see the outputs in `~/mm-tts/training/`


### Training

```
python3 train.py
```

If you want to restore the step from a checkpoint
```
python3 train.py --restore_step Number
```


### Evaluation

There are some sentences defined in test.py, you may test them out with the trained model to see how good the current model is.
```
python3 test.py --checkpoint /path/to/checkpoint
```


### Testing with Custom Inputs

There is a simple app implemented to try out the trained models for their performance.
```
python3 app.py --checkpoint /path/to/checkpoint
```
This will create a simple web app listening at port 4000 unless you specify.
Open up your browser and go to `http://localhost:4000`, you should see a simple interface with a text input to get the text from the user.


### References

* [Tacotron: Towards End-to-End Speech Synthesis](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiBjuL828vkAhWh6nMBHYccCdYQFjABegQIABAB&url=https%3A%2F%2Farxiv.org%2Fabs%2F1703.10135&usg=AOvVaw0_KT-Hbe9h_egPMynMsJOM)
* [keithito/tacotron](https://github.com/keithito/tacotron)
