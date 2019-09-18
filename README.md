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

You can download the trained model [here](https://drive.google.com/open?id=1P3JQYjGNoPbNykOg4-45LPUsZGuWkAd7)

```
python3 app.py --checkpoint /path/to/checkpoint/model.ckpt.150000
```


### Notes

* Google Colab which gives excellent GPU access was used for training this model.
* On average, each step tooks about 1.6 seconds and at peak, each step took about 1.2 and sometimes 1.1 seconds.
* For my thesis, I have trained this model for 150,000 steps (took me about a week).

Below is the produced loss curves from training mmSpeech for 150,000 Steps.

![Loss](https://user-images.githubusercontent.com/34838719/65132116-7353e080-da26-11e9-9299-a08883811f47.png)

![Linear Loss](https://user-images.githubusercontent.com/34838719/65132950-d42fe880-da27-11e9-8bc1-1ef7ba95cae3.png)

![Mel Loss](https://user-images.githubusercontent.com/34838719/65132986-e578f500-da27-11e9-9214-a5ead8c0bdfc.png)


### References

* [Tacotron: Towards End-to-End Speech Synthesis](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiBjuL828vkAhWh6nMBHYccCdYQFjABegQIABAB&url=https%3A%2F%2Farxiv.org%2Fabs%2F1703.10135&usg=AOvVaw0_KT-Hbe9h_egPMynMsJOM)
* [keithito/tacotron](https://github.com/keithito/tacotron)
