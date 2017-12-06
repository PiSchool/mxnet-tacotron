# Tacotron over MXNet

A tech demo of MXNet capabilities consisting of a Tacotron implementation. This is a work in progress. 

This project was made during the 8 weeks from 10-2017 to 12-2017 at the PiCampus AI School in Rome.
## List of functionalities and TODOs
- [x] Multithreading data iterator
- [x] DSP tools 
- [x] CBHG module for spectrograms
- [x] Basic seq2seq example for string reverse. It we'll be used as Tacotron backbone
- [ ] Encoder with CBHG 
- [ ] Attention model
- [ ] Custom decoder for processing r * mel_bands spectrograms frames for each time step during the cell unrolling  
- [ ] Switch to MXNet 1.0
- [ ] Switch to Gluon
- [ ] Clean up and organize code for better understanding

## Getting Started
* install MXNet: 
<code> pip install -r requirements.txt </code>
* run:
<code> python tacotron.py </code>

Using the default setting, a simple dataset will be used as training. Predictions samples will be generated at the end of the training phase.

If you want to train over a big dataset, Kyubyong has cut and formatted <a href="http://www.audiotreasure.com/webindex.htm"> this </a> English bible. You can find his dataset  <a href="https://www.dropbox.com/s/nde56czgda8q77e/WEB.zip?dl=0"> here </a> and the CSV text <a href="https://www.dropbox.com/s/lcfhs1kk9shvypj/text.csv?dl=0">here</a> . 
### Prerequisites
This project has been developed on 

- MXNet 0.12
- librosa

## Authors

* **Alberto Massidda** - https://github.com/aijanai
* **Stefano Artuso**

## Acknowledgments
* Thanks to Roberto Barra Chicote for supporting us
* Thanks to Keith Ito https://github.com/keithito, Kyubyong Park https://github.com/Kyubyong for making us start diving in


