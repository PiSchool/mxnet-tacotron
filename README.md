# Project Title

A Tacotron implementation over MXNet. 

This project was made during the 8 weeks of the AI School at PiSchool in Rome.
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
<code> pip install mxnet==0.12 </code>
note: mxnet 1.0 was not released during the development of this project. Try it, if you want
* run:
<code> python tacotron.py </code>

Using the default setting, a simple dataset will be used as training. Predictions samples will be generated at the end of the training phase.

If you want to train over a big dataset, Kyubyong has cutted and formatted <a href="http://www.audiotreasure.com/webindex.htm"> this </a> english bible. You can find his dataset  <a href="https://www.dropbox.com/s/nde56czgda8q77e/WEB.zip?dl=0"> here </a> and the CSV text <a href="https://www.dropbox.com/s/lcfhs1kk9shvypj/text.csv?dl=0">here</a> . 
### Prerequisites
This project has been developed on 

- MXNet 0.12

## Authors

* **Alberto Massidda** - https://github.com/aijanai
* **Stefano Artuso**

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* Thanks to Roberto Barra Chicote for supporting us
* Thanks to Keith Ito https://github.com/keithito, Kyubyong Park https://github.com/Kyubyong for make it us start


