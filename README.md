## KWS

--- 

### Instalation Guide

--- 

To install, run the following commands in a terminal.

```shell
# ffmpeg4 instalation
add-apt-repository -y ppa:savoury1/ffmpeg4
apt-get -qq install -y ffmpeg

git clone https://github.com/fedorovgv/kws.git

pip install -r requirements.txt
```

For Google Command Speech downloading.

```shell
cd kws && chmod +x speech_comands.sh && cd .. && kws/./speech_comands.sh 
rm speech_commands_v0.01.tar.gz
```
