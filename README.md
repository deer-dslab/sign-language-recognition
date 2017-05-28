# sign-language-recognition

# Database
Downloaded from [American Sign Language Lexicon Video Dataset (ASLLVD)](http://vlm1.uta.edu/~athitsos/asl_lexicon/)

Use only few videos bellow for I/O test.
- Liz, Naomi and Tyler's scene1-camera1, scene2-camera1 and scene3-camera1 of
ASL_2008_01_18


# Steps
1. Split videos into images by [FFmpeg](http://qiita.com/f2um2326/items/5940b6cf2fada8132f85).
1. Generate label.txt for each videos by using csv2txt.py. The CSV file can be downloaded from the database.
1. By using setup.py and label.txt, separate images into each label dir.
1. Extract features by using extractor.py. Extracted features are in /dataset/results
1. Run lstm.py

# Reference
Source code in /reference dir is from [aidiary/keras-examples: Kerasのサンプルプログラム](https://github.com/aidiary/keras-examples).
