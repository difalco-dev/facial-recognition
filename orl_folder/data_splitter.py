import os, sys

### INITS ###
method = "REFORMAT_METHOD"
print('hey')

### FOR EVERY SUBJECT ###
for i in range(1,41):
    pre = ''
    if i < 10:
        pre= '0'
    dir = 's{0}'.format(i)
    file_stem = 'subject{0}{1}.'.format(pre,i)
    ###  REFORMAT ROOT IMAGES ###
    if method=="REFORMAT_METHOD":
        emotes = ['centerlight','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy','surprised','wink']
        for j in range(0,len(emotes)):
            file_name = file_stem+emotes[j]
            print('move {0}\\{1}.pgm {2}'.format(dir,j+1,file_name))
            os.system('move {0}\\{1}.pgm {2}'.format(dir,j+1,file_name))


    ### SPLIT BY SUBJECT ###
    elif method=="SUBJECT_METHOD":
        for emote in ['centerlight','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy','surprised','wink']:
            file_name = file_stem+emote
            ### TRAINING ###
            if i < 6:
                print('move {0} training'.format(file_name))
                os.system('move {0} training'.format(file_name))
            ### VALIDATION ###
            elif i > 5 and i <11:
                print('move {0} validation'.format(file_name))
                os.system('move {0} validation'.format(file_name))
            ### TESTING ###
            else:
                print('move {0} testing'.format(file_name))
                os.system('move {0} testing'.format(file_name))

    ### SPLIT BY FEATURE ###
    elif method=="FEATURE_METHOD":
        ### TRAINING ###
        for emote in ['centerlight','glasses','happy','leftlight']:
            file_name = file_stem+emote
            print('move {0} training'.format(file_name))
            os.system('move {0} training'.format(file_name))
        ### VALIDATION ###
        for emote in ['noglasses','normal','rightlight','sad']:
            file_name = file_stem+emote
            print('move {0} validation'.format(file_name))
            os.system('move {0} validation'.format(file_name))
        ### TESTING ###
        for emote in ['sleepy','surprised','wink']:
            file_name = file_stem+emote
            print('move {0} validation'.format(file_name))
            os.system('move {0} validation'.format(file_name))
