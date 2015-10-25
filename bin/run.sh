rm model.dat 
rm model/*
let WIDTH=48
let HEIGHT=32
let TRAIN_SAMPLE_SIZE=2300
MAX_FPR=0.0005
MAX_FNR=0.005
let STAGE=4

make -C .. 


./train --stage $STAGE -X $WIDTH -Y $HEIGHT --numPos $TRAIN_SAMPLE_SIZE --numNeg $TRAIN_SAMPLE_SIZE --false_alarm_rate $MAX_FPR --missing_rate $MAX_FNR \
    --pos pos_list.txt --neg neg_list.txt -m model.dat
