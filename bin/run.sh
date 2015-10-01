let WIDTH=48
let HEIGHT=32
let POS_SAMPLE_SIZE=900
let NEG_SAMPLE_SIZE=10000
let TRAIN_SAMPLE_SIZE=700
let MAX_FPR=0.0005
let MAX_FNR=0.005
let STAGE=15

make -C ..

echo "GENERATE POSITIVE SAMPLES ..."
./samples 1 pos_list.txt $WIDTH $HEIGHT $POS_SAMPLE_SIZE

echo "GENERATE NEGATIVE SAMPLES ..."
./samples 0 neg_list.txt $WIDTH $HEIGHT $NEG_SAMPLE_SIZE

./train --stage $STAGE -X $WIDTH -Y $HEIGHT --numPos $TRAIN_SAMPLE_SIZE --numNeg $TRAIN_SAMPLE_SIZE --false_alarm_rate $MAX_FPR --missing_rate $MAX_FNR \
    --pos pos_sample.bin --neg neg_sample.bin -m model.dat
