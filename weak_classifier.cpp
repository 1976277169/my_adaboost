#include "weak_classifier.h"
#include "tool.h"

typedef struct {
    float value;
    int label;
    float weight;
} Scores;


#define LT(a, b) ((a).value < (b).value)

static IMPLEMENT_QSORT(sort_arr_scores, Scores, LT);

void init_weak_classifier(WeakClassifier *weak, float thresh, int sign, Feature *feat)
{
    weak->thresh = thresh;
    weak->sign = sign;
    weak->feat = feat;
}


float train(WeakClassifier *weak, float *value, int posSize, int negSize, float *weights)
{
    int sampleSize = negSize + posSize;

    float maxv = -FLT_MAX;
    float minv = FLT_MAX;

    const int LENGTH = 1024;

    float posWs[LENGTH];
    float negWs[LENGTH];

    float *ptrValue, *ptrW;
    float sumpw = 0, sumnw = 0;

    for(int i = 0; i < sampleSize; i++){
        maxv = HU_MAX(value[i], maxv);
        minv = HU_MIN(value[i], minv);
    }

    float step = (maxv - minv) / (LENGTH - 1);

    memset(posWs, 0, sizeof(float) * LENGTH);
    memset(negWs, 0, sizeof(float) * LENGTH);

    ptrValue = value;
    ptrW = weights;

    for(int i = 0; i < posSize; i++){
        int id = (ptrValue[i] - minv) / step;

        posWs[id] += weights[i];
        sumpw += weights[i];
    }

    ptrValue = value + posSize;
    ptrW = weights + posSize;

    for(int i = 0; i < negSize; i++){
        int id = (ptrValue[i] - minv) / step;

        negWs[id] += weights[i];
        sumnw += weights[i];
    }


    float minError = FLT_MAX, error;
    float t1, t2;

    posWs[0] /= sumpw;
    negWs[0] /= sumnw;

    for(int i = 1; i < LENGTH; i++){
        posWs[i] /= sumpw;
        posWs[i] += posWs[i - 1];

        negWs[i] /= sumnw;
        negWs[i] += negWs[i - 1];

        t1 = posWs[i] + 1 - negWs[i];
        t2 = negWs[i] + 1 - posWs[i];

        error = (HU_MIN(t1, t2)) / 2;

        if(error < minError){
            minError = error;
            weak->thresh = minv + i * step;

            if(t1 < t2)
                weak->sign = 1;
            else
                weak->sign = 0;
        }
    }

    return minError;
}


int classify(WeakClassifier *weak, Sample *sample){
    assert(weak->feat != NULL);

    float value = get_value(sample, weak->feat);

    float thresh = weak->thresh;
    float sign = weak->sign;

    if( (value > thresh && sign == 1) || (value <= thresh && sign == 0) )
        return POS_SAMPLE_FLAG;

    else
        return NEG_SAMPLE_FLAG;

}


void save(WeakClassifier *weak, FILE *fout)
{
    assert(fout != NULL);

    fwrite(&weak->thresh, sizeof(float), 1, fout);
    fwrite(&weak->sign, sizeof(int), 1, fout);
    fwrite(weak->feat, sizeof(Feature), 1, fout);
}


void load(WeakClassifier **aWeak, FILE *fin)
{
    WeakClassifier *weak = new WeakClassifier;
    int ret = 0;
    weak->feat = new Feature;

    assert(fin != NULL && weak != NULL && weak->feat != NULL);

    ret = fread(&weak->thresh, sizeof(float), 1, fin);
    ret = fread(&weak->sign, sizeof(int), 1, fin);
    ret = fread(weak->feat, sizeof(Feature), 1, fin);

    *aWeak = weak;
}


void clear(WeakClassifier **weak)
{
    if((*weak)->feat != NULL){
        delete (*weak)->feat;
        (*weak)->feat = NULL;
    }

    delete *weak;
    *weak = NULL;
}
