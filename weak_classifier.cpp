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
    int sampleSize = posSize + negSize;

    Scores *scores = new Scores[sampleSize];

    float *wp = new float[sampleSize];
    float *wn = new float[sampleSize];

    float tp, tn;

    float minError = 1;
    float errorPos = 0;
    float errorNeg = 0;

    int i, j;


    for(i = 0; i < posSize; i++)
    {
        scores[i].value = value[i];
        scores[i].label = 1;
        scores[i].weight = weights[i];
    }

    for(; i < sampleSize; i++)
    {
        scores[i].value = value[i];
        scores[i].label = 0;
        scores[i].weight = weights[i];
    }

    sort_arr_scores(scores, sampleSize);

    if(scores[0].label == 1)
    {
        wp[0] = 0;
        wn[0] = scores[0].weight;
    }
    else
    {
        wn[0] = scores[0].weight;
        wp[0] = 0;
    }

    for(i = 1; i < sampleSize; i++)
    {
        if(scores[i].label == 1)
        {
            wp[i] = wp[i - 1] + scores[i].weight;
            wn[i] = wn[i - 1];
        }
        else
        {
            wp[i] = wp[i - 1];
            wn[i] = wn[i - 1] + scores[i].weight;
        }
    }

    tp = wp[sampleSize - 1];
    tn = wn[sampleSize - 1];


    for(i = 0; i < sampleSize; i++)
    {
        for(j = i + 1; j < sampleSize; j++)
        {
            if(scores[i].value != scores[j].value)
                break;
        }

        i = j - 1;

        errorPos = wp[i] + tn - wn[i];
        errorNeg = wn[i] + tp - wp[i];

        //printf("value: %f, errorPos: %f, errorNeg: %f\n", scores[i].value, errorPos, errorNeg);
        if(errorPos < minError && errorPos < errorNeg)
        {
            minError = errorPos;
            weak->sign = 1;
            weak->thresh = scores[i].value;
        }
        else if(errorNeg < minError && errorNeg < errorPos)
        {
            minError = errorNeg;
            weak->sign = 0;
            weak->thresh = scores[i].value;
        }
    }

    delete[] scores;
    delete[] wp;
    delete[] wn;

    return minError;
}


int classify(WeakClassifier *weak, float *img, int stride, int x, int y)
{
    float value = get_value(weak->feat, img, stride, x, y);
    float thresh = weak->thresh;
    float sign = weak->sign;

    if( (value > thresh && sign == 1) || (value <= thresh && sign == 0) )
        return 1;

    else
        return -1;
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


void clear(WeakClassifier *weak)
{
    delete weak->feat;
    delete weak;
    weak = NULL;
}
