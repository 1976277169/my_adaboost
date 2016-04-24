#ifndef _WEAK_CLASSIFIER_H_
#define _WEAK_CLASSIFIER_H_

#define USE_HAAR_FEATURE

#include "sample.h"

#define POS_SAMPLE_FLAG 1
#define NEG_SAMPLE_FLAG -1


typedef struct {
    float thresh;
    int sign;
    Feature *feat;
} WeakClassifier;


void init_weak_classifier(WeakClassifier *weak, float thresh, int sign, Feature *feat);
float train(WeakClassifier *weak, float *value, int posSize, int negSize, float *weights);

int classify(WeakClassifier *weak, Sample *sample);
void save(WeakClassifier *weak, FILE *fout);
void load(WeakClassifier **aWeak, FILE *fin);
void clear(WeakClassifier **weak);

#endif
