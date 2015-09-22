#ifndef _WEAK_CLASSIFIER_H_
#define _WEAK_CLASSIFIER_H_

#include "feature.h"


typedef struct {
    float thresh;
    int sign;
    Feature *feat;
} WeakClassifier;


void init_weak_classifier(WeakClassifier *weak, float thresh, int sign, Feature *feat);
float train(WeakClassifier *weak, float *value, int posSize, int negSize, float *weights);
int classify(WeakClassifier *weak, float *img, int stride, int x, int y);

#endif
