#ifndef _WEAK_CLASSIFIER_H_
#define _WEAK_CLASSIFIER_H_

#define USE_HAAR_FEATURE

#if defined(USE_HAAR_FEATURE)
#include "feature.h"

#else
#include "feature.h"

#endif


typedef struct {
    float thresh;
    int sign;
    Feature *feat;
} WeakClassifier;


void init_weak_classifier(WeakClassifier *weak, float thresh, int sign, Feature *feat);
float train(WeakClassifier *weak, float *value, int posSize, int negSize, float *weights);
int classify(WeakClassifier *weak, float *img, int stride, int x, int y);

void save(WeakClassifier *weak, FILE *fout);
void load(WeakClassifier **aWeak, FILE *fin);
void clear(WeakClassifier *weak);

#endif
