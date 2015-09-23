#ifndef _STRONG_CLASSIFIER_
#define _STRONG_CLASSIFIER_

#include "weak_classifier.h"

#include <list>
#include <vector>

typedef struct{
    std::vector<float> weights;
    std::vector<WeakClassifier*> wcs;

    float thresh;
} StrongClassifier;


void init_strong_classifier(StrongClassifier *sc, std::vector<float> &aweights, std::vector<WeakClassifier*> &awcs, float thresh);
void add(StrongClassifier *sc, WeakClassifier *wc, float weight);

void train(StrongClassifier *sc, std::list<float *> &posSamples, int stride, float maxfnr);
int classify(StrongClassifier *sc, float *img, int stride, int x, int y);

float fnr(StrongClassifier *sc, std::list<float*> &posSamples, int stride);
float fpr(StrongClassifier *sc, std::list<float*> &negSamples, int stride);

int empty(StrongClassifier *sc);
void load(StrongClassifier **asc, FILE *fin);
void save(StrongClassifier *sc, FILE* fout);
void clear(StrongClassifier *sc);

#endif
