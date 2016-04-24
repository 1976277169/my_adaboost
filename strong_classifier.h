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

void train(StrongClassifier *sc, std::vector<Sample*> &samples, float recall);
int classify(StrongClassifier *sc, Sample *sample);

float fnr(StrongClassifier *sc, std::vector<Sample*> &samples);
float fpr(StrongClassifier *sc, std::vector<Sample*> &samples);

int empty(StrongClassifier *sc);

void load(StrongClassifier **asc, FILE *fin);
void save(StrongClassifier *sc, FILE* fout);
void clear(StrongClassifier **sc);

#endif
