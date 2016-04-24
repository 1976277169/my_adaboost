#ifndef _CASCADE_CLASSIFIER_H_
#define _CASCADE_CLASSIFIER_H_

#include "strong_classifier.h"
#include <list>

typedef struct {
    std::vector<StrongClassifier*> scs;
    int WIDTH, HEIGHT;
} CascadeClassifier;

void init_cascade_classifier(CascadeClassifier *cascade, std::vector<StrongClassifier*> scs, int WIDTH, int HEIGHT);

void add(CascadeClassifier *cascade, StrongClassifier *sc);
void del(CascadeClassifier *cascade);

int classify(CascadeClassifier *cascade, Sample *sample);
float fnr(CascadeClassifier *cc, std::vector<Sample*> &samples);
float fpr(CascadeClassifier *cc, std::vector<Sample*> &samples);

void clean_samples(CascadeClassifier *cc, std::vector<Sample*> &samples);

void load(CascadeClassifier **aCascade, const char *fileName);
void save(CascadeClassifier *cascade, const char *fileName);
void clear(CascadeClassifier **cascade);

#endif
