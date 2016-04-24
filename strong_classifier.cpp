#include "strong_classifier.h"
#include "tool.h"

#include <assert.h>

void init_strong_classifier(StrongClassifier *sc, std::vector<float> &aweights, std::vector<WeakClassifier*> &awcs, float thresh)
{
    int size = aweights.size();

    assert(aweights.size() == awcs.size());

    sc->weights.clear();
    sc->wcs.clear();

    for(int i = 0; i < size; i++)
    {
        sc->weights.push_back(aweights[i]);
        sc->wcs.push_back(awcs[i]);
    }

    sc->thresh= thresh;
}


void train(StrongClassifier *sc, std::vector<Sample*> &samples, float recall){
    int size = samples.size();

    float *scores = new float[size];
    int wcSize = sc->wcs.size();

    for(int i = 0; i < size; i++){
        scores[i] = 0;

        for(int j = 0; j < wcSize; j++){
            scores[i] += sc->weights[j] * classify(sc->wcs[j], samples[i]);
        }
    }

    sort_arr_float_ascend(scores, size);

    sc->thresh = scores[int(recall * size)] - 0.00001;
/*
    for(int i = 0; i < 10; i++)
        printf("%f ", scores[i]);
    printf("\n");
//*/
    delete [] scores;
}


int classify(StrongClassifier *sc, Sample *sample)
{
    int size = sc->wcs.size();
    float scores = 0;

    for(int i = 0; i < size; i++)
        scores += sc->weights[i] * classify(sc->wcs[i], sample);

    if(scores > sc->thresh)
        return 1;

    return 0;
}


void add(StrongClassifier *sc, WeakClassifier *wc, float weight)
{
    sc->wcs.push_back(wc);
    sc->weights.push_back(weight);
}


float fnr(StrongClassifier *sc, std::vector<Sample*> &samples){
    int size = samples.size();
    int fn = 0;

    for(int i = 0; i < size; i++)
        fn += (classify(sc, samples[i]) == 0);

    return float(fn) / size;
}


float fpr(StrongClassifier *sc, std::vector<Sample*> &samples){
    int size = samples.size();
    int fp = 0;

    for(int i = 0; i < size; i++)
        fp += (classify(sc, samples[i]) == 1);

    return float(fp) / size;
}


int empty(StrongClassifier *sc)
{
    if(sc == NULL || sc->wcs.size() == 0)
        return 1;

    return 0;
}


void save(StrongClassifier *sc, FILE* fout)
{
    int size = sc->wcs.size();

    fwrite(&size, sizeof(int), 1, fout);

    for(int i = 0; i < size; i++)
    {
        fwrite(&(sc->weights[i]), sizeof(float), 1, fout);
        save(sc->wcs[i], fout);
    }

    fwrite(&sc->thresh, sizeof(float), 1, fout);
}


void load(StrongClassifier **asc, FILE *fin)
{
    StrongClassifier *sc = new StrongClassifier;

    int size;

    int ret;

    ret = fread(&size, sizeof(int), 1, fin);

    sc->weights = std::vector<float>(size);
    sc->wcs = std::vector<WeakClassifier*>(size);

    for(int i = 0; i < size; i++)
    {
       ret = fread(&(sc->weights[i]), sizeof(float), 1, fin);
       load(&(sc->wcs[i]), fin);
    }

    ret = fread(&sc->thresh, sizeof(float), 1, fin);

    *asc = sc;
}


void clear(StrongClassifier **aSc)
{
    StrongClassifier *sc = *aSc;

    int size = sc->wcs.size();

    for(int i = 0; i < size; i++)
        clear(&sc->wcs[i]);

    sc->weights.clear();
    sc->wcs.clear();

    delete *aSc;

    *aSc = NULL;
}
