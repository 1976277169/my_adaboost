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


void train(StrongClassifier *sc, std::list<float *> &posSamples, int stride, float maxfnr)
{
    int size = posSamples.size();
    float *scores = new float[size];
    int idx = maxfnr * size;
    int wcsSize = sc->wcs.size();

    assert(0 <= maxfnr && maxfnr < 1);

    std::list<float *>::iterator iter = posSamples.begin();

    for(int i = 0; i < size; i++, iter++)
    {
        scores[i] = 0;

        for(int j = 0; j < wcsSize; j++)
            scores[i] += sc->weights[j] * classify(sc->wcs[j], *iter, stride, 0, 0);
    }

    sort_arr_float_ascend(scores, size);

    sc->thresh = scores[idx];

    delete [] scores;
}


int classify(StrongClassifier *sc, float *img, int stride, int x, int y)
{
    int size = sc->wcs.size();
    float scores = 0;

    for(int i = 0; i < size; i++)
        scores += sc->weights[i] * classify(sc->wcs[i], img, stride, x, y);

    if(scores >= sc->thresh)
        return 1;

    return 0;
}


void add(StrongClassifier *sc, WeakClassifier *wc, float weight)
{
    sc->wcs.push_back(wc);
    sc->weights.push_back(weight);
}


float fnr(StrongClassifier *sc, std::list<float*> &posSamples, int stride)
{
    int size = posSamples.size();
    int fn = 0;
    std::list<float*>::iterator iter = posSamples.begin();

    for(int i = 0; i < size; i++, iter++)
        fn += (classify(sc, *iter, stride, 0, 0) == 0);

    return float(fn) / size;
}


float fpr(StrongClassifier *sc, std::list<float*> &negSamples, int stride)
{
    int size = negSamples.size();
    int fp = 0;
    std::list<float*>::iterator iter = negSamples.begin();

    for(int i = 0; i < size; i++, iter++)
        fp += (classify(sc, *iter, stride, 0, 0) == 1);

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


void clear(StrongClassifier *sc)
{
    int size = sc->wcs.size();

    for(int i = 0; i < size; i++)
        clear(sc->wcs[i]);

    sc->weights.clear();
    sc->wcs.clear();

    delete sc;
}


#ifdef USE_HAAR_FEATURE
void print_feature(StrongClassifier *sc)
{
    int size = sc->wcs.size();

    for(int i = 0; i < size; i++)
    {
        Feature * feat = sc->wcs[i]->feat;

        printf("%d %2d %2d %2d %2d\n", feat->type, feat->x0, feat->y0, feat->w, feat->h);
    }
}
#endif
