#ifndef _SAMPLE_H_
#define _SAMPLE_H_

#include "tool.h"


typedef struct {
    uint8_t *img;
    int stride;

    uint32_t *iImgBuf;
    uint32_t *iImg;
    int istride;

    float score;

    char patchName[100];
} Sample;


void release_data(Sample *sample);
void release(Sample **sample);


typedef struct{
    Sample **samples;
    int ssize;
    int winw;
    int winh;
} SampleSet;


int read_samples(const char *fileList, int ssize);


#endif
