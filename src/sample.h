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

    int capacity;
} SampleSet;


int read_samples(const char *fileList, int ssize, int WINW, int WINH, SampleSet **posSet);
void reserve(SampleSet *set, int size);

void create_sample(Sample *sample, uint8_t *img, int width, int height, int stride, const char *patchName);

void split(SampleSet *src, float rate, SampleSet *res);

void write_images(SampleSet *set, const char *outDir);

void release_data(Sample *sample);
void release_data(SampleSet *set);
void release(SampleSet **set);

void print_info(SampleSet *set, const char *tag);
#endif
