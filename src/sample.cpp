#include "sample.h"


void release_data(Sample *sample){
    if(sample->img != NULL)
        delete [] sample->img;
    sample->img = NULL;

    if(sample->iImgBuf != NULL)
        delete [] sample->iImgBuf;
    sample->iImgBuf = NULL;

    sample->iImg = NULL;
}


void release(Sample **sample){
    if(*sample != NULL){
        release_data(*sample);
        delete *sample;
        *sample = NULL;
    }
}


int read_samples(const char *fileList, int ssize, int WINW, int WINH, SampleSet **posSet){
    std::vector<std::string> imgList;
    char rootDir[256], fileName[256], ext[30];

    if(read_file_list(fileList, imgList) != 0)
        return 0;

    if(ssize <= 100)
        ssize = imgList.size();
    else
        ssize = HU_MIN(ssize, imgList.size());

    if(ssize <= 0) return 0;

    *posSet = new SampleSet;

    (*posSet)->winw = WINW;
    (*posSet)->winh = WINH;
    (*posSet)->ssize = ssize;
    (*posSet)->samples = new Sample*[ssize];

    memset((*posSet)->samples, 0, sizeof(Sample*) * ssize);

    for(int i = 0; i < ssize; i++){
        const char *imgPath = imgList[i].c_str();

        cv::Mat img = cv::imread(imgPath, 0);
        assert(!img.empty());

        analysis_file_path(imgPath, rootDir, fileName, ext);

        Sample *sample = new Sample;

        sample->img = new uint8_t[WINW * WINH];
        sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
        sample->iImg = sample->iImgBuf + (WINW + 1) + 1;
        sample->istride = WINW + 1;
        sample->stride = WINW;

        sprintf(sample->patchName, "%s", fileName);

        resizer_bilinear_gray(img.data, img.cols, img.rows, img.step,
                sample->img, WINW, WINH, WINW);

        memset(sample->iImgBuf, 0, sizeof(uint32_t) * (WINW + 1) * (WINH + 1));

        integral_image(sample->img, WINW, WINH, WINW, sample->iImg, WINW + 1);

        (*posSet)->samples[i] = sample;
    }

    (*posSet)->capacity = (*posSet)->ssize;

    return ssize;
}


void reserve(SampleSet *set, int size){
    if(set->capacity > size)
        return ;

    Sample **samples = new Sample*[size];

    if(set->ssize > 0){
        memcpy(set->samples, samples, sizeof(Sample*) * set->ssize);

        for(int i = set->ssize; i < size; i++){
            samples[i] = new Sample;
            memset(samples[i], 0, sizeof(Sample));
        }
    }
    else{
        for(int i = 0; i < size; i++){
            samples[i] = new Sample;
            memset(samples[i], 0, sizeof(Sample));
        }
    }

    if(set->samples != NULL)
        delete [] set->samples;

    set->samples = samples;
    set->capacity = size;
}


void split(SampleSet *src, float rate, SampleSet *res){
    cv::RNG rng(cv::getTickCount());

    int count = rate * src->ssize;

    if(count == 0) {
        res->ssize = 0;
        return;
    }

    res->winw = src->winw;
    res->winh = src->winh;
    res->ssize = 0;
    res->samples = new Sample*[count];

    for(int i = 0; i < count; i++){
        int id = rng.uniform(0, src->ssize);

        res->samples[i] = src->samples[id];

        src->samples[id] = src->samples[src->ssize - 1];

        src->ssize --;
        res->ssize ++;
    }

    memset(src->samples + src->ssize, 0, sizeof(Sample*) * (src->capacity - src->ssize));

    res->capacity = res->ssize;
}


void create_sample(Sample *sample, uint8_t *img, int width, int height, int stride, const char *patchName){
    if(sample->img == NULL)
        sample->img = new uint8_t[width * height];

    if(sample->iImgBuf == NULL)
        sample->iImgBuf = new uint32_t[(width + 1) * (height + 1)];

    sample->iImg = sample->iImgBuf + (width + 1) + 1;
    sample->istride = width + 1;

    strcpy(sample->patchName, patchName);

    for(int y = 0; y < height; y++)
        memcpy(sample->img + y * width, img + y * stride, sizeof(uint8_t) * width);

    integral_image(sample->img, width, height, stride, sample->iImg, sample->istride);
}


void random_subset(SampleSet *oriSet, SampleSet *subset, float rate){
    int size = oriSet->ssize * rate;

    cv::RNG rng(cv::getTickCount());

    assert(size > 0);

    subset->winw = oriSet->winw;
    subset->winh = oriSet->winh;

    reserve(subset, size);

    for(int i = 0; i < size; i++){
        int id = rng.uniform(0, oriSet->ssize);

        subset->samples[i] = oriSet->samples[id];

        HU_SWAP(oriSet->samples[oriSet->ssize - 1], oriSet->samples[id], Sample*);

        oriSet->samples[oriSet->ssize - 1] = NULL;
        oriSet->ssize --;
    }
}


void release_data(SampleSet *set){
    if(set != NULL){
        if(set->samples != NULL){
            for(int i = 0; i < set->capacity; i++)
                release_data(set->samples[i]);

            delete [] set->samples;
            set->samples = NULL;
        }

        set->ssize = 0;
        set->capacity = 0;
    }
}


void release(SampleSet **set){
    release_data(*set);

    delete *set;
    *set = NULL;
}


void print_info(SampleSet *set, const char *tag){
    printf("SampleSet: %s\n", tag);

    printf("ssize: %d, winw: %d, winh: %d\n", set->ssize, set->winw, set->winh);
    printf("capacity: %d\n", set->capacity);
}


void write_images(SampleSet *set, const char *outDir){
    int ssize = set->ssize;

    int WINW = set->winw;
    int WINH = set->winh;

    char outPath[256], command[256];

    sprintf(command, "mkdir -p %s", outDir);

    int ret = system(command);

    for(int i = 0; i < ssize; i++){
        Sample *sample = set->samples[i];
        cv::Mat img(WINH, WINW, CV_8UC1, sample->img, sample->stride);

        sprintf(outPath, "%s/%s.jpg", outDir, sample->patchName);
        if(!cv::imwrite(outPath, img))
            printf("Can't write image %s\n", outPath);

        printf("%d\r", i); fflush(stdout);
    }
}
