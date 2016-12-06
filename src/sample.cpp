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


int read_samples(const char *fileList, int ssize, int WINW, int WINH, SampleSet **posSet){
    std::vector<std::string> imgList;
    char rootDir[256], fileName[256], ext[30];

    if(read_file_list(fileList, imgList) != 0)
        return 0;

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
        sprintf(sample->patchName, "%s.jpg", fileName);

        resizer_bilinear_gray(img.data, img.cols, img.rows, img.step,
                sample->img, WINW, WINH, WINW);

        memset(sample->iImgBuf, 0, sizeof(uint32_t) * (WINW + 1) * (WINH + 1));

        integral_image(sample->img, WINW, WINH, WINW, sample->iImg, WINW + 1);

        (*posSet)->samples[i] = sample;
    }

    return ssize;
}

