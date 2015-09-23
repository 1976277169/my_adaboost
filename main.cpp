#include "tool.h"
#include "feature.h"
#include "weak_classifier.h"
#include "strong_classifier.h"
#include "cascade_classifier.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#define ADD_MIRROR_SAMPLE
//#define ADD_ROTATE_SAMPLE

int generate_positive_samples(const char *fileName, std::list<float *> &positiveSet, int width, int height)
{
    FILE *fin = fopen(fileName, "r");

    if(fin == NULL){
        printf("Can't open file %s\n", fileName);
        return 0;
    }

    int h, w, size, ret;
    int sq = width * height;

    ret = fread(&w, sizeof(int), 1, fin);
    ret = fread(&h, sizeof(int), 1, fin);

    if(w != width || h != height)
    {
        printf("Sample resolution error\n");
        return 0;
    }

    ret = fread(&size, sizeof(int), 1, fin);

    for(int i = 0; i < size; i++)
    {
        float *data = new float[sq];
        ret = fread(data, sizeof(float), sq, fin);

        if(ret == 0)
        {
            delete[] data;
            return 0;
        }


//        normalize_image(data, width, height);
        positiveSet.push_back(data);
    }


#ifdef ADD_ROTATE_SAMPLE
    add_rotated_images(positiveSet, size, width, height);
#endif


#ifdef ADD_MIRROR_SAMPLE
    add_vertical_mirror(positiveSet, size, width, height);
#endif

    std::list<float *>::iterator iter = positiveSet.begin();
    std::list<float *>::iterator iterEnd = positiveSet.end();

    while(iter != iterEnd)
    {
        integral_image((*iter), width, height);
        iter ++;
    }

    return positiveSet.size();
}


int generate_valid_samples(FILE *fin, std::list<float*> &validateSet, int width, int height, int size)
{
    if(fin == NULL)
        return 0;

    int sq = width * height;

    for(int i = 0; i < size; i++)
    {
        float *data = new float[sq];
        int ret = fread(data, sizeof(float), sq, fin);

        if(ret == 0)
        {
            delete[] data;
            return 0;
        }

//        normalize_image(data, width, height);

        validateSet.push_back(data);
    }

    std::list<float *>::iterator iter = validateSet.begin();
    std::list<float *>::iterator iterEnd = validateSet.end();

    while(iter != iterEnd)
    {
        integral_image((*iter), width, height);
        iter ++;
    }

    return validateSet.size();
}


int generate_negative_samples(FILE* fin, std::list<float *> &negativeSet, int width, int height, int size, CascadeClassifier *cc)
{
    int sq = width * height;

    while(negativeSet.size() < size)
    {
        float *data = new float[sq];
        int ret = fread(data, sizeof(float), sq, fin);

        if(ret == 0)
        {
            printf("Can't read enough negative samples\n");
            delete[] data;
            return 0;
        }

//        normalize_image(data, width, height);

        std::list<float *> tmplist;
        tmplist.push_back(data);

        std::list<float*>::iterator iter = tmplist.begin();
        std::list<float*>::iterator iterEnd = tmplist.end();

        while(iter != iterEnd)
        {
            integral_image(*iter, width, height);
            if(classify(cc, *iter, width, 0, 0) == 0)
                negativeSet.push_back(*iter);
            else
                delete[] (*iter);

            iter++;
        }

        tmplist.clear();
    }

    return negativeSet.size();
}


StrongClassifier* adaboost_learning(CascadeClassifier *cc, std::list<float *> &positiveSet,
            std::list<float *> &negativeSet, std::list<float *> &validateSet, std::vector<Feature *> &featureSet,
            float maxfpr, float maxfnr)
{
    StrongClassifier *sc = new StrongClassifier;

    int width = cc->WIDTH;
    int height = cc->HEIGHT;
    float *weights = NULL, *values = NULL;
    int numPos = positiveSet.size();
    int numNeg = negativeSet.size();
    int sampleSize = numPos + numNeg;
    int fsize = featureSet.size();

    float cfpr = 1.0;

    printf("maxfpr: %f, maxfnr: %f\n\n", maxfpr, maxfnr);

    init_weights(&weights, numPos, numNeg);

    values = new float[sampleSize];
    memset(values, 0, sizeof(float) * sampleSize);

    while(cfpr > maxfpr)
    {
        std::list<float *>::iterator iter;
        float minError = 1, error;
        WeakClassifier *bestWC = new WeakClassifier;

        if(!empty(sc))
        {
            float strongFpr = fpr(sc, negativeSet, width);
            printf("Strong classifier fpr = %f\n", strongFpr);
            if(strongFpr == 0)
                break;
        }

        update_weights(weights, sampleSize);

        for(int i = 0; i < fsize; i++)
        {
            Feature *feat = featureSet[i];
            WeakClassifier *wc = new WeakClassifier;
            init_weak_classifier(wc, 0, 0, feat);

            iter = positiveSet.begin();
            for(int j = 0; j < numPos; j++, iter++)
                values[j] = get_value(feat, *iter, width, 0, 0);

            iter = negativeSet.begin();
            for(int j = 0; j < numNeg; j++, iter++)
                values[j + numPos] = get_value(feat, *iter, width, 0, 0);

            error = train(wc, values, numPos, numNeg, weights);

            if(error < minError)
            {
                delete [] bestWC;
                bestWC = wc;
                minError = error;

                printf("%f\r", minError);
                fflush(stdout);
            }
            else
                delete [] wc;
        }

        printf("best weak classifier error = %f\n", minError);

        float beta = minError / (1 - minError);

        iter = positiveSet.begin();
        for(int i = 0; i < numPos; i++, iter++)
            if(classify(bestWC, *iter, width, 0, 0) == 1)
                weights[i] *= beta;

        iter = negativeSet.begin();
        for(int i = numPos; i < sampleSize; i++, iter++)
            if(classify(bestWC, *iter, width, 0, 0) != 1)
                weights[i] *= beta;

        add(sc, bestWC, log(1/beta));

        train(sc, positiveSet, width, maxfnr);

        add(cc, sc);
        cfpr = fpr(cc, validateSet, width);
        printf("fpr validate: %f\n\n", cfpr);
        del(cc);
    }

    printf("\nWeak classifier size %ld\n", sc->wcs.size());

    delete [] values;
    delete [] weights;

    return sc;
}


int main_train(int argc, char **argv);
int main_detect(int argc, char **argv);
int main_generate_samples(int argc, char **argv);

int main(int argc, char **argv)
{
#if defined(TRAIN_MODEL)
    main_train(argc, argv);

#elif defined(DETECT)
    main_detect(argc, argv);

#elif defined(GENERATE_SAMPLE)
    main_generate_samples(argc, argv);

#else
    printf("Please compile with macro TRAIN_MODEL DETECT GENERATE_SAMPLE\n");

#endif
    return 0;
}


void print_train_usage(char *proc)
{
    printf("Usage: %s\n", proc);
    printf("    --stage <number of stage\n");
    printf("    -X <sample width>\n");
    printf("    -Y <sample height>\n");
    printf("    --false_alarm_rate <target false alarm rate>\n");
    printf("    --missing_rate <max missing rate>\n");
    printf("    --pos <pos sample>\n");
    printf("    --neg <neg sample>\n");
    printf("    -m <output model file>\n");
}


int main_train(int argc, char **argv)
{
    for(int i = 0; i < argc; i ++)
        printf("%s ", argv[i]);

    printf("\n");

    char *posSplFile = NULL;
    char *negSplFile = NULL;
    char *modelFile = NULL;

    int stage = 15;
    int width = 0;
    int height = 0;

    float tarfpr = 0.05;
    float maxfnr = 0.05;

    if((argc - 1) / 2 != 8)
    {
        print_train_usage(argv[0]);
        return 1;
    }

    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "--stage") == 0)
            stage = atoi(argv[++i]);

        else if(strcmp(argv[i], "-X") == 0)
            width = atoi(argv[++i]);

        else if(strcmp(argv[i], "-Y") == 0)
            height = atoi(argv[++i]);

        else if(strcmp(argv[i], "--false_alarm_rate") == 0)
            tarfpr = atof(argv[++i]);

        else if(strcmp(argv[i], "--missing_rate") == 0)
            maxfnr = atof(argv[++i]);

        else if(strcmp(argv[i], "--pos") == 0)
            posSplFile = argv[++i];

        else if(strcmp(argv[i], "--neg") == 0)
            negSplFile = argv[++i];

        else if(strcmp(argv[i], "-m") == 0)
            modelFile = argv[++i];

        else
        {
            printf("Can't recognize params %s\n", argv[i]);
            print_train_usage(argv[0]);
            return 1;
        }
    }

    if(posSplFile == NULL || negSplFile == NULL || width == 0 || height == 0){
        print_train_usage(argv[0]);
        return 1;
    }

    std::list<float *> positiveSet, negativeSet, validateSet;
    FILE *fin;
    int ret;
    std::vector<Feature*> featureSet;
    float *stepFPR;
    int trainnigSampleSize = 0;
    int sq = width * height;

    CascadeClassifier *cc = new CascadeClassifier();
    StrongClassifier* sc;
    float maxfpr = 1.0;

    printf("Init cascade classifier\n");

    {
        std::list<StrongClassifier *> scs;
        init_cascade_classifier(cc, scs, width, height);
    }

    printf("Generate positive samples and validate samples\n");

    {
        int nw, nh, nsize;

        ret = generate_positive_samples(posSplFile, positiveSet, width, height);
        if(ret == 0) return 1;

        fin = fopen(negSplFile, "rb");
        if(ret == 0) return 1;

        ret = fread(&nw, 1, sizeof(int), fin);
        ret = fread(&nh, 1, sizeof(int), fin);
        ret = fread(&nsize, 1, sizeof(int), fin);

        assert(nw == width && nh == height);

        ret = generate_valid_samples(fin, validateSet, width, height, positiveSet.size());

        generate_feature_set(featureSet, width, height);

        printf("pos size %ld\n", positiveSet.size());
    }

    init_steps_false_positive(&stepFPR, stage, tarfpr);
    trainnigSampleSize = positiveSet.size();

    for(int i = 0; i < stage; i++)
    {
        printf("\n--------------stage %d-----------------\n", i);

        ret = generate_negative_samples(fin, negativeSet, width, height, trainnigSampleSize, cc);

        if(ret == 0) break;

        maxfpr *= stepFPR[i];

        sc = adaboost_learning(cc, positiveSet, negativeSet, validateSet, featureSet, maxfpr, maxfnr);
        add(cc, sc);

        std::list<float*>::iterator iter = positiveSet.begin();
        std::list<float*>::iterator iterEnd = positiveSet.end();

        while(iter != iterEnd)
        {
            if(classify(cc, *iter, width, 0, 0) == 0)
            {
                std::list<float*>::iterator iterTmp = iter;
                iter++;
                delete[] (*iterTmp);
                positiveSet.erase(iterTmp);
                iter--;
            }

            iter++;
        }


        iter = negativeSet.begin();
        iterEnd = negativeSet.end();

        while(iter != iterEnd)
        {
            if(classify(cc, *iter, width, 0, 0) == 1)
            {
                std::list<float*>::iterator iterTmp = iter;
                iter++;
                delete[] (*iterTmp);
                negativeSet.erase(iterTmp);
                iter--;
            }

            iter++;
        }


        printf("TP: %ld\n", positiveSet.size());
        printf("FP: %ld\n", negativeSet.size());

        if(negativeSet.size() == trainnigSampleSize)
        {
            int nsize = trainnigSampleSize * 0.1;

            for(int n = 0; n < nsize; n++){
                float *data = negativeSet.front();
                delete[] data;
                negativeSet.pop_front();
            }
        }

        printf("----------------------------------------\n");
    }

    save(cc, modelFile);
    clear(cc);

    clear_list(positiveSet);
    clear_list(negativeSet);
    clear_list(validateSet);

    return 0;
}


int main_detect(int argc, char **argv)
{

    return 0;
}


int main_generate_samples(int argc, char **argv)
{
    if(argc < 3)
    {
        printf("Usage: %s [flag] [sample image list] [WIDTH] [HEIGHT] [size]\n", argv[0]);
        return 1;
    }

    int flag, width, height, ssize, ret;
    std::vector<std::string> imageList;

    flag = atoi(argv[1]);
    ret = read_image_list(argv[2], imageList);
    if(ret != 0) return 1;

    width = atoi(argv[3]);
    height = atoi(argv[4]);
    ssize = atoi(argv[5]);

    if(flag == 0)
    {
        FILE *fout = fopen("neg_sample.bin", "w");
        int size = imageList.size();

        fwrite(&width, sizeof(int), 1, fout);
        fwrite(&height, sizeof(int), 1, fout);
        fwrite(&ssize, sizeof(int), 1, fout);

        float *fdata = new float[width];

        for(int i = 0, j = 0; i < size && j < ssize; i++)
        {
            cv::Mat img = cv::imread(imageList[i], 0);
            int w = img.cols;
            int h = img.rows;

            if(img.empty())
            {
                printf("Can't open image %s\n", imageList[i].c_str());
                exit(0);
            }

            w -= width;
            h -= height;

            for(int y = 0; y < h; y += height)
            {
                for(int x = 0; x < w; x += width, j++)
                {
                    cv::Mat part(img, cv::Rect(x, y, width, height));
                    uchar *data = part.data;

                    for(int m = 0; m < height; m++){
                        for(int n = 0; n < width; n++)
                            fdata[n] = data[n] / 255.0;

                        fwrite(fdata, sizeof(float), width, fout);
                        data += part.step;
                    }
                }
            }
        }

        delete [] fdata;
        fclose(fout);
    }
    else
    {
        FILE *fout = fopen("pos_sample.bin", "w");
        int size = imageList.size();
        if(size > ssize) size = ssize;

        fwrite(&width, sizeof(int), 1, fout);
        fwrite(&height, sizeof(int), 1, fout);
        fwrite(&size, sizeof(int), 1, fout);

        float *fdata =new float[width];

        for(int i = 0; i < size; i++)
        {
            cv::Mat img =  cv::imread(imageList[i], 0);
            if(img.empty())
            {
                printf("Can't read image %s\n", imageList[i].c_str());
                break;
            }
            if(img.cols != width && img.rows != height)
                cv::resize(img, img, cv::Size(width, height));

            uchar *data = img.data;

            for(int y = 0; y < height; y++)
            {
                for(int i = 0; i < width; i++)
                    fdata[i] = data[i] / 255.0;

                fwrite(fdata, sizeof(float), width, fout);
                data += img.step;
            }
        }

        delete[] fdata;

        fclose(fout);
    }

    return 0;
}
