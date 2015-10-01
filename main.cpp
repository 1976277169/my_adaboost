#include "tool.h"
#include "feature.h"
#include "weak_classifier.h"
#include "strong_classifier.h"
#include "cascade_classifier.h"


#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#define ADD_MIRROR_SAMPLE
//#define ADD_ROTATE_SAMPLE

int generate_positive_samples(FILE *fin, std::list<float*> &positiveSet, int width, int height, int size)
{
    assert(fin != NULL);

    int ret;
    int sq = width * height;
    int count = size - positiveSet.size();

    for(int i = 0; i < count; i++)
    {
        float *data = new float[sq];

        ret = fread(data, sizeof(float), sq, fin);

        if(ret == 0){
            printf("Can't read enough positive samples\n");
            delete []data;
            return 0;
        }

        positiveSet.push_back(data);

#ifdef ADD_ROTATE_SAMPLE
        {
            float *t = rotate_90deg(data, width, height);
            integral_image(t, width, height);
            positiveSet.push_back(t);

            t = rotate_180deg(data, width, height);
            integral_image(t, width, height);
            positiveSet.push_back(t);

            t = rotate_270deg(data, width, height);
            integral_image(t, width, height);
            positiveSet.push_back(t);
        }
#endif

#ifdef ADD_MIRROR_SAMPLE
        {
            float *t = vertical_mirror(data, width, height);
            integral_image(t, width, height);
            positiveSet.push_back(t);
        }
#endif

        integral_image(data, width, height);
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

        integral_image(data, width, height);
        validateSet.push_back(data);
    }

    return validateSet.size();
}


int generate_negative_samples(FILE* fin, std::list<float *> &negativeSet, int width, int height, int size, CascadeClassifier *cc)
{
    int sq = width * height;

    while(negativeSet.size() < size)
    {

        std::list<float *> tmplist;
        std::list<float*>::iterator iter, iterEnd;

        int count = size - negativeSet.size();

        for(int i = 0; i < count; i++)
        {
            float *data = new float[sq];
            int ret = fread(data, sizeof(float), sq, fin);

            if(ret == 0)
            {
                printf("Can't read enough negative samples\n");
                delete[] data;
                return 0;
            }

            integral_image(data, width, height);
            tmplist.push_back(data);
        }

        iter = tmplist.begin();
        iterEnd = tmplist.end();

        while(iter != iterEnd)
        {

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


void select_feature(std::vector<Feature*> &featureSet,
        std::list<float*> &positiveSet, std::list<float *> &negativeSet, int width)
{
    WeakClassifier *wc = new WeakClassifier;

    std::list<float *>::iterator iterPos, iterNeg;

    int numPos = positiveSet.size();
    int numNeg = negativeSet.size();
    int dPos = numPos / 100;
    int dNeg = numNeg / 100;
    int sampleSize = 200;
    int count = dPos < dNeg ? dPos : dNeg;

    float *values = new float[sampleSize];
    float *weights = new float[sampleSize];

    for(int i = 0; i < sampleSize; i++)
        weights[i] = 0.005;

    iterPos = positiveSet.begin();
    iterNeg = negativeSet.begin();

    assert(count > 1);

#ifdef SHOW_FEATURE
    FILE *fout = fopen("feature.txt", "w");
#endif

    for(int i = 0; i < count; i++)
    {
        int fsize = featureSet.size();
        int top = 0;
        std::list<float *>::iterator iterPos2, iterNeg2;

        for(int j = 0; j < fsize; j++)
        {
            init_weak_classifier(wc, 0, 0, featureSet[j]);

            iterPos2 = iterPos;
            iterNeg2 = iterNeg;

            for(int k = 0; k < 100; k++){
                values[k] = get_value(featureSet[j], *iterPos2, width, 0, 0);
                values[k + 100] = get_value(featureSet[j], *iterNeg2, width, 0, 0);
                iterPos2++; iterNeg2++;
            }

            float error = train(wc, values, 100, 100, weights);

#ifdef SHOW_FEATURE
            fprintf(fout, "%d %2d %2d %2d %2d %f\n", featureSet[j]->type, featureSet[j]->x0, featureSet[j]->y0, featureSet[j]->w, featureSet[j]->h, error);
#endif

            if(error < 0.3){
                featureSet[top++] = featureSet[j];
            }

            else{
                delete featureSet[j];
                featureSet[j] = NULL;
            }
        }

        iterPos = iterPos2;
        iterNeg = iterNeg2;

#ifdef SHOW_FEATURE
        fprintf(fout, "\n");
#endif

        featureSet.erase(featureSet.begin() + top, featureSet.end());
        printf("Select feature template %d, size = %d\r", i, top);
        fflush(stdout);
    }

    printf("Fine weak classifier size: %ld        \n", featureSet.size());

#ifdef SHOW_FEATURE
    fclose(fout);
#endif

    delete[] values;
    delete[] weights;
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

    init_weights(&weights, numPos, numNeg);

    values = new float[sampleSize];
    memset(values, 0, sizeof(float) * sampleSize);

    while(cfpr > maxfpr)
    {
        std::list<float *>::iterator iter;
        float minError = 1, error, beta;
        WeakClassifier *bestWC = NULL;

        for(int i = 0; i < fsize; i++)
        {
            Feature *feat = new Feature;
            WeakClassifier *wc = new WeakClassifier;

            init_feature(feat, featureSet[i]->type, featureSet[i]->x0, featureSet[i]->y0, featureSet[i]->w, featureSet[i]->h);
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
                if(bestWC != NULL){
                    clear(bestWC);
                    bestWC = NULL;
                }

                bestWC = wc;

                minError = error;

                printf("Select best weak classifier, min error: %f\r", minError);
                fflush(stdout);
            }

            else
                delete wc;
        }

        printf("best weak classifier error = %f                      \n", minError);

        beta = minError / (1 - minError);

        int tp = 0;
        iter = positiveSet.begin();

        for(int i = 0; i < numPos; i++, iter++){
            if(classify(bestWC, *iter, width, 0, 0) == 1){
                weights[i] *= beta;
                tp ++;
            }
        }

        int tn = 0;
        iter = negativeSet.begin();

        for(int i = numPos; i < sampleSize; i++, iter++){
            if(classify(bestWC, *iter, width, 0, 0) != 1){
                weights[i] *= beta;
                tn++;
            }
        }

        update_weights(weights, numPos, numNeg);

        printf("TP = %d, TN = %d, beta = %f, log(1/beta) = %f\n", tp, tn, beta, log(1/beta));

        add(sc, bestWC, log(1/beta));

        train(sc, positiveSet, width, maxfnr);

        cfpr = fpr(sc, validateSet, width);

        printf("fpr validate: %f\n", fpr(sc, validateSet, width));
        printf("fpr negative: %f\n", fpr(sc, negativeSet, width));
        printf("\n");
    }

    printf("\nWeak classifier size %ld\n", sc->wcs.size());


    delete [] values;
    delete [] weights;

    return sc;
}


int main_train(int argc, char **argv);
int main_detect(int argc, char **argv);
int main_generate_samples(int argc, char **argv);

int main_test_weak_classifier(int argc, char **argv);

int main(int argc, char **argv)
{
#if defined(TRAIN_MODEL)
    main_train(argc, argv);

#elif defined(DETECT)
    main_detect(argc, argv);

#elif defined(GENERATE_SAMPLE)
    main_generate_samples(argc, argv);

#else
    main_test_weak_classifier(argc, argv);
    //printf("Please compile with macro TRAIN_MODEL DETECT GENERATE_SAMPLE\n");

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
    printf("    --pos <positive sample set file>\n");
    printf("    --numPos <number of positive sample used to train\n");
    printf("    --neg <negative sample set file>\n");
    printf("    --numNeg <number of negative sample used to train\n");
    printf("    -m <output model file>\n");
}


void print_detect_usage(char *proc)
{
    printf("Usage: %s\n", proc);
    printf("    -m <model file>\n");
    printf("    --scaleStep <scale step>\n");
    printf("    --slideStep <slide step>\n");
    printf("    -i <input image>\n");
    printf("    -o <output image>\n");
}


int main_train(int argc, char **argv)
{
    char *posSplFile = NULL;
    char *negSplFile = NULL;
    char *modelFile = NULL;

    int stage = 15;
    int width = 0, height = 0;
    int numPos = 0, numNeg = 0;

    float tarfpr = 0.05;
    float maxfnr = 0.05;

    if((argc - 1) / 2 != 10)
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

        else if(strcmp(argv[i], "--numPos"))
            numPos = atoi(argv[++i]);

        else if(strcmp(argv[i], "--numNeg"))
            numNeg = atoi(argv[++i]);

        else
        {
            printf("Can't recognize params %s\n", argv[i]);
            print_train_usage(argv[0]);
            return 1;
        }
    }

    if(posSplFile == NULL || negSplFile == NULL || width == 0 || height == 0 || numPos <= 0 || numNeg <= 0){
        print_train_usage(argv[0]);
        return 1;
    }

    std::list<float *> positiveSet, negativeSet, validateSet;
    FILE *nfin, *pfin;
    int ret;
    std::vector<Feature*> featureSet;
    float *stepFPR;

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

        pfin = fopen(posSplFile, "rb");
        if(pfin == NULL)
        {
            printf("Can't open file %s\n", posSplFile);
            return 2;
        }
        ret = fread(&nw, 1, sizeof(int), pfin);
        ret = fread(&nh, 1, sizeof(int), pfin);
        ret = fread(&nsize, 1, sizeof(int), pfin);

        assert(nw == width && nh == height && nsize > numPos);

        ret = generate_positive_samples(pfin, positiveSet, width, height, numPos);

        if(ret == 0)
            return 3;

    }

    {
        int nw, nh, nsize;

        nfin = fopen(negSplFile, "rb");
        if(nfin == NULL)
        {
            printf("Can't open file %s\n", negSplFile);
            return 2;
        }

        ret = fread(&nw, 1, sizeof(int), nfin);
        ret = fread(&nh, 1, sizeof(int), nfin);
        ret = fread(&nsize, 1, sizeof(int), nfin);

        assert(nw == width && nh == height && nsize > numNeg * 10);

        ret = generate_valid_samples(nfin, validateSet, width, height, numNeg * 3);

        if(ret == 0)
            return 3;
    }

    printf("Positive sample size: %d\n", numPos);
    printf("Negative sample size: %d\n", numNeg);
    printf("Validate sample size: %ld\n", validateSet.size());

    generate_feature_set(featureSet, width, height);
    select_feature(featureSet, positiveSet, validateSet, width);

    init_steps_false_positive(&stepFPR, stage, tarfpr);

    for(int i = 0; i < stage; i++)
    {
        printf("\n--------------cascade stage %d-----------------\n", i+1);

        int correctSize = 0;

        std::list<float*>::iterator iter, iterEnd;

        ret = generate_positive_samples(pfin, positiveSet, width, height, numPos);
        if(ret == 0) break;

        ret = generate_negative_samples(nfin, negativeSet, width, height, numNeg, cc);
        if(ret == 0) break;


        maxfpr *= stepFPR[i];

        printf("Target false positive rate: %f\n", maxfpr);
        printf("Target false negative rate: %f\n", maxfnr);

        sc = adaboost_learning(cc, positiveSet, negativeSet, validateSet, featureSet, maxfpr, maxfnr);
        add(cc, sc);

        iter = positiveSet.begin();
        iterEnd = positiveSet.end();

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

        correctSize = positiveSet.size();
        printf("cascade TP: %d\n", correctSize);

/*
        correctSize -= 0.95 * numPos;
        if(correctSize > 0)
        {
            for(int n = 0; n < correctSize; n++)
            {
                float *t = positiveSet.front();
                delete [] t;
                positiveSet.pop_front();
            }
        }

*/
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


        correctSize = negativeSet.size();
        printf("cascade FP: %d\n", correctSize);

        correctSize -= 0.8 * numNeg;

        if(correctSize > 0)
        {

            for(int n = 0; n < correctSize; n++)
            {
                float *t = negativeSet.front();
                delete [] t;
                negativeSet.pop_front();
            }
        }

        printf("----------------------------------------\n");

        save(cc, modelFile);

#ifdef SHOW_FEATURE
        print_feature(cc);
#endif
    }

    fclose(pfin);
    fclose(nfin);
    clear(cc);

    clear_list(positiveSet);
    clear_list(negativeSet);
    clear_list(validateSet);

    clear_features(featureSet);

    return 0;
}


void detect_object2(CascadeClassifier *cc, cv::Mat &img, float scaleStep, float slideStep, std::vector<cv::Rect> &rects)
{
    cv::Mat gray, sImg;

    int width = img.cols;
    int height = img.rows;

    int WIDTH = cc->WIDTH;
    int HEIGHT = cc->HEIGHT;

    int dx = slideStep * WIDTH;
    int dy = slideStep * HEIGHT;

    float *data = new float[WIDTH * HEIGHT];
    float cumScale = 1.0f;

    if(img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();


    for(int i = 0; i < 5; i++)
    {
        int dh = height - HEIGHT;
        int dw = width - WIDTH;

        cv::Mat flag(height, width, CV_8UC1, cv::Scalar(0));

        for(int y = 0; y < dh; y += dy)
        {
            for(int x = 0; x < dw; x += dx)
            {
                if(flag.at<uchar>(y, x) == 1) continue;

                cv::Mat part(gray, cv::Rect(x, y, WIDTH, HEIGHT));

                uchar *pImg = part.data;
                float *pData = data;

                for(int j = 0; j < HEIGHT; j++)
                {
                    for(int i = 0; i < WIDTH; i++)
                        pData[i] = pImg[i] / 255.0;

                    pData += WIDTH;
                    pImg += part.step;
                }

//                normalize_image_npd(data, WIDTH, HEIGHT);
                integral_image(data, WIDTH, HEIGHT);

                if(classify(cc, data, WIDTH, 0, 0) == 1){
                    cv::Mat part(flag, cv::Rect(x, y, WIDTH, HEIGHT));
                    part = cv::Scalar(1);
                    rects.push_back(cv::Rect(x * cumScale, y * cumScale, WIDTH * cumScale, HEIGHT * cumScale));
                }
            }
        }

        width /= scaleStep;
        height /= scaleStep;
        cumScale *= scaleStep;

        cv::resize(gray, gray, cv::Size(width, height));
        printf("%d\r", i);
        fflush(stdout);
    }

    delete [] data;
}


void detect_object(CascadeClassifier *cc, cv::Mat &img, float scaleStep, float slideStep, std::vector<cv::Rect> &rects)
{
    cv::Mat gray;
    int width = img.cols;
    int height = img.rows;

    int WIDTH = cc->WIDTH;
    int HEIGHT = cc->HEIGHT;

    int baseW = WIDTH;//scaleStep;
    int baseH = HEIGHT;//scaleStep;


    if(img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    float *data = new float[WIDTH * HEIGHT];

    for(int i = 0; i < 5; i++)
    {
        int dy = baseW * slideStep;
        int dx = baseH * slideStep;

        int dw = width - baseW;
        int dh = height - baseH;

        for(int y = 0; y < dh; y += dy)
        {
            for(int x = 0; x < dw; x+= dx)
            {
                cv::Mat part(gray, cv::Rect(x, y, baseW, baseH));
                cv::resize(part, part, cv::Size(WIDTH, HEIGHT));

//                cv::imshow("part", part);
//                cv::waitKey();

                uchar *pImg = part.data;
                float *pData = data;

                for(int j = 0; j < HEIGHT; j++)
                {
                    for(int i = 0; i < WIDTH; i++)
                        pData[i] = pImg[i] / 255.0;

                    pData += WIDTH;
                    pImg += part.step;
                }

//                normalize_image_npd(data, WIDTH, HEIGHT);
                integral_image(data, WIDTH, HEIGHT);

                if(classify(cc, data, WIDTH, 0, 0) == 1)
                    rects.push_back(cv::Rect(x, y, baseW, baseH));
            }
        }

        baseW *= scaleStep;
        baseH *= scaleStep;

        printf("%d\n", i);
    }

    delete[] data;
}


int main_detect(int argc, char **argv)
{
    char *modelFile = NULL;
    float scaleStep = 0.0f;
    float slideStep = 0.0f;

    char *iImgName = NULL;
    char *oImgName = NULL;

    if(argc < 11)
    {
        print_detect_usage(argv[0]);
        return 0;
    }

    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-m") == 0)
            modelFile = argv[++i];

        else if(strcmp(argv[i], "--scaleStep") == 0)
            scaleStep = atof(argv[++i]);

        else if(strcmp(argv[i], "--slideStep") == 0)
            slideStep = atof(argv[++i]);

        else if(strcmp(argv[i], "-i") == 0)
            iImgName = argv[++i];

        else if(strcmp(argv[i], "-o") == 0)
            oImgName = argv[++i];

        else {
            print_detect_usage(argv[0]);
            return 1;
        }
    }


    cv::Mat img = cv::imread(iImgName, 1);
    CascadeClassifier *cc;
    load(&cc, modelFile);

    std::vector<cv::Rect> rects;

    detect_object(cc, img, scaleStep, slideStep, rects);

    clear(cc);

    int size = rects.size();

    printf("object size %d\n", size);

    for(int i = 0; i < size; i++)
    {
        cv::rectangle(img, rects[i], cv::Scalar(0, 0, 255));
    }

    cv::imshow("img", img);
    cv::waitKey();

    if(!cv::imwrite(oImgName, img))
    {
        printf("Can't write image %s\n", oImgName);
    }

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



int main_test_weak_classifier(int argc, char **argv)
{
    WeakClassifier *wc = new WeakClassifier;

    float values[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8};
    float weights[20] = {0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05};


    float error = train(wc, values, 10, 10, weights);

    printf("thresh: %f, sign: %d, error: %f\n", wc->thresh, wc->sign, error);

    delete wc;

    return 0;
}
