#include "tool.h"
#include "weak_classifier.h"
#include "strong_classifier.h"
#include "cascade_classifier.h"


#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


//#define ADD_MIRROR_SAMPLE
//#define ADD_ROTATE_SAMPLE


float * mat_to_float(cv::Mat &img)
{
    int w = img.cols;
    int h = img.rows;

    float *data = new float[w * h];
    float *pdata = data;

    uchar *pImg = img.data;

    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++)
            pdata[x] = pImg[x] / 255.0;

        pdata += w;
        pImg += img.step;
    }

    return data;
}


int generate_positive_samples(const char *imgListFile, std::list<float*> &positiveSet, const int WIDTH, const int HEIGHT)
{
    std::vector<std::string> imgList;
    int ret;

    ret = read_image_list(imgListFile, imgList);
    int size = imgList.size();

    for(int i = 0; i < size; i++)
    {
        cv::Mat img = cv::imread(imgList[i], 0);
        float *fData = NULL;

        if(img.empty())
        {
            printf("Can't open image %s\n", imgList[i].c_str());
            return 2;
        }

        cv::resize(img, img, cv::Size(WIDTH, HEIGHT));
        fData = mat_to_float(img);
        positiveSet.push_back(fData);


#ifdef USE_HAAR_FEATURE
        integral_image(fData, WIDTH, HEIGHT);
#endif
        printf("%.2f\r", 100.0 * i / size);
        fflush(stdout);
    }

    return 0;
}


void generate_validate_samples(std::vector<std::string> &imgList, int WIDTH, int HEIGHT, std::list<float*> &validateSet, int size)
{
    assert(size < imgList.size());

    if(validateSet.size() > 0)
        clear_list(validateSet);

    for(int i = 0 ; i < size; i++)
    {
        cv::Mat img = cv::imread(imgList[i], 0);
        cv::Mat sImg;

        if(img.empty())
        {
            printf("Can't open image %s\n", imgList[i].c_str());
            exit(0);
        }

        cv::resize(img, sImg, cv::Size(WIDTH, HEIGHT));

        float *data = mat_to_float(sImg);

#ifdef USE_HAAR_FEATURE
        integral_image(data, WIDTH, HEIGHT);
#endif

        validateSet.push_back(data);
    }
}


int read_neg_sample_from_file(std::vector<std::string> &imgList, int WIDTH, int HEIGHT, float **res)
{
    static float scaleFactor = 0.8;
    static cv::Point oriPt(0, 0);
    static cv::Size offset(8, 8);
    static int curIdx = 0;
    static cv::Mat sImg;

    *res = NULL;

    while(1)
    {
        if(oriPt.y + HEIGHT < sImg.rows)
        {
            if(oriPt.x + WIDTH < sImg.cols)
            {
                cv::Mat img(sImg, cv::Rect(oriPt, cv::Size(WIDTH, HEIGHT)));

                *res = mat_to_float(img);

                oriPt.x += offset.width;
                return 1;
            }
            else
            {
                oriPt = cv::Point(0, oriPt.y + offset.height);
            }
        }
        else{

            if(sImg.cols < WIDTH || sImg.rows < HEIGHT)
            {
                if(curIdx >= imgList.size())
                    return 0;

                sImg = cv::imread(imgList[curIdx], 0);

                if(sImg.empty())
                    return 0;

                curIdx++;
            }
            else
                cv::resize(sImg, sImg, cv::Size(scaleFactor * sImg.cols, scaleFactor * sImg.rows));

            oriPt = cv::Point(0, 0);
        }
    }

    return 0;
}



int generate_negative_samples(std::vector<std::string> &imgList, int WIDTH, int HEIGHT, CascadeClassifier *cc, std::list<float*> &negativeSet, int size)
{
    int count = negativeSet.size();

    while(count < size)
    {
        float *data;
        int ret = read_neg_sample_from_file(imgList, WIDTH, HEIGHT, &data);

        if(ret == 0) return count;

#ifdef USE_HAAR_FEATURE
        integral_image(data, WIDTH, HEIGHT);
#endif

        if(classify(cc, data, WIDTH, 0, 0) == 1) {
            negativeSet.push_back(data);
            count++;
            printf("%6.2f%%\r", 100.0 * count/size);
            fflush(stdout);
        }
        else
        {
            delete [] data;
        }
    }

    if(count > size) return size;
    return count;
}


void select_feature(std::vector<Feature*> &featureSet, int featDim, std::vector<Feature*> &fineFeats){
    cv::RNG rng(cv::getTickCount());

    int size = featureSet.size();
    int len = size / featDim;

    fineFeats.resize(featDim);

    int count = featDim - 1;
    for(int i = 0; i < count; i++){
        int idx = rng.uniform(i * len, (i+1) * len);
        fineFeats[i] = featureSet[idx];
    }

    fineFeats[count] = featureSet[rng.uniform(count * len, size)];
}


StrongClassifier* adaboost_learning(CascadeClassifier *cc, std::list<float *> &positiveSet, int numPos,
        std::list<float *> &negativeSet, int numNeg, std::list<float *> &validateSet, std::vector<Feature *> &featureSet,
            float maxfpr, float maxfnr)
{
    StrongClassifier *sc = new StrongClassifier;

    int width = cc->WIDTH;
    int height = cc->HEIGHT;

    float *weights = NULL, *values = NULL;

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

            init_feature(feat, featureSet[i]);
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

        assert(minError > 0);

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


int main(int argc, char **argv)
{
#if defined(TRAIN_MODEL)
    main_train(argc, argv);

#elif defined(DETECT)
    main_detect(argc, argv);
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


int WINW = 48, WINH = 48, STAGE = 5;

float FALSE_ALARM_RATE = 0.05;
float MISSING_RATE = 0.05;
float NP_RATE = 1.0;

char POS_LIST_FILE[128], NEG_LIST_FILE[128];

int read_config(char *fileName)
{
    FILE *fin = fopen(fileName, "r");
    if(fin == NULL){
        printf("Can't open file %s\n", fileName);
        return 1;
    }

    char line[8192];
    char confIterm[][40] = {"WIDTH", "HEIGHT", "STAGE", "FALSE_ALARM_RATE", "MISSING_RATE", "POS_LIST", "NEG_LIST", "NP_RATE"};

    while(fgets(line, 8191, fin) != NULL){
        char name[128], value[20];
        sscanf(line, "%s %s", name, value);

        if(strcmp(name, confIterm[0]) == 0){
            WINW = atoi(value);
        }
        else if(strcmp(name, confIterm[1]) == 0){
            WINH = atoi(value);
        }
        else if(strcmp(name, confIterm[2]) == 0){
            STAGE = atoi(value);
        }
        else if(strcmp(name, confIterm[3]) == 0){
            FALSE_ALARM_RATE = atof(value);
        }
        else if(strcmp(name, confIterm[4]) == 0){
            MISSING_RATE = atof(value);
        }
        else if(strcmp(name, confIterm[5]) == 0){
            strcpy(POS_LIST_FILE, value);
        }
        else if(strcmp(name, confIterm[6]) == 0){
            strcpy(NEG_LIST_FILE, value);
        }
        else if(strcmp(name, confIterm[7]) == 0){
            NP_RATE = atof(value);
        }
        else {
        }
    }

    return 0;
}

int main_train(int argc, char **argv)
{
    if(argc < 3){
        printf("Usage: %s [configuration] [model]\n", argv[0]);
        return 1;
    }

    int ret = read_config(argv[1]);
    if(ret != 0) return 1;
    char *modelFile = argv[2];

    std::vector<std::string> negImgList;
    std::list<float *> positiveSet, negativeSet, validateSet;
    std::vector<Feature*> featureSet, fineFeatSet;
    float *stepFPR;

    int numPos, numNeg, numVal;
    CascadeClassifier *cc = new CascadeClassifier();
    StrongClassifier* sc;
    float maxfpr = 1.0;
    float maxfnr = MISSING_RATE;

    std::list<StrongClassifier *> scs;
    init_cascade_classifier(cc, scs, WINW, WINH);

    printf("GENERATE POSITIVE SAMPLES\n");
    ret = generate_positive_samples(POS_LIST_FILE, positiveSet, WINW, WINH);
    if(ret != 0) return 2;

    printf("GENERATE NEGATIVE SAMPLES\n");
    read_image_list(NEG_LIST_FILE, negImgList);

    numPos = positiveSet.size();
    numNeg = numPos * NP_RATE;

    for(int i = 0; i < numNeg; i ++)
    {
        float *data = NULL;
        read_neg_sample_from_file(negImgList, WINW, WINH, &data);

#ifdef USE_HAAR_FEATURE
        integral_image(data, WINW, WINH);
#endif
        negativeSet.push_back(data);
    }


    printf("GENERATE VALIDATE SAMPLES\n");
    generate_validate_samples(negImgList, WINW, WINH, validateSet, numPos);

    numVal = validateSet.size();
    printf("Positive sample size: %ld\n", positiveSet.size());
    printf("Negative sample size: %ld\n", negativeSet.size());
    printf("Validate sample size: %d\n", numVal);

    printf("GENERATE FEATURE TEMPLATE\n");
    generate_feature_set(featureSet, WINW, WINH);

    init_steps_false_positive(&stepFPR, STAGE, FALSE_ALARM_RATE);

    clock_t startTime = clock();
    char outname[128];

    for(int i = 0; i < STAGE; i++){
        printf("\n--------------cascade stage %d-----------------\n", i+1);

        int correctSize = 0;

        std::list<float*>::iterator iter, iterEnd;

        numNeg = numPos * NP_RATE;
        printf("READ NEGATIVE SAMPLES\n");
        ret = generate_negative_samples(negImgList, WINW, WINH, cc, negativeSet, numNeg);
        if(ret != numNeg) {
            printf("Can't generate enough negatvie samples %d:%d\n", ret, numNeg);
            break;
        }

        printf("READ VALIDATE SAMPLES\n");
        ret = generate_negative_samples(negImgList, WINW, WINH, cc, validateSet, numVal);
        if(ret != numVal) {
            printf("Can't generate enough validate samples %d:%d\n", ret, numVal);
            break;
        }

        select_feature(featureSet, 2000, fineFeatSet);
        maxfpr *= stepFPR[i];

        printf("Positive sample size: %d\n", numPos);
        printf("Negative sample size: %d\n", numNeg);
        printf("Target false positive rate: %f\n", maxfpr);
        printf("Target false negative rate: %f\n", maxfnr);

        sc = adaboost_learning(cc, positiveSet, numPos, negativeSet, numNeg, validateSet, fineFeatSet, maxfpr, maxfnr);
        add(cc, sc);

        iter = positiveSet.begin();
        iterEnd = positiveSet.end();

        while(iter != iterEnd)
        {
            if(classify(cc, *iter, WINW, 0, 0) == 0)
            {
                std::list<float*>::iterator iterTmp = iter;
                iter++;
                delete[] (*iterTmp);
                positiveSet.erase(iterTmp);
                iter--;
            }

            iter++;
        }

        numPos = positiveSet.size();
        printf("cascade TP: %d\n", numPos);

        iter = negativeSet.begin();
        iterEnd = negativeSet.end();

        correctSize = negativeSet.size();

        while(iter != iterEnd)
        {
            if(classify(cc, *iter, WINW, 0, 0) == 0)
            {
                std::list<float*>::iterator iterTmp = iter;
                iter++;
                delete[] (*iterTmp);
                negativeSet.erase(iterTmp);
                iter--;
            }

            iter++;
        }

        printf("cascade TN: %ld\n", correctSize - negativeSet.size());

        iter = validateSet.begin();
        iterEnd = validateSet.end();
        while(iter != iterEnd)
        {
            if(classify(cc, *iter, WINW, 0, 0) == 0)
            {
                std::list<float*>::iterator iterTmp = iter;
                iter++;
                delete[] (*iterTmp);
                validateSet.erase(iterTmp);
                iter--;
            }

            iter++;
        }

        printf("----------------------------------------\n");

        sprintf(outname, "model/cascade_%d.dat", i+1);
        save(cc, outname);

#ifdef SHOW_FEATURE
        print_feature(cc);
#endif
    }

    save(cc, modelFile);

    clock_t trainTime = clock() - startTime;

    printf("Train time:");
    print_time(trainTime);
    printf("\n");
    clear(cc);

    clear_list(positiveSet);
    clear_list(negativeSet);
    clear_list(validateSet);

    clear_features(featureSet);

    return 0;
}



void detect_object2(CascadeClassifier *cc, cv::Mat &img, float startScale, float endScale, int layers, float offsetFactor, std::vector<cv::Rect> &rects)
{
    cv::Mat gray, sImg;

    int winX = cc->WIDTH;
    int winY = cc->HEIGHT;

    int dx = offsetFactor * winX;
    int dy = offsetFactor * winY;

    float *data = new float [winX * winY];
    float scaleStep = (endScale - startScale) / layers;

    if(endScale > startScale) {
        startScale = endScale;
        scaleStep = -scaleStep;
    }

    if(img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    for(int i = 0; i < layers; i++)
    {
        cv::resize(gray, sImg, cv::Size(startScale * gray.cols, startScale * gray.rows));

        int ws = sImg.cols - winX;
        int hs = sImg.rows - winY;

        for(int y = 0; y < hs; y += dy)
        {
            for(int x = 0; x < ws; x += dx)
            {
                cv::Rect rect = cv::Rect(x, y, winX, winY);
                cv::Mat patch(sImg, rect);

                float *pData = data;
                uchar *iData = patch.data;

                for(int m = 0; m < winY; m++)
                {
                    for(int n = 0; n < winX; n++)
                        pData[n] = iData[n] / 255.0;

                    pData += winX;
                    iData += patch.step;
                }

#ifdef USE_HAAR_FEATURE
                integral_image(data, winX, winY);
#endif

                if(classify(cc, data, winX, 0, 0) == 1)
                {
                    rect.x /= startScale;
                    rect.y /= startScale;

                    rect.width /= startScale;
                    rect.height /= startScale;

                    rects.push_back(rect);
                }
            }

        }

        startScale += scaleStep;
    }

    delete [] data;
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

    detect_object2(cc, img, 0.4, 1.0, 5, slideStep, rects);
    merge_rect(rects);

    clear(cc);

    int size = rects.size();

    printf("object size %d\n", size);

    for(int i = 0; i < size; i++){
        cv::rectangle(img, rects[i], cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("img", img);
    cv::waitKey();

    if(!cv::imwrite(oImgName, img))
    {
        printf("Can't write image %s\n", oImgName);
    }

    return 0;
}

