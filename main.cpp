#include "adaboost.h"

#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


//#define ADD_MIRROR_SAMPLE
//#define ADD_ROTATE_SAMPLE



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

    CascadeClassifier *cc = new CascadeClassifier();

    std::vector<Feature*> featureSet;
    std::vector<std::string> negImgList;
    std::vector<Sample*> positiveSet, negativeSet, validateSet;
    float *stepFPR;

    int numPos, numNeg, numVal;
    float maxfpr = 1.0;
    float maxfnr = MISSING_RATE;

    char outname[128];

    init_cascade_classifier(cc, std::vector<StrongClassifier *>(), WINW, WINH);

    printf("GENERATE FEATURE TEMPLATE\n");
    generate_feature_set(featureSet, WINW, WINH);

    printf("GENERATE POSITIVE SAMPLES\n");
    if(read_positive_sample_from_file(POS_LIST_FILE, WINW, WINH, positiveSet)){
        return 1;
    }

    printf("READ NEGATIVE SAMPLES\n");
    read_image_list(NEG_LIST_FILE, negImgList);

    numPos = positiveSet.size();

    printf("GENERATE VALIDATE SAMPLES\n");
    numVal = numPos < 1000 ? numPos : 1000;

    printf("Positive sample size: %d\n", numPos);

    init_steps_false_positive(&stepFPR, STAGE, FALSE_ALARM_RATE);

    clock_t startTime = clock();

    for(int i = 0; i < STAGE; i++){
        printf("\n--------------cascade stage %d-----------------\n", i+1);

        numPos = positiveSet.size();
        numNeg = numPos * NP_RATE;
        maxfpr *= stepFPR[i];

        printf("numPos: %d, numNeg = %d, TP: %6.4f, FN: %6.4f\n", numPos, numNeg, maxfpr, maxfnr);

        printf("ADD NEGATIVE SAMPLES\n");
        ret = generate_negative_samples(negImgList, WINW, WINH, cc, negativeSet, numNeg);
        if(ret != 0) {
            printf("Can't generate enough negatvie samples %ld\n", negativeSet.size());
            break;
        }

        printf("ADD VALIDATE SAMPLES\n");
        ret = generate_negative_samples(negImgList, WINW, WINH, cc, validateSet, numVal);
        if(ret != 0) {
            printf("Can't generate enough validate samples %ld\n", validateSet.size());
            break;
        }

        write_samples(positiveSet, "log/pos");
        write_samples(negativeSet, "log/neg");
        write_samples(validateSet, "log/val");

        adaboost_learning(cc, positiveSet, negativeSet, validateSet, featureSet, maxfpr, maxfnr);

        printf("CLEAR SAMPLES\n");

        clean_samples(cc, positiveSet);
        clean_samples(cc, negativeSet);
        clean_samples(cc, validateSet);

        printf("%ld %ld %ld\n", positiveSet.size(), negativeSet.size(), validateSet.size());

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
    clear(&cc);

    clear_list(positiveSet);
    clear_list(negativeSet);
    clear_list(validateSet);

    clear_features(featureSet);

    delete [] stepFPR;

    return 0;
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

    detect_object(cc, img, 0.4, 1.0, 5, slideStep, rects);
    merge_rect(rects);

    clear(&cc);

    int size = rects.size();

    printf("object size %d\n", size);

    for(int i = 0; i < size; i++){
        cv::rectangle(img, rects[i], cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("img", img);
    cv::waitKey();

    if(!cv::imwrite(oImgName, img)){
        printf("Can't write image %s\n", oImgName);
    }

    return 0;
}

