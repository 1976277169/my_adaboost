#include "adaboost.h"



int main_train(int argc, char **argv){
    if(argc < 4){
        printf("Usage: %s [pos list] [neg list] [out model]\n", argv[0]);
        return 1;
    }

    Cascade objDetector;

    int ret = train(&objDetector, argv[1], argv[2]);
    if(ret != 0){
        printf("TRAIN ERROR\n");
        return 2;
    }

    save(&objDetector, argv[3]);

    release_data(&objDetector);

    return 0;
}


int main_detect(int argc, char **argv){
    if(argc < 3){
        printf("Usage: %s [model] [image list]\n", argv[1]);
        return 1;
    }

    Cascade objDetector;
    std::vector<std::string> imgList;
    int size;

    int ret = load(&objDetector, argv[1]);

    if(ret != 0)
        return 1;

    read_file_list(argv[2], imgList);

    size = imgList.size();

    for(int i = 0; i < size; i++){
        cv::Mat img = cv::imread(imgList[i], 1);
        cv::Mat gray;
        HRect *rects;
        int num;

        if(img.empty()){
            printf("Can't open image %s\n", imgList[i].c_str());
            continue;
        }

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        num = detect(&objDetector, gray.data, gray.cols, gray.rows, gray.step, &rects);

        for(int j = 0; j < num; j++){
            cv::rectangle(img, cv::Rect(rects[j].x, rects[j].y, rects[j].width, rects[j].height), cv::Scalar(0, 255, 0), 2);
        }

        printf("DETECT: %d\n", num);
        cv::imshow("img", img);
        cv::waitKey();

        if(num > 0)
            delete [] rects;
        rects = NULL;
    }

    release_data(&objDetector);

    return 0;
}


int main(int argc, char **argv){
#if defined(MAIN_TRAIN)
    main_train(argc, argv);

#elif defined(MAIN_DETECT_IMAGES)
    main_detect(argc, argv);

#endif

    return 0;
}
