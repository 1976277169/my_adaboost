#include "tool.h"


int read_file_list(const char *filePath, std::vector<std::string> &fileList)
{
    char line[512];
    FILE *fin = fopen(filePath, "r");

    if(fin == NULL){
        printf("Can't open file: %s\n", filePath);
        return -1;
    }

    while(fscanf(fin, "%s\n", line) != EOF){
        fileList.push_back(line);
    }

    fclose(fin);

    return 0;
}


void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext)
{
    int len = strlen(filePath);
    int idx = len - 1, idx2 = 0;

    while(idx >= 0){
        if(filePath[idx] == '.')
            break;
        idx--;
    }

    if(idx >= 0){
        strcpy(ext, filePath + idx + 1);
        ext[len - idx] = '\0';
    }
    else {
        ext[0] = '\0';
        idx = len - 1;
    }

    idx2 = idx;
    while(idx2 >= 0){
#if defined(WIN32)
        if(filePath[idx2] == '\\')
#elif defined(linux)
        if(filePath[idx2] == '/')
#endif
            break;

        idx2 --;
    }

    if(idx2 > 0){
        strncpy(rootDir, filePath, idx2);
        rootDir[idx2] = '\0';
    }
    else{
        rootDir[0] = '.';
        rootDir[1] = '\0';
    }

    strncpy(fileName, filePath + idx2 + 1, idx - idx2 - 1);
    fileName[idx - idx2 - 1] = '\0';
}


void integral_image(uint8_t *img, int width, int height, int stride, uint32_t *intImg, int istride){
    int id0 = 0, id1 = 0;

    for(int y = 0; y < height; y++){
        intImg[id0] = img[id1];
        for(int x = 1; x < width; x++){
            intImg[id0 + x] = img[id1] + intImg[id0 + x - 1];
        }

        id0 += istride;
        id1 = stride;
    }

    id0 = 0, id1 = istride;

    for(int y = 1; y < height; y++){
        for(int x = 0; x < width; x++){
            intImg[id1 + x] += intImg[id0 + x];
        }

        id0 += istride;
        id1 += istride;
    }
}


void update_weights(double *weights, int size){
    double sum = 0.0;

    for(int i = 0; i < size; i++)
        sum += weights[i];

    sum = 1.0f / sum;
    for(int i = 0; i < size; i++)
        weights[i] = weights[i] * sum;
}


#define FIX_INTER_POINT 14

void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
    uint16_t *table = NULL;

    uint16_t FIX_0_5 = 1 << (FIX_INTER_POINT - 1);
    float scalex, scaley;

    scalex = srcw / float(dstw);
    scaley = srch / float(dsth);

    table = new uint16_t[dstw * 3];

    for(int i = 0; i < dstw; i++){
        float x = i * scalex;

        if(x < 0) x = 0;
        if(x > srcw - 1) x = srcw - 1;

        int x0 = int(x);

        table[i * 3] = x0;
        table[i * 3 + 2] = (x - x0) * (1 << FIX_INTER_POINT);
        table[i * 3 + 1] = (1 << FIX_INTER_POINT) - table[i * 3 + 2];
    }

    int sId = 0, dId = 0;

    for(int y = 0; y < dsth; y++){
        int x;
        float yc;

        uint16_t wy0, wy1;
        uint16_t y0, y1;
        uint16_t *ptrTab = table;
        int buffer[8];
        yc = y * scaley;
        yc = yc > 0 ? yc : 0;
        yc = yc < srch - 1 ? yc : srch - 1;

        y0 = uint16_t(yc);
        y1 = y0 + 1;

        wy1 = uint16_t((yc - y0) * (1 << FIX_INTER_POINT));
        wy0 = (1 << FIX_INTER_POINT) - wy1;

        sId = y0 * srcs;

        uint8_t *ptrDst = dst + dId;

        for(x = 0; x <= dstw - 4; x += 4){
            uint16_t x0, x1, wx0, wx1;
            uint8_t *ptrSrc0, *ptrSrc1;

            //1
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[0] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[1] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //2
            x0 = ptrTab[3], x1 = x0 + 1;

            wx0 = ptrTab[4], wx1 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[2] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[3] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //3
            x0 = ptrTab[6], x1 = x0 + 1;

            wx0 = ptrTab[7], wx1 = ptrTab[8];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[4] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[5] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //4
            x0 = ptrTab[9], x1 = x0 + 1;
            wx0 = ptrTab[10], wx1 = ptrTab[11];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[6] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[7] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst[0] = (wy0 * (buffer[0] - buffer[1]) + (buffer[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[1] = (wy0 * (buffer[2] - buffer[3]) + (buffer[3] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[2] = (wy0 * (buffer[4] - buffer[5]) + (buffer[5] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[3] = (wy0 * (buffer[6] - buffer[7]) + (buffer[7] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst += 4;
            ptrTab += 12;
        }

        for(; x < dstw; x++){
            uint16_t x0, x1, wx0, wx1, valuex0, valuex1;

            uint8_t *ptrSrc0, *ptrSrc1;
            x0 = ptrTab[0], x1 = x0 + 1;

            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            valuex0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            valuex1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            dst[y * dsts + x] = (wy0 * (valuex0 - valuex1) + (valuex1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrTab += 3;
        }

        dId += dsts;
    }


    delete [] table;
}

