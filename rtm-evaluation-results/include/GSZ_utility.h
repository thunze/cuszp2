#ifndef GSZ_INCLUDE_GSZ_UTILITY_H
#define GSZ_INCLUDE_GSZ_UTILITY_H

void symTransForm_4Bytes(unsigned char data[4]);
void symTransform_8bytes(unsigned char data[8]);
unsigned char *readByteData_Yafan(char *srcFilePath, size_t *byteLength, int *status);
float *readFloatData_systemEndian_Yafan(char *srcFilePath, size_t *nbEle, int *status);
float *readFloatData_Yafan(char *srcFilePath, size_t *nbEle, int *status);
double *readDoubleData_systemEndian_Yafan(char *srcFilePath, size_t *nbEle, int *status);
double *readDoubleData_Yafan(char *srcFilePath, size_t *nbEle, int *status);
void writeByteData_Yafan(unsigned char *bytes, size_t byteLength, char *tgtFilePath, int *status);
void writeFloatData_inBytes_Yafan(float *data, size_t nbEle, char* tgtFilePath, int *status);
void writeDoubleData_inBytes_Yafan(double *data, size_t nbEle, char* tgtFilePath, int *status);
double SSIM_3d_calcWindow_float(float* data, float* other, size_t size1, size_t size0, int offset0, int offset1, int offset2, int windowSize0, int windowSize1, int windowSize2);
double computeSSIM(float* oriData, float* decData, size_t size2, size_t size1, size_t size0);
double *computePSNR(size_t nbEle, float *ori_data, float *data);

#endif // GSZ_INCLUDE_GSZ_UTILITY_H