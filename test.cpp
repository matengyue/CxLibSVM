#include "cxlibsvm.hpp"
#include <time.h>
#include <iostream>
using namespace std;

void init_svm_param(struct svm_parameter& param)
{
    //������ʼ�����������������������޸ļ���
    // Ĭ�ϲ���
    param.svm_type = C_SVC;        //�㷨����
    param.kernel_type = LINEAR;    //�˺�������
    param.degree = 3;    //����ʽ�˺����Ĳ���degree
    param.coef0 = 0;    //����ʽ�˺����Ĳ���coef0
    param.gamma = 0.5;    //1/num_features��rbf�˺�������
    param.nu = 0.5;        //nu-svc�Ĳ���
    param.C = 10;        //������ĳͷ�ϵ��
    param.eps = 1e-3;    //��������
    param.cache_size = 100;    //�����ڴ滺�� 100MB
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;    //1��ʾѵ��ʱ���ɸ���ģ�ͣ�0��ʾѵ��ʱ�����ɸ���ģ�ͣ�����Ԥ���������������ĸ���
    param.nr_weight = 0;    //���Ȩ��
    param.weight = NULL;    //����Ȩ��
    param.weight_label = NULL;    //���Ȩ��
}

void gen_train_sample(vector<vector<double>>&    x, vector<double>&    y, long sample_num = 200, long dim = 10, double scale = 1)
{
    //long sample_num = 200;        //������
    //long dim = 10;    //�������
    //double scale = 1;    //�������ų߶�

    srand((unsigned)time(NULL));//�����
    //�����������������
    for (int i = 0; i < sample_num; i++)
    {
        vector<double> rx;
        for (int j = 0; j < dim; j++)
        {
            rx.push_back(scale*(rand() % 10));
        }
        x.push_back(rx);
        y.push_back(1);
    }

    //��������ĸ�������
    for (int i = 0; i < sample_num; i++)
    {
        vector<double> rx;
        for (int j = 0; j < dim; j++)
        {
            rx.push_back(-scale*(rand() % 10));
        }
        x.push_back(rx);
        y.push_back(2);
    }
}

void gen_test_sample(vector<double>&    x, long sample_num = 200, long dim = 10, double scale = 1)
{
    //long sample_num = 200;        //������
    //long dim = 10;    //�������
    //double scale = 1;    //�������ų߶�

    srand((unsigned)time(NULL));//�����
                                //�����������������
    for (int j = 0; j < dim; j++)
    {
        x.push_back(-scale*(rand() % 10));
    }
}

void main()
{
    //��ʼ��libsvm
    CxLibSVM    svm;

    //��ʼ������
    struct svm_parameter param;
    init_svm_param(param);

    /*1��׼��ѵ������*/
    vector<vector<double>>    x;    //������
    vector<double>    y;            //�������
    gen_train_sample(x, y, 200, 10, 1);

    /*1��������֤*/
    int fold = 10;
    param.C = 100;
    param.svm_type = LINEAR;
    svm.do_cross_validation(x, y, param, fold);

    /*2��ѵ��*/
    svm.train(x, y, param);

    /*3������ģ��*/
    string model_path = ".\\svm_model.txt";
    svm.save_model(model_path);

    /*4������ģ��*/
    string model_path_p = ".\\svm_model.txt";
    svm.load_model(model_path_p);

    /*5��Ԥ��*/
    //���������������
    vector<double> x_test;
    gen_test_sample(x_test, 200, 10, 1);
    double prob_est;
    //Ԥ��
    double value = svm.predict(x_test, prob_est);

    //��ӡԤ�����͸���
    printf("label:%f,prob:%f", value, prob_est);
}