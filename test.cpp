#include "cxlibsvm.hpp"
#include <time.h>
#include <iostream>
using namespace std;

void init_svm_param(struct svm_parameter& param)
{
    //参数初始化，参数调整部分在这里修改即可
    // 默认参数
    param.svm_type = C_SVC;        //算法类型
    param.kernel_type = LINEAR;    //核函数类型
    param.degree = 3;    //多项式核函数的参数degree
    param.coef0 = 0;    //多项式核函数的参数coef0
    param.gamma = 0.5;    //1/num_features，rbf核函数参数
    param.nu = 0.5;        //nu-svc的参数
    param.C = 10;        //正则项的惩罚系数
    param.eps = 1e-3;    //收敛精度
    param.cache_size = 100;    //求解的内存缓冲 100MB
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;    //1表示训练时生成概率模型，0表示训练时不生成概率模型，用于预测样本的所属类别的概率
    param.nr_weight = 0;    //类别权重
    param.weight = NULL;    //样本权重
    param.weight_label = NULL;    //类别权重
}

void gen_train_sample(vector<vector<double>>&    x, vector<double>&    y, long sample_num = 200, long dim = 10, double scale = 1)
{
    //long sample_num = 200;        //样本数
    //long dim = 10;    //样本类别
    //double scale = 1;    //数据缩放尺度

    srand((unsigned)time(NULL));//随机数
    //生成随机的正类样本
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

    //生成随机的负类样本
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
    //long sample_num = 200;        //样本数
    //long dim = 10;    //样本类别
    //double scale = 1;    //数据缩放尺度

    srand((unsigned)time(NULL));//随机数
                                //生成随机的正类样本
    for (int j = 0; j < dim; j++)
    {
        x.push_back(-scale*(rand() % 10));
    }
}

void main()
{
    //初始化libsvm
    CxLibSVM    svm;

    //初始化参数
    struct svm_parameter param;
    init_svm_param(param);

    /*1、准备训练数据*/
    vector<vector<double>>    x;    //样本集
    vector<double>    y;            //样本类别集
    gen_train_sample(x, y, 200, 10, 1);

    /*1、交叉验证*/
    int fold = 10;
    param.C = 100;
    param.svm_type = LINEAR;
    svm.do_cross_validation(x, y, param, fold);

    /*2、训练*/
    svm.train(x, y, param);

    /*3、保存模型*/
    string model_path = ".\\svm_model.txt";
    svm.save_model(model_path);

    /*4、导入模型*/
    string model_path_p = ".\\svm_model.txt";
    svm.load_model(model_path_p);

    /*5、预测*/
    //生成随机测试数据
    vector<double> x_test;
    gen_test_sample(x_test, 200, 10, 1);
    double prob_est;
    //预测
    double value = svm.predict(x_test, prob_est);

    //打印预测类别和概率
    printf("label:%f,prob:%f", value, prob_est);
}