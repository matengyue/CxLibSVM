#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "./libsvm/svm.h"
using namespace std;
//内存分配
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* 封装svm                                                                     */
/************************************************************************/
class CxLibSVM
{
private:
    
    struct svm_model*    model_;
    struct svm_parameter    param;
    struct svm_problem        prob;
    struct svm_node *        x_space;
public:
    //************************************
    // 描    述: 构造函数
    // 方    法: CxLibSVM
    // 文 件 名: CxLibSVM::CxLibSVM
    // 访问权限: public 
    // 返 回 值: 
    // 限 定 符:
    //************************************
    CxLibSVM()
    {
        model_ = NULL;
    }

    //************************************
    // 描    述: 析构函数
    // 方    法: ~CxLibSVM
    // 文 件 名: CxLibSVM::~CxLibSVM
    // 访问权限: public 
    // 返 回 值: 
    // 限 定 符:
    //************************************
    ~CxLibSVM()
    {
        free_model();
    }
    
    //************************************
    // 描    述: 训练模型
    // 方    法: train
    // 文 件 名: CxLibSVM::train
    // 访问权限: public 
    // 参    数: const vector<vector<double>> & x
    // 参    数: const vector<double> & y
    // 参    数: const int & alg_type
    // 返 回 值: void
    // 限 定 符:
    //************************************
    void train(const vector<vector<double>>&  x, const vector<double>& y, const struct svm_parameter& param)
    {
        if (x.size() == 0)return;

        //释放先前的模型
        free_model();

        /*初始化*/        
        long    len = x.size();
        long    dim = x[0].size();
        long    elements = len*dim;

        //参数初始化，参数调整部分在这里修改即可
        // 默认参数
        //param.svm_type = C_SVC;        //算法类型
        //param.kernel_type = LINEAR;    //核函数类型
        //param.degree = 3;    //多项式核函数的参数degree
        //param.coef0 = 0;    //多项式核函数的参数coef0
        //param.gamma = 0.5;    //1/num_features，rbf核函数参数
        //param.nu = 0.5;        //nu-svc的参数
        //param.C = 10;        //正则项的惩罚系数
        //param.eps = 1e-3;    //收敛精度
        //param.cache_size = 100;    //求解的内存缓冲 100MB
        //param.p = 0.1;    
        //param.shrinking = 1;
        //param.probability = 1;    //1表示训练时生成概率模型，0表示训练时不生成概率模型，用于预测样本的所属类别的概率
        //param.nr_weight = 0;    //类别权重
        //param.weight = NULL;    //样本权重
        //param.weight_label = NULL;    //类别权重
        

        //转换数据为libsvm格式
        prob.l = len;
        prob.y = Malloc(double, prob.l);
        prob.x = Malloc(struct svm_node *, prob.l);
        x_space    = Malloc(struct svm_node, elements+len);
        int j = 0;
        for (int l = 0; l < len; l++)
        {
            prob.x[l] = &x_space[j];
            for (int d = 0; d < dim; d++)
            {                
                x_space[j].index = d+1;
                x_space[j].value = x[l][d];    
                j++;
            }
            x_space[j++].index = -1;
            prob.y[l] = y[l];
        }

        /*训练*/
        model_ = svm_train(&prob, &param);    
    }

    //************************************
    // 描    述: 预测测试样本所属类别和概率
    // 方    法: predict
    // 文 件 名: CxLibSVM::predict
    // 访问权限: public 
    // 参    数: const vector<double> & x    样本
    // 参    数: double & prob_est            类别估计的概率
    // 返 回 值: double                        预测的类别
    // 限 定 符:
    //************************************
    int predict(const vector<double>& x,double& prob_est)
    {
        //数据转换
        svm_node* x_test = Malloc(struct svm_node, x.size()+1);
        for (unsigned int i=0; i<x.size(); i++)
        {
            x_test[i].index = i;
            x_test[i].value = x[i];
        }
        x_test[x.size()].index = -1;
        double *probs = new double[model_->nr_class];//存储了所有类别的概率
        //预测类别和概率
        int value = (int)svm_predict_probability(model_, x_test, probs);
        for (int k = 0; k < model_->nr_class;k++)
        {//查找类别相对应的概率
            if (model_->label[k] == value)
            {
                prob_est = probs[k];
                break;
            }
        }
        delete[] probs;
        return value;
    }

    void do_cross_validation(const vector<vector<double>>&  x, const vector<double>& y, const struct svm_parameter&    param, const int & nr_fold)
    {
        if (x.size() == 0)return;

        /*初始化*/
        long    len = x.size();
        long    dim = x[0].size();
        long    elements = len*dim;

        //转换数据为libsvm格式
        prob.l = len;
        prob.y = Malloc(double, prob.l);
        prob.x = Malloc(struct svm_node *, prob.l);
        x_space = Malloc(struct svm_node, elements + len);
        int j = 0;
        for (int l = 0; l < len; l++)
        {
            prob.x[l] = &x_space[j];
            for (int d = 0; d < dim; d++)
            {
                x_space[j].index = d + 1;
                x_space[j].value = x[l][d];
                j++;
            }
            x_space[j++].index = -1;
            prob.y[l] = y[l];
        }

        int i;
        int total_correct = 0;
        double total_error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double *target = Malloc(double, prob.l);

        svm_cross_validation(&prob, &param, nr_fold, target);
        if (param.svm_type == EPSILON_SVR ||
            param.svm_type == NU_SVR)
        {
            for (i = 0; i < prob.l; i++)
            {
                double y = prob.y[i];
                double v = target[i];
                total_error += (v - y)*(v - y);
                sumv += v;
                sumy += y;
                sumvv += v*v;
                sumyy += y*y;
                sumvy += v*y;
            }
            printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
            printf("Cross Validation Squared correlation coefficient = %g\n",
                ((prob.l*sumvy - sumv*sumy)*(prob.l*sumvy - sumv*sumy)) /
                ((prob.l*sumvv - sumv*sumv)*(prob.l*sumyy - sumy*sumy))
            );
        }
        else
        {
            for (i = 0; i < prob.l; i++)
                if (target[i] == prob.y[i])
                    ++total_correct;
            printf("Cross Validation Accuracy = %g%%\n", 100.0*total_correct / prob.l);
        }
        free(target);
    }

    //************************************
    // 描    述: 导入svm模型
    // 方    法: load_model
    // 文 件 名: CxLibSVM::load_model
    // 访问权限: public 
    // 参    数: string model_path    模型路径
    // 返 回 值: int 0表示成功；-1表示失败
    // 限 定 符:
    //************************************
    int load_model(string model_path)
    {
        //释放原来的模型
        free_model();
        //导入模型
        model_ = svm_load_model(model_path.c_str());
        if (model_ == NULL)return -1;
        return 0;
    }

    //************************************
    // 描    述: 保存模型
    // 方    法: save_model
    // 文 件 名: CxLibSVM::save_model
    // 访问权限: public 
    // 参    数: string model_path    模型路径
    // 返 回 值: int    0表示成功，-1表示失败
    // 限 定 符:
    //************************************
    int save_model(string model_path)
    {
        int flag = svm_save_model(model_path.c_str(), model_);
        return flag;
    }

private:

    //************************************
    // 描    述: 释放svm模型内存
    // 方    法: free_model
    // 文 件 名: CxLibSVM::free_model
    // 访问权限: private 
    // 返 回 值: void
    // 限 定 符:
    //************************************
    void free_model()
    {
        if (model_ != NULL)
        {
            svm_free_and_destroy_model(&model_);
            svm_destroy_param(&param);
            free(prob.y);
            free(prob.x);
            free(x_space);
        }
    }
};