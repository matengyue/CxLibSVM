#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "./libsvm/svm.h"
using namespace std;
//�ڴ����
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* ��װsvm                                                                     */
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
    // ��    ��: ���캯��
    // ��    ��: CxLibSVM
    // �� �� ��: CxLibSVM::CxLibSVM
    // ����Ȩ��: public 
    // �� �� ֵ: 
    // �� �� ��:
    //************************************
    CxLibSVM()
    {
        model_ = NULL;
    }

    //************************************
    // ��    ��: ��������
    // ��    ��: ~CxLibSVM
    // �� �� ��: CxLibSVM::~CxLibSVM
    // ����Ȩ��: public 
    // �� �� ֵ: 
    // �� �� ��:
    //************************************
    ~CxLibSVM()
    {
        free_model();
    }
    
    //************************************
    // ��    ��: ѵ��ģ��
    // ��    ��: train
    // �� �� ��: CxLibSVM::train
    // ����Ȩ��: public 
    // ��    ��: const vector<vector<double>> & x
    // ��    ��: const vector<double> & y
    // ��    ��: const int & alg_type
    // �� �� ֵ: void
    // �� �� ��:
    //************************************
    void train(const vector<vector<double>>&  x, const vector<double>& y, const struct svm_parameter& param)
    {
        if (x.size() == 0)return;

        //�ͷ���ǰ��ģ��
        free_model();

        /*��ʼ��*/        
        long    len = x.size();
        long    dim = x[0].size();
        long    elements = len*dim;

        //������ʼ�����������������������޸ļ���
        // Ĭ�ϲ���
        //param.svm_type = C_SVC;        //�㷨����
        //param.kernel_type = LINEAR;    //�˺�������
        //param.degree = 3;    //����ʽ�˺����Ĳ���degree
        //param.coef0 = 0;    //����ʽ�˺����Ĳ���coef0
        //param.gamma = 0.5;    //1/num_features��rbf�˺�������
        //param.nu = 0.5;        //nu-svc�Ĳ���
        //param.C = 10;        //������ĳͷ�ϵ��
        //param.eps = 1e-3;    //��������
        //param.cache_size = 100;    //�����ڴ滺�� 100MB
        //param.p = 0.1;    
        //param.shrinking = 1;
        //param.probability = 1;    //1��ʾѵ��ʱ���ɸ���ģ�ͣ�0��ʾѵ��ʱ�����ɸ���ģ�ͣ�����Ԥ���������������ĸ���
        //param.nr_weight = 0;    //���Ȩ��
        //param.weight = NULL;    //����Ȩ��
        //param.weight_label = NULL;    //���Ȩ��
        

        //ת������Ϊlibsvm��ʽ
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

        /*ѵ��*/
        model_ = svm_train(&prob, &param);    
    }

    //************************************
    // ��    ��: Ԥ����������������͸���
    // ��    ��: predict
    // �� �� ��: CxLibSVM::predict
    // ����Ȩ��: public 
    // ��    ��: const vector<double> & x    ����
    // ��    ��: double & prob_est            �����Ƶĸ���
    // �� �� ֵ: double                        Ԥ������
    // �� �� ��:
    //************************************
    int predict(const vector<double>& x,double& prob_est)
    {
        //����ת��
        svm_node* x_test = Malloc(struct svm_node, x.size()+1);
        for (unsigned int i=0; i<x.size(); i++)
        {
            x_test[i].index = i;
            x_test[i].value = x[i];
        }
        x_test[x.size()].index = -1;
        double *probs = new double[model_->nr_class];//�洢���������ĸ���
        //Ԥ�����͸���
        int value = (int)svm_predict_probability(model_, x_test, probs);
        for (int k = 0; k < model_->nr_class;k++)
        {//����������Ӧ�ĸ���
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

        /*��ʼ��*/
        long    len = x.size();
        long    dim = x[0].size();
        long    elements = len*dim;

        //ת������Ϊlibsvm��ʽ
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
    // ��    ��: ����svmģ��
    // ��    ��: load_model
    // �� �� ��: CxLibSVM::load_model
    // ����Ȩ��: public 
    // ��    ��: string model_path    ģ��·��
    // �� �� ֵ: int 0��ʾ�ɹ���-1��ʾʧ��
    // �� �� ��:
    //************************************
    int load_model(string model_path)
    {
        //�ͷ�ԭ����ģ��
        free_model();
        //����ģ��
        model_ = svm_load_model(model_path.c_str());
        if (model_ == NULL)return -1;
        return 0;
    }

    //************************************
    // ��    ��: ����ģ��
    // ��    ��: save_model
    // �� �� ��: CxLibSVM::save_model
    // ����Ȩ��: public 
    // ��    ��: string model_path    ģ��·��
    // �� �� ֵ: int    0��ʾ�ɹ���-1��ʾʧ��
    // �� �� ��:
    //************************************
    int save_model(string model_path)
    {
        int flag = svm_save_model(model_path.c_str(), model_);
        return flag;
    }

private:

    //************************************
    // ��    ��: �ͷ�svmģ���ڴ�
    // ��    ��: free_model
    // �� �� ��: CxLibSVM::free_model
    // ����Ȩ��: private 
    // �� �� ֵ: void
    // �� �� ��:
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