import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest, DataInputPredict
from model_fancy import Model
from Hyparams import Hyparams as hp

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
best_auc = 0.0

datapath = "/Users/liuhongbing/Documents/tensorflow/git/recommend_system/DIN/Din/"
with open(datapath+'dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    # cate_list = pickle.load(f)
    # user_count, item_count, cate_count = pickle.load(f)


def calc_auc(raw_arr):
    """
    Summary
    Args:raw_arr (TYPE): Description
    Returns:TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:    # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
    score_p = score[:,0]
    score_n = score[:,1]
    #print "============== p ============="
    #print score_p
    #print "============== n ============="
    #print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


def _eval(sess, model,step,saver):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, hp.test_batch_size):
        auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        print("begin save model")
        saver.save(sess, hp.save_dir+'/ckpt', step)
    return test_gauc, Auc


def _eval2(sess, model):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, hp.test_batch_size):
        auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    return test_gauc, Auc



def _test(sess, model):
    auc_sum = 0.0
    score_arr = []
    predicted_users_num = 0
    print("test sub items")
    for _, uij in DataInputTest(test_set, hp.predict_batch_size):
        if predicted_users_num >= hp.predict_users_num:
            break
        score_ = model.test(sess, uij)
        score_arr.append(score_)
        predicted_users_num += hp.predict_batch_size
    return score_[0]

'''
单线程访问接口时间：对100个广告进行排序
10ms以上占比：941/194202 = 0.25%
10ms以下占比：99.75%
平均响应时间：0.0016s == 1.6ms
最大响应时间：0.072832 == 72.832ms
最小响应时间：0.001023 == 1.023ms

'''
def main_predict_time_cost():
    print("start load test data .....")
    dip = DataInputPredict(test_set)
    u_all, i_all, sl_all, hist_all = dip.get_input()

    print("start load trained model ....")
    model_din = Model()
    time_cost = []
    with tf.Session() as sess:
        model_din._initialize_session(sess)
        if not model_din.trained:
            print("Please train the model first! (./train.py -g)")
            sys.exit(1)
        test_gauc, Auc = _eval2(sess, model_din)
        print('Eval_GAUC: %.4f\tEval_AUC: %.4f' % (test_gauc, Auc))
        cate_emb_w_tmp = sess.run(model_din.item_emb_w)
        avetime = 0
        for ind in range(len(u_all)):
            start_time = time.time()
            score_i_tmp = sess.run(model_din.score_i,
                                feed_dict = {model_din.u: u_all[ind], 
                                model_din.i : i_all[ind],
                                model_din.hist_i:hist_all[ind],
                                model_din.sl:sl_all[ind] })
    
            result_dict = dict(zip(i_all[ind], np.reshape(score_i_tmp,[-1])))
            result_sort = sorted(result_dict, key=result_dict.__getitem__, reverse=True)
            time_one = time.time()-start_time
            avetime += time_one
            print("cost:",time_one,"秒!")
            time_cost.append(time_one)
        
#            if ind >= 10:
#                break
        print("cost_average:", 1.0*avetime/len(u_all))
        return time_cost

def predict_init_model():
    '''
    tensorflow模型初始化内存中
    '''
    return True


def get_predict_input(userId, hist,target_list):
    ad_predict_num = len(target_list)
    userlist = [userId]*ad_predict_num
    histlist = np.reshape(hist * ad_predict_num,[ad_predict_num,-1])
    sl_list = [len(hist)]*ad_predict_num
    return userlist, histlist,sl_list

'''
每次调用接口，都要加载模型
'''
def main_predict_online(userId, hist, target_list, topN):

    userlist, histlist,sl_list = get_predict_input(userId, hist, target_list)
    
    print("start load model .....")
    model_din = Model()
    with tf.Session() as sess:
        model_din._initialize_session(sess)
        if not model_din.trained:
            print("Please train the model first! (./train.py -g)")
            sys.exit(1)
        score_i_tmp = sess.run(model_din.score_i,
                    feed_dict = {model_din.u: userlist, 
                    model_din.i : target_list,
                    model_din.hist_i:histlist,
                    model_din.sl:sl_list })
        result_dict = dict(zip(target_list, np.reshape(score_i_tmp,[-1])))
        result_sort = sorted(result_dict, key=result_dict.__getitem__, reverse=True)
        
    return result_sort[:topN]

                    

def main_train():
    tf.reset_default_graph()
    model = Model()
#    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('test_gauc: %.4f\t test_auc: %.4f' % _eval2(sess, model))
        sys.stdout.flush()
        lr = 1.0
        start_time = time.time()
        for _ in range(50):
            random.shuffle(train_set)
            epoch_size = round(len(train_set) / hp.train_batch_size)
            loss_sum = 0.0
            for _, uij in DataInput(train_set, hp.train_batch_size):
                loss = model.train(sess, uij, lr)
                loss_sum += loss
    
                if model.global_step.eval() % 10000 == 0:
                    test_gauc, Auc = _eval(sess, model, model.global_step.eval(),model.saver)
                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                                (model.global_epoch_step.eval(), model.global_step.eval(),
                                 loss_sum / 1000, test_gauc, Auc))
                    sys.stdout.flush()
                    loss_sum = 0.0
    
                if model.global_step.eval() % 336000 == 0:
                    lr = 0.1
    
            print('Epoch %d DONE\tCost time: %.2f' %
                        (model.global_epoch_step.eval(), time.time()-start_time))
            sys.stdout.flush()
            model.global_epoch_step_op.eval()
    
        print('best test_gauc:', best_auc)
        sys.stdout.flush()



if __name__=='__main__':
    main_train()
    # #time_cost = main_predict_time_cost()
    # userId = 1
    # hist = [[28839, 32507, 34101, 34339],[42354, 19191, 13067, 33134, 33800, 58952],[ 2923,  5389, 10020, 15366, 17051, 32556,  7069]]
    # target = [50997,	 28883,	 7657,	 490,	 5940,	 59704,	 61781,	 61567,	 52887,	 38156,	 2288,	 44011,	 45422,	 5500,	 6450,	 50232,	 23241,	 15519,	 1142,	 2019,	 51693,	 1038,	 22681,	 42488,	 40847,	 31735,	 40358,	 30379,	 9735,	 5982,	 11999,	 46754,	 7498,	 55401,	 958,	 32970,	 31899,	 57704,	 16372,	 4231,	 43729,	 35460,	 59793,	 30533,	 4493,	 39417,	 44245,	 5828,	 33753,	 37945,	 3012,	 17667,	 54026,	 36466,	 4133,	 42246,	 19819,	 31525,	 55718,	 23280,	 17462,	 16328,	 42957,	 61188,	 13109,	 29713,	 57446,	 34744,	 43608,	 1264,	 61712,	 33298,	 4626,	 378,	 21856,	 9422,	 46531,	 30987,	 43080,	 24729,	 10951,	 3550,	 4996,	 38504,	 32543,	 10748,	 4936,	 36525,	 14096,	 9453,	 1777,	 61438,	 22690,	 50526,	 7267,	 62755,	 34713,	 9255,	 43942,	 56743]
    # result = main_predict_online(userId, hist[0], target, 20)

    
    