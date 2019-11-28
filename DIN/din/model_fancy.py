import tensorflow as tf
from Hyparams import Hyparams as hp
import os
import pickle


class Model(object):
    
    def __init__(self):
        with open('cate_list.pkl','rb') as f:
            self.cate_list = pickle.load(f)
        self._build_graph()
        if not os.path.exists(hp.save_dir):
            os.mkdir(hp.save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self.trained = False

    def _create_input_tensor(self):
        self.u = tf.placeholder(tf.int32, [None,],name="user_id") # [B]
        self.i = tf.placeholder(tf.int32, [None,],name="item_i")  # [B]
        self.y = tf.placeholder(tf.float32, [None,],name="label") # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None],name="history") # [B, T]
        self.sl = tf.placeholder(tf.int32, [None,],name="sequence_length")  # [B]
        self.lr = tf.placeholder(tf.float64, [],name="learning_rate")

    def _create_weight_matrix(self):
        # user_embedding: 192403 * H
        #user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        # item_embedding: 63001 * H//2
        self.item_emb_w = tf.get_variable("item_emb_w", [hp.item_count, hp.hidden_units // 2])
        # item bias : 63001
        self.item_b = tf.get_variable("item_b", [hp.item_count],initializer=tf.constant_initializer(0.0))
        # item_embedding : 801 * H//2
        self.cate_emb_w = tf.get_variable("cate_emb_w", [hp.cate_count, hp.hidden_units // 2])

        self.cate_list = tf.convert_to_tensor(self.cate_list, dtype=tf.int64)

        
    def _train_model(self):
        '''
        计算 userEmbedding：通过预测广告id与历史行为的embedding 加权求和
        '''
        # item_i_embed + cate_i_embed ---> B * H
        ic = tf.gather(self.cate_list, self.i)
        self.i_emb = tf.concat(values = [
                tf.nn.embedding_lookup(self.item_emb_w, self.i),
                tf.nn.embedding_lookup(self.cate_emb_w, ic),
                ], axis=1)
        self.i_b = tf.gather(self.item_b, self.i)
        
        # item_hist_i_embed + cate_hist_i_embed ---> B * sl * H
        hc = tf.gather(self.cate_list, self.hist_i)

        self.h_emb = tf.concat([
                tf.nn.embedding_lookup(self.item_emb_w, self.hist_i),
                tf.nn.embedding_lookup(self.cate_emb_w, hc),
                ], axis=2)
        hist_i =attention(self.i_emb, self.h_emb, self.sl)    # B * 1 * H
        #-- attention end ---
        hist_i = tf.layers.batch_normalization(inputs = hist_i)
        hist_i = tf.reshape(hist_i, [-1, hp.hidden_units], name='hist_bn') # B*H
        hist_i = tf.layers.dense(hist_i, hp.hidden_units, name='hist_fcn')
        u_emb_i = hist_i    ## batch_size * 128
        
        print(u_emb_i.get_shape().as_list())
        print(self.i_emb.get_shape().as_list())
        
        #-- fcn begin -------
        '''
        模型训练 model
        【userembedding + item_embedding】--> FC_lay1[dim->80]-->FC_lay2[dim->40]-->FC_lay3 [dim->1]
        '''
        din_i = tf.concat([u_emb_i, self.i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        #if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
        # d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
        # d_layer_1_i = dice(d_layer_1_i, name='dice_1')
        # d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        #d_layer_2_i = dice(d_layer_2_i, name='dice_2')
        self.d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        
        self.d_layer_3_i = tf.reshape(self.d_layer_3_i, [-1])
        
        self.logits = self.i_b + self.d_layer_3_i

        self.score_i = tf.sigmoid(self.i_b + self.d_layer_3_i)
        self.score_i = tf.reshape(self.score_i, [-1, 1])

    def _train_optimize(self):
        '''
        模型优化目标函数
        '''
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.logits,
                        labels=self.y)
                )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

        
    def _build_graph(self):
        print("start gen input tensor...")
        self._create_input_tensor()
        print("start gen weight matrix...")
        self._create_weight_matrix()
        print("start gen train model...")
        self._train_model()
        print("start gen eval model...")
        self._eval_model()
        print("start gen optimize ...")
        self._train_optimize()
        print("graph gen end ...")

#        '''
#        测试代码：模型 批量测试: Batch * N * T
#        模型测试
#        '''
#        # prediciton for selected items
#        # logits for selected item:
#        # item_emb_w:                                                                            
#        # tf.nn.embedding_lookup(cate_emb_w, cate_list) :    
#        # 总的    item_Emb + cate_Emb
#        item_emb_all = tf.concat([
#                item_emb_w,
#                tf.nn.embedding_lookup(cate_emb_w, cate_list)
#                ], axis=1)
#        
#        # 取得要预测的前 100
#        item_emb_sub = item_emb_all[:predict_ads_num,:] # 100 * H
#        item_emb_sub = tf.expand_dims(item_emb_sub, 0)  # 1 * 100 * H
#        item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1]) # predict_batch * 100 * H
#
#        hist_sub =attention_multi_items(item_emb_sub, h_emb, self.sl)    # predict_batch * 100 * 1 * H
#        #-- attention end ---
#        
#        hist_sub = tf.layers.batch_normalization(inputs = hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
#        # print hist_sub.get_shape().as_list() 
#        hist_sub = tf.reshape(hist_sub, [-1, hidden_units]) # predict_batch 100  * H
#        hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE) # predict_batch 100 * H
#        u_emb_sub = hist_sub
#
#        item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])  # predict_batch 100 * H
#        din_sub = tf.concat([u_emb_sub, item_emb_sub], axis=-1)      # predict_batch 200 * H 
#
#        din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True) # predict_batch 200 *H
#        d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True) # predict_batch 200 *80
#        #d_layer_1_sub = dice(d_layer_1_sub, name='dice_1_sub')
#        d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.sigmoid, name='f2', reuse=True) # predict_batch 200 *40
#        #d_layer_2_sub = dice(d_layer_2_sub, name='dice_2_sub')
#        d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True) # predict_batch 200 *1
#        d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num]) # predict_batch *2 100
#
#        self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
#        self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
#        #-- fcn end -------
      
    def _eval_model(self):       
        '''
        模型eval  model
        neg embdeding + userEmbedding FC_lay1[dim->80]-->FC_lay2[dim->40]-->FC_lay3 [dim->1]
        计算：auc、pos_i_score、neg_j_score
        '''
        self.j = tf.placeholder(tf.int32, [None,],name="item_j") # [B]
            #    item_j_embed + cate_j_embed ---> B * H
        jc = tf.gather(self.cate_list, self.j)
        j_emb = tf.concat([
                tf.nn.embedding_lookup(self.item_emb_w, self.j),
                tf.nn.embedding_lookup(self.cate_emb_w, jc),
                ], axis=1)
        j_b = tf.gather(self.item_b, self.j)
        
        hist_j =attention(j_emb, self.h_emb, self.sl)
          #-- attention end ---
        hist_j = tf.layers.batch_normalization(inputs = hist_j)
        hist_j = tf.reshape(hist_j, [-1, hp.hidden_units], name='hist_bn')
        hist_j = tf.layers.dense(hist_j, hp.hidden_units, name='hist_fcn', reuse=True)
        u_emb_j = hist_j    ##    ## B * H
        print(u_emb_j.get_shape().as_list())
        print(j_emb.get_shape().as_list())
        
        din_j = tf.concat([u_emb_j, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        # d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
        # d_layer_1_j = dice(d_layer_1_j, name='dice_1')
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        # d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
        # d_layer_2_j = dice(d_layer_2_j, name='dice_2')
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1]) # 
        
        x = self.i_b - j_b + self.d_layer_3_i - d_layer_3_j # [B]
    
        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
    
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        print(self.p_and_n.get_shape().as_list())
        
    def _initialize_session(self, session):
        checkpoint = tf.train.get_checkpoint_state(hp.save_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            session.run(init_op)
        else:
            self.saver.restore(session, checkpoint.model_checkpoint_path)
            self.trained = True


    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.y: uij[2],
                self.hist_i: uij[3],
                self.sl: uij[4],
                self.lr: l,
                })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.j: uij[2],
                self.hist_i: uij[3],
                self.sl: uij[4],
                })
        return u_auc, socre_p_and_n
    
    def test(self, sess, uij):
        return sess.run(self.logits_sub, feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.j: uij[2],
                self.hist_i: uij[3],
                self.sl: uij[4],
                })
    

def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

def attention(queries, keys, keys_length):
    '''
        queries:         [B, H]
        keys:                [B, T, H]
        keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]    # H 128
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])                # B * TH
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units]) # B*T*H
    din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1) # B*T* 4H
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE) # B*T*80
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE) # B*T*40
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE) # B*T*1
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])    # B*1*T
    outputs = d_layer_3_all 
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])     # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)    # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)    # [B, 1, T]

    # Weighted sum   T
    outputs = tf.matmul(outputs, keys)    # [B, 1, H]

    return outputs

def attention_multi_items(queries, keys, keys_length):
        
    '''
        queries:     [B, N, H]  N is the number of ads
        keys:        [B, T, H] 
        keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1] # H
    queries_nums = queries.get_shape().as_list()[1]         # N
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])   # B*N*HT
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units]) # shape : [B, N, T, H]
    max_len = tf.shape(keys)[1]                 # T
    keys = tf.tile(keys, [1, queries_nums, 1])  # B*NT*H
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units]) # shape : [B, N, T, H]
    din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all 
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)     # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len]) # shape : [B, N, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)    # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)    # [B, N, 1, T]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
    #print outputs.get_shape().as_list()
    #print keys.get_sahpe().as_list()
    # Weighted sum
    outputs = tf.matmul(outputs, keys)
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])    # [B, N, 1, H]
    print(outputs.get_shape().as_list())
    return outputs
