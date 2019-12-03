import tensorflow as tf
from Hyparams import Hyparams as hp
import os
import pickle

datapath = "/Users/liuhongbing/Documents/tensorflow/git/recommend_system/DIN/Din/"

class Model(object):
    
    def __init__(self):
        with open(datapath + 'cate_list.pkl','rb') as f:
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
        #-- sum begin DNN_first_layers =========================
        # mask the zero padding part
        mask = tf.sequence_mask(self.sl, tf.shape(self.h_emb)[1], dtype=tf.float32) # [B, T]
        mask = tf.expand_dims(mask, -1) # [B, T, 1]
        mask = tf.tile(mask, [1, 1, tf.shape(self.h_emb)[2]]) # [B, T, H]
        self.h_emb *= mask # [B, T, H]
        hist = self.h_emb
        hist = tf.reduce_sum(hist, 1)  # B * H 
        tmp = tf.cast(tf.tile(tf.expand_dims(self.sl,1), [1, hp.hidden_units]), tf.float32) # B * H
        hist = tf.div(hist, tmp)
        print(self.h_emb.get_shape().as_list())
        #-- sum end ---------
        
        hist = tf.layers.batch_normalization(inputs = hist) #  B * H
        hist = tf.reshape(hist, [-1, hp.hidden_units])         # B * H
        hist = tf.layers.dense(hist, hp.hidden_units)          # B * H
        self.u_emb = hist
        #-- fcn begin -------
        din_i = tf.concat([self.u_emb, self.i_emb], axis=-1)  # B * 2H
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1') # B * 80
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')  # B * 80
        self.d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3') 
        self.d_layer_3_i = tf.reshape(self.d_layer_3_i, [-1])
        # DNN end===========================================
        # fm part
        # <W,X> + sigma sigma<vi vj>xi xj
        self.d_layer_fm_i = tf.concat([tf.reduce_sum(self.u_emb*self.i_emb, axis=-1, keep_dims=True),  # B*1
                                        tf.gather(self.u_emb, [0], axis=-1) + tf.gather(self.i_emb, [0], axis=-1)], axis=-1)
        
        
        self.d_layer_fm_i = tf.layers.dense(self.d_layer_fm_i, 1, activation=None, name='f_fm')
        
        self.d_layer_fm_i = tf.reshape(self.d_layer_fm_i, [-1])

        self.logits = self.i_b + self.d_layer_3_i + self.d_layer_fm_i #[B]

        self.score_i = tf.sigmoid(self.i_b + self.d_layer_3_i + self.d_layer_fm_i)
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

        '''
        模型评价 eval阶段
        每个epoch，进行模型验证
        '''
        
    def _eval_model(self):       
        
        self.j = tf.placeholder(tf.int32, [None,])
        
        jc = tf.gather(self.cate_list, self.j)
        j_emb = tf.concat([
                tf.nn.embedding_lookup(self.item_emb_w, self.j),
                tf.nn.embedding_lookup(self.cate_emb_w, jc),
                ], axis=1)
        j_b = tf.gather(self.item_b, self.j)
        din_j = tf.concat([self.u_emb, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_fm_j = tf.concat([tf.reduce_sum(self.u_emb*j_emb, axis=-1, keep_dims=True), tf.gather(self.u_emb, [0], axis=-1) + tf.gather(j_emb, [0], axis=-1)], axis=-1)
        d_layer_fm_j = tf.layers.dense(d_layer_fm_j, 1, activation=None, name='f_fm', reuse=True)
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        d_layer_fm_j = tf.reshape(d_layer_fm_j, [-1])
        x = self.i_b - j_b + self.d_layer_3_i - d_layer_3_j + self.d_layer_fm_i - d_layer_fm_j # [B]

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_j = tf.sigmoid(j_b + d_layer_3_j + d_layer_fm_j)
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

    def eval_score(self, sess, uij):
        score_tmp = sess.run([self.score_i], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.hist_i: uij[2],
                self.sl: uij[3],
                })
        return score_tmp
        

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


