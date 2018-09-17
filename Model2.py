# *-encoding=utf-8

import sys
import os
import time
import numpy as np
import tensorflow as tf
import data_utils
from colorama import *
import attention
import re



class Model(object):

    def __init__(self, num_tasks = 1, num_classes=2, vocab_size=5000,
                 state_size=512, num_layers=2, max_length=100, max_gradient_norm=5.0,
                 learning_rate=0.001,learning_rate_decay=0.5,
                 drop_out=0.0,l2_regulizer=0,
                 is_train = True,
                 task_name=None,
                 pretrained_embedding_file=None,
                 vocab_file=None):
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.max_gradient_norm = max_gradient_norm
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.drop_out = drop_out
        self.l2_regulizer = l2_regulizer
        self.global_step = tf.Variable(0, trainable=False)
        self.is_train = is_train
        self.task_name= ['task_%d'%i for i in xrange(num_tasks)] if task_name is None else task_name
        self.pretrained_embedding_file = pretrained_embedding_file
        self.vocab_file = vocab_file

    def create_session(self):
        self.session = tf.Session()

    def build_graph(self):
        
        #输入数据
        self.inputs = tf.placeholder(tf.int32, [self.max_length,None]) #句子
        print 'inputs\t',self.inputs.get_shape(),'\n'+''.join(['-'*50])
        self.input_lengths = tf.placeholder(tf.int32, [None]) #句子长度
        print 'input lengths\t',self.input_lengths.get_shape(),'\n'+''.join(['-'*50])
        self.input_weights = tf.placeholder(tf.float32, [self.max_length,None,self.num_tasks]) # 句子长度掩模
        print 'input weights\t',self.input_weights.get_shape(),'\n'+''.join(['-'*50])
        self.sparse_labels = tf.placeholder(tf.int32, [self.num_tasks, None])# 类别
        print 'sparse labels\t',self.sparse_labels.get_shape(),'\n'+''.join(['-'*50])
        self.expanded_sparse_labels = tf.unstack(self.sparse_labels,axis=0)
        self.label_weights = tf.placeholder(tf.float32, [self.num_tasks, None])# 类别掩模
        print 'label weights\t',self.sparse_labels.get_shape(),'\n'+''.join(['-'*50])
        self.expand_label_weights = tf.unstack(self.label_weights,axis=0)

        #词向量
        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable('word_embedding', [self.vocab_size,self.state_size])#词向量表
            self.embedding_inputs= tf.nn.embedding_lookup(self.embedding,self.inputs)#输入文本查表翻译成词向量
        print 'embedding table\t', self.embedding.get_shape(),'\n'+''.join(['-'*50])
        print 'embdeding inputs\t', self.embedding_inputs.get_shape(),'\n'+''.join(['-'*50])

        #网络
        cell = tf.contrib.rnn.GRUCell(self.state_size,activation=tf.nn.relu)#RNN
        droped_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0-self.drop_out)#增加dropout
        if self.num_layers>1:
            cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers)#增加多层网络
            droped_cell = tf.contrib.rnn.MultiRNNCell([droped_cell]*self.num_layers)

        #attention
        self.task_attentions = tf.get_variable('task_attentions', [self.state_size,self.num_tasks]) #增加attention
        print 'attention\t', self.task_attentions.get_shape(),'\n'+''.join(['-'*50])

        #分类器
        self.w = [tf.get_variable("w_%d"%i, [self.state_size, self.num_classes]) for i in xrange(self.num_tasks)]#系数
        self.b = [tf.get_variable("b_%d"%i, [self.num_classes]) for i in xrange(self.num_tasks)]#截距
        print 'w\t', tf.stack(self.w,axis=0).get_shape(),'\n'+''.join(['-'*50])
        print 'b\t', tf.stack(self.b,axis=0).get_shape(),'\n'+''.join(['-'*50])

        #前向运行网络
        self.rnn_outputs,_ = tf.nn.dynamic_rnn(droped_cell if self.drop_out>0.0 and self.is_train else cell , self.embedding_inputs,self.input_lengths,dtype = tf.float32,time_major=True) #[length, batch, hidden]
        print 'rnn outputs\t',self.rnn_outputs.get_shape(),'\n'+''.join(['-'*50])

        #加权平均attention获取分类输入
        attention_vecs = []#num_tasks * batch_size * hidden_size 加权和 
        self.alphas = []#num_tasks * length * batch_size 权重分布
        for k in xrange(self.num_tasks):
            attention_vec, alpha = attention.attention(self.rnn_outputs,self.state_size,name='attention_%d'%k)
            attention_vecs.append(attention_vec)
            self.alphas.append(alpha)
        print 'attention vecs\t', tf.stack(attention_vecs,axis=0).get_shape(),'\n'+''.join(['-'*50])
        print 'alphas\t', tf.stack(self.alphas,axis=0).get_shape(),'\n'+''.join(['-'*50])
        
        #概率
        self.logits = [tf.matmul(attention_vecs[i],self.w[i])+self.b[i] for i in xrange(self.num_tasks)]#num_tasks * batch * num_classes
        print 'logits\t',tf.stack(self.logits,axis=0).get_shape(),'\n'+''.join(['-'*50])

        #误差
        self.losses=[]
        for k in xrange(self.num_tasks):
            logit_k = self.logits[k]# batch * num_classes
            label_k = self.expanded_sparse_labels[k]# batch * num_classes
            weight_k = self.label_weights[k]# 
            #loss_k = tf.losses.sparse_softmax_cross_entropy(labels=label_k,logits=logit_k,weights=weight_k)
            loss_k = tf.losses.sparse_softmax_cross_entropy(labels=label_k,logits=logit_k,weights=weight_k)
            self.losses.append(loss_k)
        self.loss = tf.reduce_mean(self.losses)
        
        #正则
        self.regulizer_w=tf.nn.l2_loss(self.w)
        self.regulizer_b=tf.nn.l2_loss(self.b)
        self.regulizer_a=tf.nn.l2_loss(self.task_attentions)
        self.regulizer = self.regulizer_w+self.regulizer_b+self.regulizer_a
        
        #梯度
        params = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss+self.regulizer*self.l2_regulizer, params)
        self.clipped_gradients, norm = tf.clip_by_global_norm(self.gradients,self.max_gradient_norm)

        #优化参数
        self.learning_rate = tf.Variable(float(self.learning_rate),trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay)
        self.update = self.opt.apply_gradients(zip(self.clipped_gradients, params), global_step=self.global_step)

        #计算预测和准确率
        self.predicts = []# num_tasks * batch
        corrects=[]
        self.task_accuracys=[]
        for i in xrange(self.num_tasks):
            predict = tf.cast(tf.argmax(self.logits[i], axis=1),"int32")# batch_size
            predict = tf.multiply(predict, tf.cast(self.expand_label_weights[i],'int32'))#无意义类别置0
            correct = tf.cast(tf.equal(predict, tf.unstack(self.sparse_labels, axis=0)[i]),"float")
            correct = tf.multiply(correct, self.expand_label_weights[i])#无意义类别置0
            corrects.append(correct)
            accuracy = tf.reduce_sum(correct)/tf.reduce_sum(self.expand_label_weights[i]) 
            self.predicts.append(predict)
            self.task_accuracys.append(accuracy)
        self.total_accuracy = tf.reduce_sum(tf.stack(corrects,axis=0))/tf.reduce_sum(self.label_weights)
        print 'predicts', tf.stack(self.predicts,axis=0).get_shape(),'\n'+''.join(['-'*50])

        #保存器
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
        
        # if(self.is_train):
        #tensorboard
        summary_loss = tf.summary.scalar('loss',self.loss)
        summary_task_accs = [tf.summary.scalar('task_%s_acc'%(self.task_name[i]),self.task_accuracys[i]) for i in xrange(self.num_tasks)]
        summary_total_accs = tf.summary.scalar('total_acc', self.total_accuracy)
        summary_learning_rate =  tf.summary.scalar('learning_rate',self.learning_rate)
        self.summary = tf.summary.merge_all()

    def initilize(self,model_dir='save'):
        if not os.path.exists(model_dir):
            os.system('mkdir %s'%model_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+'.index'):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:    
            print("Creating model with fresh parameters.")
            self.session.run(tf.global_variables_initializer())
            if self.pretrained_embedding_file is not None:
                self.load_pretrained_embedding(self.pretrained_embedding_file, self.vocab_file)
        # if(self.is_train):
        self.train_summary_writer = tf.summary.FileWriter(model_dir+'/train')
        self.validation_summary_writer = tf.summary.FileWriter(model_dir+'/validation')
        
    def load_pretrained_embedding(self,w2v_file,vocab_file):
        e = np.zeros((self.vocab_size,self.state_size),float)
        vocab = open(vocab_file).read().decode('utf-8').strip().split('\n')
        inv_vocab = {w:i for i,w in enumerate(vocab)}
        pretrained = []
        unfind=[]
        re_word = re.compile('^([^0-9]+) -?[0-9]+')
        for i,line in enumerate(open(w2v_file)):
            line = line.rstrip('\n').decode('utf-8')
            if i==0:
                word_num, embedding_size = map(int,line.split())
                if embedding_size!=self.state_size:
                    sys.stderr.write('%s詞向量维度不匹配!\n'%(w2v_file))
                    return
            else:
                m = re_word.search(line)
                word = m.group(1)
                embedding = np.array(map(float, line[m.end(0):].split()))
                pretrained.append(embedding)
                if not word in inv_vocab:
                    unfind.append(i)
                    continue
                word_id = inv_vocab[word]
                e[word_id,:] = embedding
        pretrained = np.array(pretrained)
        mean = np.mean(pretrained,axis=0)
        cov = np.cov(np.transpose(pretrained,[1,0]))
        pretrained_embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.state_size])
        embedding_init_op = self.embedding.assign(pretrained_embedding_placeholder)
        print 'loading pretrained word embedding'
        self.session.run(embedding_init_op, feed_dict={pretrained_embedding_placeholder: e})


    def fit(self, Xtrain,Ytrain,Xvalid,Yvalid,batch_size=32,steps_per_checkpoint=100,model_dir='save',verbose=100):
        self.create_session()
        self.build_graph()
        self.initilize()
        mean_train_losses,train_losses,train_accs,val_losses,val_accs=[],[],[],[],[]
        xv,xlv,xwv,yv,ywv = self.get_batch(Xvalid,Yvalid,len(Xvalid))
        while True:
            x,xl,xw,y,yw = self.get_batch(Xtrain,Ytrain,batch_size)
            feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
            logits,loss,gradients,task_accs, total_acc,_,summary= self.session.run([self.logits,self.loss,self.gradients,self.task_accuracys, self.total_accuracy,self.update,self.summary],feed)
            step = self.global_step.eval(self.session)
            train_losses.append(loss)
            train_accs.append(total_acc)
            self.train_summary_writer.add_summary(summary,step)
            self.train_summary_writer.flush()
            if verbose is not None and step%verbose==0:
                self.is_train = False
                feed={self.inputs:xv, self.input_lengths:xlv,self.input_weights: xwv ,self.label_weights:ywv,self.sparse_labels:yv}
                logits,loss,task_accs,total_acc,summary = self.session.run([self.logits,self.loss,self.task_accuracys,self.total_accuracy,self.summary],feed)
                self.is_train = True
                val_losses.append(loss)
                val_accs.append(total_acc)
                mean_train_losses.append(np.mean(train_losses[-verbose:]))
                self.validation_summary_writer.add_summary(summary,step)
                self.validation_summary_writer.flush()
                print 'step %05d\tlearning-rate %.4f\ttrain-loss %.4f\ttrain-acc %.4f\tval-loss %.4f\tval-acc %.4f'%(step,self.learning_rate.eval(self.session),mean_train_losses[-1],np.mean(train_accs[-verbose:]),val_losses[-1],val_accs[-1]),'\n'+''.join(['-'*100])
                #if len(mean_train_losses)>3 and mean_train_losses[-1]> max(mean_train_losses[-4:-1]):
                #    self.session.run([self.learning_rate_decay_op],{})
            if step%steps_per_checkpoint==0:
                self.saver.save(self.session, os.path.join(model_dir,'checkpoint'),global_step = step )

    def get_batch(self, X,Y,batch_size):
        inputs = np.full((self.max_length,batch_size),data_utils.PAD_ID,int)#字符串序列
        input_lengths = np.zeros(batch_size,int)#字符串长度
        input_weights = np.zeros((self.max_length, batch_size, self.num_tasks),float)#字符串长度掩模，用于置0长度之外的attention权重
        label_weights = np.ones((self.num_tasks, batch_size),float)#类别标签掩模，用于消除无意义类别对loss的影响
        sparse_labels = np.zeros((self.num_tasks, batch_size),int)#类别标签

        choiceids = np.random.choice(range(0,len(X)),size= batch_size,replace=False)
        choiceids = sorted(choiceids)
        x = [X[i] for i in choiceids]
        y = [Y[i] for i in choiceids]
        for i,xi in enumerate(x):
            input_lengths[i] = len(xi)
            input_weights[:len(xi),i,:]=1.0
            for j,wj in enumerate(xi):
                inputs[j,i]=wj
            for k, yk in enumerate(y[i]):
                if yk!=-1:
                    sparse_labels[k,i]=yk
                else:
                    label_weights[k,i]=0.0
        return inputs,input_lengths,input_weights,sparse_labels,label_weights 

    def get_batch_random(self, X,Y,batch_size):
        inputs = np.full((self.max_length,batch_size),data_utils.PAD_ID,int)#字符串序列
        input_lengths = np.zeros(batch_size,int)#字符串长度
        input_weights = np.zeros((self.max_length, batch_size, self.num_tasks),float)#字符串长度掩模，用于置0长度之外的attention权重
        label_weights = np.ones((self.num_tasks, batch_size),float)#类别标签掩模，用于消除无意义类别对loss的影响
        sparse_labels = np.zeros((self.num_tasks, batch_size),int)#类别标签

        choiceids = np.random.choice(range(0,len(X)),size= batch_size,replace=False)
        choiceids = sorted(choiceids)
        x = [X[i] for i in choiceids]
        y = [Y[i] for i in choiceids]
        for i,xi in enumerate(x):
            input_lengths[i] = len(xi)
            input_weights[:len(xi),i,:]=1.0
            for j,wj in enumerate(xi):
                inputs[j,i]=wj
            for k, yk in enumerate(y[i]):
                if yk!=-1:
                    sparse_labels[k,i]=yk
                else:
                    label_weights[k,i]=0.0
        return inputs,sparse_labels

    def predict(self, X):
        emptyY = [[0]*self.num_tasks for i in xrange(len(X))]
        x,xl,xw,y,yw = self.get_batch(X,emptyY,len(X))
        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        predicts, alphas = self.session.run([self.predicts,self.alphas],feed)
        return np.array(predicts), np.array(alphas)

    def display_attention(self, X,Y,vocab):
        x,xl,xw,y,yw = self.get_batch(X,Y,len(X))

        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        alphas,predicts,acc = self.session.run([self.alphas,self.predicts,self.total_accuracy],feed)
        alphas = np.array(alphas)#num_tasks * length * batch_size 权重分布
        predicts = np.array(predicts)
        for i,xi in enumerate(X):
            predict = predicts[:,i]
            print (Style.BRIGHT +u'评论'+'\t'+Style.NORMAL +' '.join([vocab[w].decode('utf-8') for w in xi]))
            print (Style.BRIGHT +u'属性'+'\t'+Style.NORMAL +'\t'.join([self.task_name[index] for index,yij in enumerate(Y[i]) if yij !=-1]))
            print Style.BRIGHT +u'真实'+'\t'+Style.NORMAL +'\t'.join([data_utils.label_dict[yij] for yij in Y[i] if yij != -1])
            print Style.BRIGHT +u'预测'+'\t'+Style.NORMAL +'\t'.join([data_utils.label_dict[predict[index]] for index,yij in enumerate(Y[i]) if yij !=-1])
            for k,p in enumerate(Y[i]):
                if p == -1:
                    continue
                attention = alphas[k,:,i]
                if predict[k]!=3:
                    print self.task_name[k],data_utils.label_dict[predict[k]]
                    attn_color = {0:Fore.RED,1:Fore.GREEN,2:Fore.YELLOW}[predict[k]]
                    out=''
                    for n,w in enumerate(xi):
                        if  attention[n]>3.0/len(xi):
                            out+= Style.BRIGHT +  attn_color +vocab[w]+' '
                        elif attention[n]>1.0/len(xi):
                            out+= Style.NORMAL + attn_color +vocab[w]+' '
                        elif attention[n]>0.5/len(xi):
                            out+= Style.DIM + attn_color +vocab[w]+' '
                        else:
                            out+= Style.NORMAL + Fore.RESET + vocab[w]+' '
                    print out
            
            print Style.RESET_ALL+ Fore.RESET+'\n-----------------------------------------------\n'
        print acc
        return predicts,acc

    def display_attention_subsystem(self, X,Y,vocab):
        x,xl,xw,y,yw = self.get_batch(X,Y,len(X))

        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        alphas,predicts,acc = self.session.run([self.alphas,self.predicts,self.total_accuracy],feed)
        alphas = np.array(alphas)#num_tasks * length * batch_size 权重分布
        predicts = np.array(predicts)
        for i,xi in enumerate(X):
            predict = predicts[:,i]
            # print (Style.BRIGHT +u'评论'+'\t'+Style.NORMAL +' '.join([vocab[w].decode('utf-8') for w in xi]))
            # print (Style.BRIGHT +u'属性'+'\t'+Style.NORMAL +'\t'.join([self.task_name[index] for index,yij in enumerate(Y[i]) if yij !=-1]))
            # print Style.BRIGHT +u'真实'+'\t'+Style.NORMAL +'\t'.join([data_utils.label_dict[yij] for yij in Y[i] if yij != -1])
            # print Style.BRIGHT +u'预测'+'\t'+Style.NORMAL +'\t'.join([data_utils.label_dict[predict[index]] for index,yij in enumerate(Y[i]) if yij !=-1])
            for k,p in enumerate(Y[i]):
                if p == -1:
                    continue
                attention = alphas[k,:,i]
                if predict[k]!=3:
                    # print self.task_name[k],data_utils.label_dict[predict[k]]
                    attn_color = {0:Fore.RED,1:Fore.GREEN,2:Fore.YELLOW}[predict[k]]
                    out=''
                    for n,w in enumerate(xi):
                        if  attention[n]>3.0/len(xi):
                            out+= Style.BRIGHT +  attn_color +vocab[w]+' '
                        elif attention[n]>1.0/len(xi):
                            out+= Style.NORMAL + attn_color +vocab[w]+' '
                        elif attention[n]>0.5/len(xi):
                            out+= Style.DIM + attn_color +vocab[w]+' '
                        else:
                            out+= Style.NORMAL + Fore.RESET + vocab[w]+' '
                    # print out
            
            # print Style.RESET_ALL+ Fore.RESET+'\n-----------------------------------------------\n'
        # print acc
        return predicts,acc

    def predict_api(self, X,vocab,ids):
        emptyY = [[0]*self.num_tasks for i in xrange(len(X))]
        XI =[]
        for x in X:
            xIds = []
            for xid in x:
                xIds.append(xid[0])
            XI.append(xIds)
        # print XI

        x,xl,xw,y,yw = self.get_batch(XI,emptyY,len(X))

        
        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        alphas,predicts = self.session.run([self.alphas,self.predicts],feed)
        alphas = np.array(alphas)#num_tasks * length * batch_size 权重分布
        predicts = np.array(predicts)

        results = []
        for i,xi in enumerate(X):  
            predict = predicts[:,i]
            sentences = self.get_sentence(xi, vocab)
            tmp = 0
            for j,pj in enumerate(emptyY[i]):
                if pj == -1:
                    continue
                attention = alphas[j,:,i]
                result={}
                
                if predicts[j] < 3:
                    result['id'] = str(ids[i])
                    result['dimid'] = str(j+1)
                    result['value'] =str(predicts[j])
                    attentions = self.get_attentions(attention, xi, vocab, sentences)
                    result['att'] = attentions
                    results.append(result)
                elif pj ==3:
                    tmp = tmp + 1
            # print tmp
            if tmp == 12:
                result={}
                result['id'] = str(ids[i])
                result['dimid'] = str(-1)
                result['value'] = str(-1)
                result['att'] = None
                results.append(result)
        return results

    def predict_pro_api(self, X, vocab, ids, pros, pro_label_ids_dict):
        # emptyY = [[0]*self.num_tasks for i in xrange(len(X))]
        XI =[]
        Y = []
        # pro_labels = [key for key,value in inv_label_ids_dict.iteritems()]
        
        for i,x in enumerate(X):
            xIds = []
            for xid in x:
                xIds.append(xid[0])
            XI.append(xIds)
            label_ids_dict = pro_label_ids_dict[pros[i]]
            # print label_ids_dict
            pro_labels = [key for key,value in label_ids_dict.iteritems()]
            y_label = []
            for label in self.task_name:
                if label in pro_labels:
                    y_label.append(0)
                else :
                    y_label.append(-1)
            Y.append(y_label)        

        # print Y

        x,xl,xw,y,yw = self.get_batch(XI,Y,len(X))

        
        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        alphas,predicts = self.session.run([self.alphas,self.predicts],feed)
        alphas = np.array(alphas)#num_tasks * length * batch_size 权重分布
        predicts = np.array(predicts)

        results = []
        for i,xi in enumerate(X):  
            predict = predicts[:,i]
            sentences = self.get_sentence(xi, vocab)
            tmp = 0
            label_ids_dict = pro_label_ids_dict[pros[i]]
            for j,pj in enumerate(predict):
                if Y[i][j] == -1:
                    continue
                attention = alphas[j,:,i]
                result={}
                
                if pj < 3:
                    result['id'] = str(ids[i])
                    result['proid'] = pros[i]
                    result['dimid'] = label_ids_dict[self.task_name[j]]
                    result['value'] =str(predict[j])
                    attentions = self.get_attentions(attention, xi, vocab, sentences)
                    result['att'] = attentions
                    results.append(result)
                elif pj ==3:
                    tmp = tmp + 1
            # print tmp
            # print str(len(label_ids_dict))
            if tmp == len(label_ids_dict):
                result={}
                result['id'] = str(ids[i])
                result['proid'] = pros[i]
                result['dimid'] = str(-1)
                result['value'] = str(-1)
                result['att'] = None
                results.append(result)
        return results

    def predict_pro_attention_api(self, X, vocab, ids, pros, pro_label_ids_dict,map_id_dimension_code):
        # emptyY = [[0]*self.num_tasks for i in xrange(len(X))]
        XI =[]
        Y = []
        # pro_labels = [key for key,value in inv_label_ids_dict.iteritems()]
        
        for i,x in enumerate(X):
            xIds = []
            for xid in x:
                xIds.append(xid[0])
            XI.append(xIds)
            label_ids_dict = pro_label_ids_dict[pros[i]]
            # print label_ids_dict
            pro_labels = [key for key,value in label_ids_dict.iteritems()]
            y_label = []
            for label in self.task_name:
                if label in pro_labels:
                    y_label.append(0)
                else :
                    y_label.append(-1)
            Y.append(y_label)        

        # print Y

        x,xl,xw,y,yw = self.get_batch(XI,Y,len(X))

        
        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        alphas,predicts = self.session.run([self.alphas,self.predicts],feed)
        alphas = np.array(alphas)#num_tasks * length * batch_size 权重分布
        predicts = np.array(predicts)

        results = []
        for i,xi in enumerate(X):  
            predict = predicts[:,i]
            # sentences = self.get_sentence(xi, vocab)
            tmp = 0
            label_ids_dict = pro_label_ids_dict[pros[i]]
            for j,pj in enumerate(predict):
                if Y[i][j] == -1:
                    continue
                attention = alphas[j,:,i]
                result={}
                
                if pj < 3:
                    result['id'] = str(ids[i])
                    result['proid'] = pros[i]
                    # result['dimid'] = label_ids_dict[self.task_name[j]]
                    result['mapid'] = label_ids_dict[self.task_name[j]]
                    result['dimid'] = map_id_dimension_code[result['mapid']]
                    # print self.task_name[j]
                    # print label_ids_dict
                    if predict[j] == 0:
                        result['value'] =str(2)
                    elif predict[j] == 1:
                        result['value'] =str(1)
                    else:
                        result['value'] =str(0)

                    attentions = self.get_attentions_words(attention, xi, vocab)
                    result['att'] = attentions
                    results.append(result)
                elif pj ==3:
                    tmp = tmp + 1
            # print tmp
            # print str(len(label_ids_dict))
            if tmp == len(label_ids_dict):
                result={}
                result['id'] = str(ids[i])
                result['proid'] = pros[i]
                result['mapid'] = str(-1)
                result['dimid'] = str(-1)
                result['value'] = str(-1)
                result['att'] = None
                results.append(result)
        return results

    def predict_pro_attention_api_test(self, X, vocab):
        # emptyY = [[0]*self.num_tasks for i in xrange(len(X))]
        XI =[]
        Y = []
        # pro_labels = [key for key,value in inv_label_ids_dict.iteritems()]

        x,xl,xw,y,yw = self.get_batch(XI,Y,len(X))

        
        feed={self.inputs:x, self.input_lengths:xl,self.input_weights: xw ,self.label_weights:yw,self.sparse_labels:y}
        self.is_train = False
        alphas,predicts = self.session.run([self.alphas,self.predicts],feed)
        alphas = np.array(alphas)#num_tasks * length * batch_size 权重分布
        predicts = np.array(predicts)

        results = []
        for i,xi in enumerate(X):  
            predict = predicts[:,i]
            # sentences = self.get_sentence(xi, vocab)
            tmp = 0
            for j,pj in enumerate(predict):
                if Y[i][j] == -1:
                    continue
                attention = alphas[j,:,i]
                result={}
                
                if pj < 3:
                    if predict[j] == 0:
                        result['value'] =str(2)
                    elif predict[j] == 1:
                        result['value'] =str(1)
                    else:
                        result['value'] =str(0)

                    attentions = self.get_attentions_words(attention, xi, vocab)
                    result['att'] = attentions
                    results.append(result)
                elif pj ==3:
                    tmp = tmp + 1
            # print tmp
            # print str(len(label_ids_dict))
            if tmp == len(label_ids_dict):
                result={}
                result['value'] = str(-1)
                result['att'] = None
                results.append(result)
        return results

    def get_attentions_words(self, attention, xi, vocab):
        attensions = []
        for n,w in enumerate(xi):
            a={}
            a['a'] = "%.2f" % float(attention[n])
            if  attention[n]>3.0/len(xi):
                a['w'] = 3
            elif attention[n]>1.0/len(xi):
                a['w'] = 2
            elif attention[n]>0.5/len(xi):
                a['w'] =1
            else:
                continue
            a['v'] = vocab[w[0]]
            a['s'] = w[1]
            a['e'] = w[2]
            r = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),a['v'])
            if r != "":
                attensions.append(a)
            
            
        return attensions
    
    def get_attentions(self, attention, xi, vocab,sentences):
        attensions = []
        for n,w in enumerate(xi):
            a={}
            a['a'] = "%.2f" % float(attention[n])
            if  attention[n]>3.0/len(xi):
                a['w'] = 3
            elif attention[n]>1.0/len(xi):
                a['w'] = 2
            # elif attention[n]>0.5/len(xi):
            #     a['w'] =1
            else:
                continue
            a['v'] = vocab[w[0]]
            
            r = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),a['v'])
            if r != "":
                for s in sentences:
                    if (w[1]>= s['s'] and w[2]<=s['e']):
                        a['s'] =s['s']
                        a['e'] =s['e']
                        break
                attensions.append(a)
        return attensions

    def get_sentence(self, xi, vocab):
        sentences =[]
        start = 0
        end = 0
        for n,w in enumerate(xi):
            s={}
            r = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),vocab[w[0]])
            if r == "":
                end = w[1]
                s['s'] = start
                s['e'] = end
                sentences.append(s)
                start = w[2]
        if end < xi[len(xi)-1][1] :
            s = {}
            s['s'] = end
            s['e'] = xi[len(xi)-1][2]
            sentences.append(s)
        # print sentences
        return sentences

if __name__ == '__main__':

    model = Model()
