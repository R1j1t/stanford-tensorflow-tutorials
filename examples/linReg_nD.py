from sklearn import datasets
import tensorflow as tf
import numpy as np
import time

# Step 1: read in the data
boston = datasets.load_boston()

n_samples = boston.target.shape[0]
print (n_samples)

# Step 2: create Dataset and iterator
dataset = tf.data.Dataset.from_tensor_slices((boston.data,boston.target))
dataset = dataset.shuffle(1000)

iterator = dataset.make_initializable_iterator()
X,Y = iterator.get_next()
X = X[tf.newaxis]

# Step 3: Adding bias to X
#X = np.hstack((tf.ones((n_samples,1),name='bias_addition'),X))

#Step 4: Defining biases
w = tf.get_variable('weights', initializer=tf.zeros((13,1),dtype=tf.float64,name='ones_init'))
b = tf.get_variable('bias', initializer=tf.zeros((1,1),dtype=tf.float64,name='ones_init'))

## Defining Y_pred
y_pred = tf.matmul(X,w,transpose_b=False) + b

## loss
loss = tf.square(tf.norm(Y-y_pred),name='loss')

## defining optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000012).minimize(loss)
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graph/lin_reg_nD',sess.graph)
    
        
    ##Applying gradient descent
    for i in range(100):
        sess.run(iterator.initializer)
        total_loss=0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass

        #print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()
    end_time = time.time()
    w_out,b_out = sess.run([w,b])
    np.set_printoptions(precision=3)
    print('w:')
    print(w_out)
    print ('\n b: %f' %(b_out))   

print ('Total time taken is {0}'.format(end_time-start_time))
