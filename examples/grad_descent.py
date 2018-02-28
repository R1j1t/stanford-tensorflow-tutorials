from sklearn import datasets
import tensorflow as tf
import matplotlib.pyplot as plt
import utils

boston = datasets.load_boston()
##DATA_FILE = 'data/birth_life_2010.txt'
##data, n_samples = utils.read_birth_life_data(DATA_FILE)
##
##n_samples = boston.target.shape[0]
##print (n_samples)

DATA_FILE = 'data/birth_life_2010.txt'

# Step 1: read in the data
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

### Step 2: create Dataset and iterator
##dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
##dataset = tf.data.Dataset.from_tensor_slices((boston.data,boston.target))

iterator = dataset.make_initializable_iterator()
X,Y = iterator.get_next()

w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

## Defining Y_pred
y_pred = X*w + b


## loss
loss = tf.square(Y-y_pred,name='loss')

## defining optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graph/lin_reg_own',sess.graph)
    
        
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

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()
    w_out, b_out = sess.run([w, b]) 
    print('w: %f, b: %f' %(w_out, b_out))   

# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
# plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
plt.legend()
plt.show()

