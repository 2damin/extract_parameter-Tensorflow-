import tensorflow as tf
import sys
import os
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

def dequote(s):
    if (s[0] == s[-1]) and s.startswitch(("'",'"')):
        return s[1:-1]
    return s


tf.reset_default_graph()

'ckpt 내부 변수 보기'
variables = tf.contrib.framework.list_variables('model.ckpt')   # Tensorflow Model Zoo 에서 제공하는 ckeckpoint file


orig_stdout = sys.stdout
f = open('variable_list', 'w')
sys.stdout = f

for i, v in enumerate(variables):
    print("{}. name : {} \n    shape : {}".format(i, v[0], v[1]))

sys.stdout = orig_stdout
f.close()

print("변수 불러오기 완료")

'ckpt 내부 로딩하기'

w0 = tf.Variable(tf.zeros([3,3,3,32]), name="FeatureExtractor/MobilenetV1/Conv2d_0/weights")
w0_beta = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/beta")
w0_gamma = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/gamma")
w0_mean = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/moving_mean")
w0_var = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_0/BatchNorm/moving_variance")

w1_d = tf.Variable(tf.zeros([3,3,32,1]), name="FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/depthwise_weights")
w1_p = tf.Variable(tf.zeros([1,1,32,64]), name="FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/weights")
w1_dbeta = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/beta")
w1_dgamma = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/gamma")
w1_pbeta = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/beta")
w1_pgamma = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/gamma")
w1_dmean = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_mean")
w1_dvar = tf.Variable(tf.zeros([32]), name="FeatureExtractor/MobilenetV1/Conv2d_1_depthwise/BatchNorm/moving_variance")
w1_pmean = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_mean")
w1_pvar = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_1_pointwise/BatchNorm/moving_variance")

w2_d = tf.Variable(tf.zeros([3,3,64,1]), name="FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/depthwise_weights")
w2_p = tf.Variable(tf.zeros([1,1,64,128]), name="FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/weights")
w2_dbeta = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/beta")
w2_dgamma = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/gamma")
w2_pbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/beta")
w2_pgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/gamma")
w2_dmean = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_mean")
w2_dvar = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_2_depthwise/BatchNorm/moving_variance")
w2_pmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_mean")
w2_pvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_2_pointwise/BatchNorm/moving_variance")

w3_d = tf.Variable(tf.zeros([3,3,128,1]), name="FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/depthwise_weights")
w3_p = tf.Variable(tf.zeros([1,1,128,128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/weights")
w3_dbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/beta")
w3_dgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/gamma")
w3_pbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/beta")
w3_pgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/gamma")
w3_dmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_mean")
w3_dvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_depthwise/BatchNorm/moving_variance")
w3_pmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_mean")
w3_pvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_3_pointwise/BatchNorm/moving_variance")

w4_d = tf.Variable(tf.zeros([3,3,128,1]), name="FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/depthwise_weights")
w4_p = tf.Variable(tf.zeros([1,1,128,256]), name="FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/weights")
w4_dbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/beta")
w4_dgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/gamma")
w4_pbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/beta")
w4_pgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/gamma")
w4_dmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_mean")
w4_dvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_4_depthwise/BatchNorm/moving_variance")
w4_pmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_mean")
w4_pvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_4_pointwise/BatchNorm/moving_variance")

w5_d = tf.Variable(tf.zeros([3,3,256,1]), name="FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/depthwise_weights")
w5_p = tf.Variable(tf.zeros([1,1,256,256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/weights")
w5_dbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/beta")
w5_dgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/gamma")
w5_pbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/beta")
w5_pgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/gamma")
w5_dmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_mean")
w5_dvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_depthwise/BatchNorm/moving_variance")
w5_pmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean")
w5_pvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_variance")

w6_d = tf.Variable(tf.zeros([3,3,256,1]), name="FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/depthwise_weights")
w6_p = tf.Variable(tf.zeros([1,1,256,512]), name="FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/weights")
w6_dbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/beta")
w6_dgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/gamma")
w6_pbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/beta")
w6_pgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/gamma")
w6_dmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_mean")
w6_dvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_6_depthwise/BatchNorm/moving_variance")
w6_pmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_mean")
w6_pvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_6_pointwise/BatchNorm/moving_variance")

w7_d = tf.Variable(tf.zeros([3,3,512,1]), name="FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/depthwise_weights")
w7_p = tf.Variable(tf.zeros([1,1,512,512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/weights")
w7_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/beta")
w7_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/gamma")
w7_pbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/beta")
w7_pgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/gamma")
w7_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_mean")
w7_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_depthwise/BatchNorm/moving_variance")
w7_pmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_mean")
w7_pvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_7_pointwise/BatchNorm/moving_variance")

w8_d = tf.Variable(tf.zeros([3,3,512,1]), name="FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/depthwise_weights")
w8_p = tf.Variable(tf.zeros([1,1,512,512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/weights")
w8_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/beta")
w8_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/gamma")
w8_pbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/beta")
w8_pgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/gamma")
w8_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_mean")
w8_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_depthwise/BatchNorm/moving_variance")
w8_pmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_mean")
w8_pvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_8_pointwise/BatchNorm/moving_variance")

w9_d = tf.Variable(tf.zeros([3,3,512,1]), name="FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/depthwise_weights")
w9_p = tf.Variable(tf.zeros([1,1,512,512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/weights")
w9_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/beta")
w9_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/gamma")
w9_pbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta")
w9_pgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/gamma")
w9_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_mean")
w9_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_depthwise/BatchNorm/moving_variance")
w9_pmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_mean")
w9_pvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_9_pointwise/BatchNorm/moving_variance")

w10_d = tf.Variable(tf.zeros([3,3,512,1]), name="FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/depthwise_weights")
w10_p = tf.Variable(tf.zeros([1,1,512,512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/weights")
w10_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/beta")
w10_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/gamma")
w10_pbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/beta")
w10_pgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/gamma")
w10_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_mean")
w10_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_depthwise/BatchNorm/moving_variance")
w10_pmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_mean")
w10_pvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_10_pointwise/BatchNorm/moving_variance")

w11_d = tf.Variable(tf.zeros([3,3,512,1]), name="FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/depthwise_weights")
w11_p = tf.Variable(tf.zeros([1,1,512,512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/weights")
w11_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/beta")
w11_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/gamma")
w11_pbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/beta")
w11_pgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/gamma")
w11_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_mean")
w11_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_depthwise/BatchNorm/moving_variance")
w11_pmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_mean")
w11_pvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_11_pointwise/BatchNorm/moving_variance")

w12_d = tf.Variable(tf.zeros([3,3,512,1]), name="FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/depthwise_weights")
w12_p = tf.Variable(tf.zeros([1,1,512,1024]), name="FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/weights")
w12_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta")
w12_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma")
w12_pbeta = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta")
w12_pgamma = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma")
w12_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_mean")
w12_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_12_depthwise/BatchNorm/moving_variance")
w12_pmean = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_mean")
w12_pvar = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_12_pointwise/BatchNorm/moving_variance")

w13_d = tf.Variable(tf.zeros([3,3,1024,1]), name="FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/depthwise_weights")
w13_p = tf.Variable(tf.zeros([1,1,1024,1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/weights")
w13_dbeta = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta")
w13_dgamma = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma")
w13_pbeta = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta")
w13_pgamma = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma")
w13_dmean = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_mean")
w13_dvar = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_depthwise/BatchNorm/moving_variance")
w13_pmean = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_mean")
w13_pvar = tf.Variable(tf.zeros([1024]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise/BatchNorm/moving_variance")

w14_p = tf.Variable(tf.zeros([1,1,1024,256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/weights")
w14_pbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/beta")
w14_pgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/gamma")
w14_pmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/moving_mean")
w14_pvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/moving_variance")

w14_2_d = tf.Variable(tf.zeros([3,3,256,512]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/weights")
w14_2_dbeta = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/beta")
w14_2_dgamma = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/gamma")
w14_2_dmean = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/moving_mean")
w14_2_dvar = tf.Variable(tf.zeros([512]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/moving_variance")

w15_p = tf.Variable(tf.zeros([1,1,512,128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/weights")
w15_pbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/beta")
w15_pgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/gamma")
w15_pmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/moving_mean")
w15_pvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/moving_variance")

w15_2_d = tf.Variable(tf.zeros([3,3,128,256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/weights")
w15_2_dbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/beta")
w15_2_dgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/gamma")
w15_2_dmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/moving_mean")
w15_2_dvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/moving_variance")

w16_p = tf.Variable(tf.zeros([1,1,256,128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/weights")
w16_pbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/beta")
w16_pgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/gamma")
w16_pmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/moving_mean")
w16_pvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/moving_variance")

w16_2_d = tf.Variable(tf.zeros([3,3,128,256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/weights")
w16_2_dbeta = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/beta")
w16_2_dgamma = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/gamma")
w16_2_dmean = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/moving_mean")
w16_2_dvar = tf.Variable(tf.zeros([256]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/moving_variance")

w17_p = tf.Variable(tf.zeros([1,1,256,64]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/weights")
w17_pbeta = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/beta")
w17_pgamma = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/gamma")
w17_pmean = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/moving_mean")
w17_pvar = tf.Variable(tf.zeros([64]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_5_1x1_64/BatchNorm/moving_variance")

w17_2_d = tf.Variable(tf.zeros([3,3,64,128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/weights")
w17_2_dbeta = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/beta")
w17_2_dgamma = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/gamma")
w17_2_dmean = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/moving_mean")
w17_2_dvar = tf.Variable(tf.zeros([128]), name="FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/BatchNorm/moving_variance")

box0_boxP_w = tf.Variable(tf.zeros([1,1,512,12]), name="BoxPredictor_0/BoxEncodingPredictor/weights")
box0_boxP_b = tf.Variable(tf.zeros([12]), name="BoxPredictor_0/BoxEncodingPredictor/biases")
box0_classP_w = tf.Variable(tf.zeros([1,1,512,108]), name="BoxPredictor_0/ClassPredictor/weights") #(class+1) * 3
box0_classP_b = tf.Variable(tf.zeros([108]), name="BoxPredictor_0/ClassPredictor/biases")#(class+1) * 3

box1_boxP_w = tf.Variable(tf.zeros([1,1,1024,24]), name="BoxPredictor_1/BoxEncodingPredictor/weights")
box1_boxP_b = tf.Variable(tf.zeros([24]), name="BoxPredictor_1/BoxEncodingPredictor/biases")
box1_classP_w = tf.Variable(tf.zeros([1,1,1024,216]), name="BoxPredictor_1/ClassPredictor/weights")#(class+1) * 6
box1_classP_b = tf.Variable(tf.zeros([216]), name="BoxPredictor_1/ClassPredictor/biases")#(class+1) * 6

box2_boxP_w = tf.Variable(tf.zeros([1,1,512,24]), name="BoxPredictor_2/BoxEncodingPredictor/weights")
box2_boxP_b = tf.Variable(tf.zeros([24]), name="BoxPredictor_2/BoxEncodingPredictor/biases")
box2_classP_w = tf.Variable(tf.zeros([1,1,512,216]), name="BoxPredictor_2/ClassPredictor/weights")#(class+1) * 6
box2_classP_b = tf.Variable(tf.zeros([216]), name="BoxPredictor_2/ClassPredictor/biases")#(class+1) * 6

box3_boxP_w = tf.Variable(tf.zeros([1,1,256,24]), name="BoxPredictor_3/BoxEncodingPredictor/weights")
box3_boxP_b = tf.Variable(tf.zeros([24]), name="BoxPredictor_3/BoxEncodingPredictor/biases")
box3_classP_w = tf.Variable(tf.zeros([1,1,256,216]), name="BoxPredictor_3/ClassPredictor/weights")
box3_classP_b = tf.Variable(tf.zeros([216]), name="BoxPredictor_3/ClassPredictor/biases")

box4_boxP_w = tf.Variable(tf.zeros([1,1,256,24]), name="BoxPredictor_4/BoxEncodingPredictor/weights")
box4_boxP_b = tf.Variable(tf.zeros([24]), name="BoxPredictor_4/BoxEncodingPredictor/biases")
box4_classP_w = tf.Variable(tf.zeros([1,1,256,216]), name="BoxPredictor_4/ClassPredictor/weights")#(class+1) * 6
box4_classP_b = tf.Variable(tf.zeros([216]), name="BoxPredictor_4/ClassPredictor/biases")#(class+1) * 6

box5_boxP_w = tf.Variable(tf.zeros([1,1,128,24]), name="BoxPredictor_5/BoxEncodingPredictor/weights")
box5_boxP_b = tf.Variable(tf.zeros([24]), name="BoxPredictor_5/BoxEncodingPredictor/biases")
box5_classP_w = tf.Variable(tf.zeros([1,1,128,216]), name="BoxPredictor_5/ClassPredictor/weights")#(class+1) * 6
box5_classP_b = tf.Variable(tf.zeros([216]), name="BoxPredictor_5/ClassPredictor/biases")#(class+1) * 6


print("변수 생성 완료")


def savetxt(conv_name,w_d, w_p, w_dbeta, w_dgamma, w_pbeta, w_pgamma, w_dmean, w_dvar, w_pmean, w_pvar):
    np.savetxt("datData/"+conv_name+"_d.dat", sess.run(w_d).flatten())
    np.savetxt("datData/"+conv_name+"_p.dat", sess.run(w_p).flatten())
    np.savetxt("datData/"+conv_name+"_dbeta.dat", sess.run(w_dbeta).flatten())
    np.savetxt("datData/"+conv_name+"_dgamma.dat", sess.run(w_dgamma).flatten())
    np.savetxt("datData/"+conv_name+"_pbeta.dat", sess.run(w_pbeta).flatten())
    np.savetxt("datData/"+conv_name+"_pgamma.dat", sess.run(w_pgamma).flatten())
    np.savetxt("datData/"+conv_name+"_dMean.dat", sess.run(w_dmean).flatten())
    np.savetxt("datData/"+conv_name+"_dVar.dat", sess.run(w_dvar).flatten())
    np.savetxt("datData/"+conv_name+"_pMean.dat", sess.run(w_pmean).flatten())
    np.savetxt("datData/"+conv_name+"_pVar.dat", sess.run(w_pvar).flatten())

def savetxt2(conv_name,conv_name2,w_d, w_p, w_dbeta, w_dgamma, w_pbeta, w_pgamma, w_dmean, w_dvar, w_pmean, w_pvar):
    np.savetxt("datData/"+conv_name2+"_d.dat", sess.run(w_d).flatten())
    np.savetxt("datData/"+conv_name+"_p.dat", sess.run(w_p).flatten())
    np.savetxt("datData/"+conv_name2+"_dbeta.dat", sess.run(w_dbeta).flatten())
    np.savetxt("datData/"+conv_name2+"_dgamma.dat", sess.run(w_dgamma).flatten())
    np.savetxt("datData/"+conv_name+"_pbeta.dat", sess.run(w_pbeta).flatten())
    np.savetxt("datData/"+conv_name+"_pgamma.dat", sess.run(w_pgamma).flatten())
    np.savetxt("datData/"+conv_name2+"_dMean.dat", sess.run(w_dmean).flatten())
    np.savetxt("datData/"+conv_name2+"_dVar.dat", sess.run(w_dvar).flatten())
    np.savetxt("datData/"+conv_name+"_pMean.dat", sess.run(w_pmean).flatten())
    np.savetxt("datData/"+conv_name+"_pVar.dat", sess.run(w_pvar).flatten())

def savetxt_box(boxName,boxp_w, boxp_b, classp_w, classp_b):
    np.savetxt("datData/"+boxName+"_boxp_w.dat", sess.run(boxp_w).flatten())
    np.savetxt("datData/"+boxName+"_boxp_b.dat", sess.run(boxp_b).flatten())
    np.savetxt("datData/"+boxName+"_classp_w.dat", sess.run(classp_w).flatten())
    np.savetxt("datData/"+boxName+"_classp_b.dat", sess.run(classp_b).flatten())


init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init_op)
saver.restore(sess, "model.ckpt")


#use numpy.savetxt and save .dat file
np.savetxt("datData/conv0_d.dat", sess.run(w0).flatten())
np.savetxt("datData/conv0_dbeta.dat", sess.run(w0_beta).flatten())
np.savetxt("datData/conv0_dgamma.dat", sess.run(w0_gamma).flatten())
np.savetxt("datData/conv0_dMean.dat", sess.run(w0_mean).flatten())
np.savetxt("datData/conv0_dVar.dat", sess.run(w0_var).flatten())

savetxt("conv1", w1_d, w1_p, w1_dbeta, w1_dgamma, w1_pbeta, w1_pgamma, w1_dmean, w1_dvar, w1_pmean, w1_pvar)
savetxt("conv2", w2_d, w2_p, w2_dbeta, w2_dgamma, w2_pbeta, w2_pgamma, w2_dmean, w2_dvar, w2_pmean, w2_pvar)
savetxt("conv3", w3_d, w3_p, w3_dbeta, w3_dgamma, w3_pbeta, w3_pgamma, w3_dmean, w3_dvar, w3_pmean, w3_pvar)
savetxt("conv4", w4_d, w4_p, w4_dbeta, w4_dgamma, w4_pbeta, w4_pgamma, w4_dmean, w4_dvar, w4_pmean, w4_pvar)
savetxt("conv5", w5_d, w5_p, w5_dbeta, w5_dgamma, w5_pbeta, w5_pgamma, w5_dmean, w5_dvar, w5_pmean, w5_pvar)
savetxt("conv6", w6_d, w6_p, w6_dbeta, w6_dgamma, w6_pbeta, w6_pgamma, w6_dmean, w6_dvar, w6_pmean, w6_pvar)
savetxt("conv7", w7_d, w7_p, w7_dbeta, w7_dgamma, w7_pbeta, w7_pgamma, w7_dmean, w7_dvar, w7_pmean, w7_pvar)
savetxt("conv8", w8_d, w8_p, w8_dbeta, w8_dgamma, w8_pbeta, w8_pgamma, w8_dmean, w8_dvar, w8_pmean, w8_pvar)
savetxt("conv9", w9_d, w9_p, w9_dbeta, w9_dgamma, w9_pbeta, w9_pgamma, w9_dmean, w9_dvar, w9_pmean, w9_pvar)
savetxt("conv10", w10_d, w10_p, w10_dbeta, w10_dgamma, w10_pbeta, w10_pgamma, w10_dmean, w10_dvar, w10_pmean, w10_pvar)
savetxt("conv11", w11_d, w11_p, w11_dbeta, w11_dgamma, w11_pbeta, w11_pgamma, w11_dmean, w11_dvar, w11_pmean, w11_pvar)
savetxt("conv12", w12_d, w12_p, w12_dbeta, w12_dgamma, w12_pbeta, w12_pgamma, w12_dmean, w12_dvar, w12_pmean, w12_pvar)
savetxt("conv13", w13_d, w13_p, w13_dbeta, w13_dgamma, w13_pbeta, w13_pgamma, w13_dmean, w13_dvar, w13_pmean, w13_pvar)
savetxt2("conv14", "conv14_2",w14_2_d, w14_p, w14_2_dbeta, w14_2_dgamma, w14_pbeta, w14_pgamma, w14_2_dmean, w14_2_dvar, w14_pmean, w14_pvar)
savetxt2("conv15", "conv15_2",w15_2_d, w15_p, w15_2_dbeta, w15_2_dgamma, w15_pbeta, w15_pgamma, w15_2_dmean, w15_2_dvar, w15_pmean, w15_pvar)
savetxt2("conv16", "conv16_2",w16_2_d, w16_p, w16_2_dbeta, w16_2_dgamma, w16_pbeta, w16_pgamma, w16_2_dmean, w16_2_dvar, w16_pmean, w16_pvar)
savetxt2("conv17", "conv17_2",w17_2_d, w17_p, w17_2_dbeta, w17_2_dgamma, w17_pbeta, w17_pgamma, w17_2_dmean, w17_2_dvar, w17_pmean, w17_pvar)

savetxt_box("box0",box0_boxP_w, box0_boxP_b, box0_classP_w, box0_classP_b)
savetxt_box("box1",box1_boxP_w, box1_boxP_b, box1_classP_w, box1_classP_b)
savetxt_box("box2",box2_boxP_w, box2_boxP_b, box2_classP_w, box2_classP_b)
savetxt_box("box3",box3_boxP_w, box3_boxP_b, box3_classP_w, box3_classP_b)
savetxt_box("box4",box4_boxP_w, box4_boxP_b, box4_classP_w, box4_classP_b)
savetxt_box("box5",box5_boxP_w, box5_boxP_b, box5_classP_w, box5_classP_b)

print("작업끝")


"""
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model.ckpt-21067.meta')
    new_saver.restore(sess, 'model.ckpt-21067')
    graph = tf.get_default_graph()
    graph_key_list = graph.get_all_collection_keys()
    graph_list = graph.get_collection(graph_key_list[0])
    print(graph_key_list[0])
"""


