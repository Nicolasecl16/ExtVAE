import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


#Model for radius generation with known prior
class Encoder(tfk.Model):
    
    def __init__(self):      
        super(Encoder,self).__init__()      
        #self.prior        = tfd.Gamma(concentration=1.5,rate = 1)
        self.prior        = tfd.Gamma(concentration=4.5,rate = 1)
        self.dense1       = tfkl.Dense(5, activation ='relu',kernel_initializer =tfk.initializers.Zeros())
        self.dense2       = tfkl.Dense(5, activation ='relu',kernel_initializer =tfk.initializers.Zeros())
        self.dense3       = tfkl.Dense(2, activation='relu',bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2))
        self.lambda1      = tfkl.Lambda(lambda x: tf.abs(x)+0.001, name='posterior_params')
        self.dist_lambda1 = tfpl.DistributionLambda(
                            make_distribution_fn=lambda t: tfd.Gamma(
                                concentration=t[...,0], rate = t[...,1]),
                                    activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior,use_exact_kl =True))  
        
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.lambda1(x)
        x = self.dist_lambda1(x)
        return x
    
class Decoder(tfk.Model):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dense1       = tfkl.Dense(5, use_bias=True, activation='relu')
        self.lambda1      = tfkl.Lambda(lambda x: 1/x)
        self.dense2       = tfkl.Dense(5, use_bias=True, activation='relu')
        self.dense31      = tfkl.Dense(1, use_bias=True,bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2))
        self.dense32      = tfkl.Dense(1, use_bias=True,bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2), kernel_initializer = tfk.initializers.RandomUniform(minval=0.1, maxval=2))
        self.dense3       = tfkl.Dense(2, activation='relu',bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2))
        self.lambda2      = tfkl.Lambda(lambda x: tf.abs(x)+0.00001)
        self.concat1      = tfkl.Concatenate()
        self.dist_lambda1 = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Gamma(
                concentration=t[...,0], rate=t[...,1]))
        self.dist_n       = tfpl.IndependentNormal(1)
        self.ind_norm1 = tfpl.DistributionLambda(
                            make_distribution_fn=lambda v: tfd.TruncatedNormal(
                                loc=v[...,0], scale=v[...,1],low = 0,high = 1000))
        
        
    def call(self, inputs):
        y     = tfkl.Reshape(target_shape=[1])(inputs)
        y1    = self.lambda1(y)
        x     = self.dense1(y1)
        x     = self.dense2(x)
        
        #for gamma output :       
        alpha = self.dense31(x*y)
        alpha = self.lambda1(alpha)
        beta  = self.dense32(x)
        beta  = self.lambda2(beta*y**2)
        x     = self.concat1([alpha,beta])                
        x     = self.dist_lambda1(x)
        
        #normal output or truncated normal:
        #mu    = self.dense31(x)
        #sig   = self.dense32(x)
        #sig   = self.lambda1(sig/(1+y1))
        #x     = self.concat1([mu,sig])
        #x     = self.dist_n(x)
        #x     = self.ind_norm1(x)
        
        return x
    
class Ext_VAE(tfk.Model):
    def __init__(self):      
        super(Ext_VAE,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self,inputs):
        return self.decoder(self.encoder(inputs))

# Model for radius generation with known prior and multimodal distribution :
class Multi_Encoder(tfk.Model):
    
    def __init__(self):      
        super(Multi_Encoder,self).__init__()      
        self.prior        = tfd.Gamma(concentration=1.8,rate = 1)
        self.dense1       = tfkl.Dense(5, activation ='relu',kernel_initializer =tfk.initializers.Zeros())
        self.dense2       = tfkl.Dense(5, activation ='relu',kernel_initializer =tfk.initializers.Zeros())
        self.dense3       = tfkl.Dense(2, activation='relu',bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2))
        self.lambda1      = tfkl.Lambda(lambda x: tf.abs(x)+0.001, name='posterior_params')
        self.dist_lambda1 = tfpl.DistributionLambda(
                            make_distribution_fn=lambda t: tfd.Gamma(
                                concentration=t[...,0], rate = t[...,1]),
                                    activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior,use_exact_kl =True))  
        
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.lambda1(x)
        x = self.dist_lambda1(x)
        return x
    
class Multi_Decoder(tfk.Model):
    def __init__(self):
        super(Multi_Decoder,self).__init__()
        self.num_components = 2
        self.ev_shape       = [1]
        self.params_size    = tfpl.MixtureSameFamily.params_size(self.num_components,
                                                 component_params_size=tfpl.IndependentNormal.params_size(self.ev_shape))
        self.lambda1        = tfkl.Lambda(lambda x: 1/x)
        self.dense1         = tfkl.Dense(8, use_bias=True, activation='relu')
        self.dense2         = tfkl.Dense(8, use_bias=True, activation='relu')
        self.dense3         = tfkl.Dense(self.params_size,activation = 'relu')
        self.dense41        = tfkl.Dense(2, 
                                         bias_initializer = tfk.initializers.RandomUniform(minval=-0.1, maxval=0.1))
        self.dense421       = tfkl.Dense(1, 
                                         bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2))
        self.dense422       = tfkl.Dense(1, 
                                         bias_initializer = tfk.initializers.RandomUniform(minval=0.5, maxval=2))
        self.dense431       = tfkl.Dense(1, 
                                         bias_initializer = tfk.initializers.RandomUniform(minval=5, maxval=10))
        self.dense432       = tfkl.Dense(1, 
                                         bias_initializer = tfk.initializers.RandomUniform(minval=0.5, maxval=2))
        self.lambda2        = tfkl.Lambda(lambda x: tf.abs(x)+0.001)
        self.concat1        = tfkl.Concatenate()
        self.mixt_distn1    = tfpl.MixtureSameFamily(self.num_components, tfpl.IndependentNormal(self.ev_shape))
        self.mixt_distg1    = tfpl.MixtureSameFamily(self.num_components, 
                                                     tfpl.DistributionLambda(
                                                         make_distribution_fn=lambda t: tfd.Gamma(
                                                             concentration=t[...,0], rate=t[...,1])))
        self.dense31        = tfkl.Dense(3, use_bias=True)
        self.dense32        = tfkl.Dense(3, activation='relu',
                                         bias_initializer = tfk.initializers.RandomUniform(minval=0.5, maxval=10))
        self.dense33        = tfkl.Dense(3, activation='relu',
                                         bias_initializer = tfk.initializers.RandomUniform(minval=1, maxval=2))
        #self.lambdan        = tfkl.Lambda(lambda x: tf.abs(x)+0.001)

        
    def call(self, inputs):
        y     = tfkl.Reshape(target_shape=[1])(inputs)
        y1    = self.lambda1(y)
        x     = self.dense1(y1)
        x     = self.dense2(x)
        
        # normal mixture
        x     = self.dense3(x)
        m     = self.dense41(x)
        #mu1   = self.dense421(x)
        #sig1  = self.dense431(x)
        #mu2   = self.dense422(x)
        #sig2  = self.dense432(x)
        #x     = self.concat1([m,mu1,sig1,mu2,sig2])
        #x     = self.lambda2(x)
        #x     = self.lambda2(x)
        #x     = self.mixt_distn1(x)
        
        # Gamma mixture 
        #m     = self.dense1(x)
        alpha1 = self.dense421(x*y)
        beta1  = self.dense431(x*y**2)
        alpha2 = self.dense422(x*y)
        beta2  = self.dense432(x*y**2)
        #beta  = self.lambda2(beta*inputs**2)
        x     = self.concat1([m,alpha1,alpha2,beta1,beta2])
        x     = self.mixt_distg1(x)
        return x
    
class Ext_Multi_VAE(tfk.Model):
    def __init__(self):      
        super(Ext_Multi_VAE,self).__init__()
        self.encoder = Multi_Encoder()
        self.decoder = Multi_Decoder()
    
    def call(self,inputs):
        return self.decoder(self.encoder(inputs))
    
    
    
    
# Model for radius generation with learned tail index of the prior
class U_Encoder(tfk.Model):    
    def __init__(self):      
        super(U_Encoder,self).__init__()
        self.alpha        = 1.5
        self.prior        = self.make_mvn_prior(1,True)
        self.dense1       = tfkl.Dense(5, activation ='relu',kernel_initializer =tfk.initializers.RandomUniform(minval=0.01, maxval=0.02))
        self.dense2       = tfkl.Dense(5, activation ='relu',kernel_initializer =tfk.initializers.RandomUniform(minval=0.01, maxval=0.02))
        self.dense3       = tfkl.Dense(2, activation='relu',bias_initializer = tfk.initializers.RandomUniform(minval=3., maxval=5.5))
        self.lambda1      = tfkl.Lambda(lambda x: tf.abs(x)+0.001)
        self.dist_lambda3 = tfpl.DistributionLambda(
                            make_distribution_fn=lambda t: (tfd.Gamma(
                                concentration=t[...,0], rate=t[...,1])),
            activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, use_exact_kl=True)
        )  
        
        
    def make_mvn_prior(self,ndim,Trainable=True):
        if Trainable:
            c = tf.Variable(tf.random.uniform([ndim], minval=3.5,maxval = 5.5, dtype=tf.float32), name='prior_c')
            print(c)
            rate = 1
        else:
            c = self.alpha
            rate = 1
        prior = (tfd.Gamma(concentration=c, rate=rate))
        return prior
    
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dist_lambda3(x)
        return(x)
    
 
 
class U_Decoder(tfk.Model):
    def __init__(self):
        super(U_Decoder,self).__init__()
        
        self.dense1  = tfkl.Dense(5, use_bias=True, activation='relu')
        self.lambda1 = tfkl.Lambda(lambda x: 1/x)
        self.dense2  = tfkl.Dense(5, use_bias=True, activation='relu')
        self.dense31 = tfkl.Dense(1, use_bias=True)
        self.dense32 = tfkl.Dense(1, use_bias=True)
        self.lambda2 = tfkl.Lambda(lambda x: tf.abs(x)+0.001)
        self.concat1 = tfkl.Concatenate()
        self.dist_lambda1 = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Gamma(
                concentration=t[...,0], rate=t[...,1]))
        
        
    def call(self, inputs):
        y1     = tfkl.Reshape(target_shape=[1])(inputs)
        y     = self.lambda1(y1)
        x     = self.dense1(y)
        x     = self.dense2(x)
        alpha = self.dense31(x)
        alpha = self.lambda2(alpha*y1)
        beta  = self.dense32(x)
        beta  = self.lambda2(beta/y**2)
        x     = self.concat1([alpha,beta])
        x     = self.dist_lambda1(x)
        return x

    
class U_Ext_VAE(tfk.Model):
    def __init__(self):      
        super(U_Ext_VAE,self).__init__()
        self.encoder = U_Encoder()
        self.decoder = U_Decoder()
        #self.Block   = tfpl.DistributionLambda(
         #   make_distribution_fn=lambda d: tfd.Blockwise(
        #        d))
    
    def call(self,inputs):
        res =  self.decoder(self.encoder(inputs))
        return res
    


    
# Standard VAE   
class Std_Encoder(tfk.Model):
    
    def __init__(self):      
        super(Std_Encoder,self).__init__()
        self.encoded_size =4
        self.prior        = tfd.MultivariateNormalDiag(loc=tf.zeros(self.encoded_size))
        self.dense1       = tfkl.Dense(5,activation='relu')
        self.dense2       = tfkl.Dense(5,activation='relu')
        self.lambda1      = tfkl.Lambda(lambda x: tf.abs(x)+0.001)
        self.dense3       = tfkl.Dense(tfpl.IndependentNormal.params_size(self.encoded_size))
        self.ind_norm1    = tfpl.IndependentNormal(self.encoded_size,
                                                   activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior,
                                                                                                     weight=1.0))
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.lambda1(x)
        x = self.dense3(x)
        x = self.ind_norm1(x)
        return x
    
class Std_Decoder(tfk.Model):
    def __init__(self):
        super(Std_Decoder,self).__init__()
        self.K         = 4
        self.dense1    = tfkl.Dense(5, use_bias=True, activation='relu')
        self.dense2    = tfkl.Dense(5, use_bias=True, activation='relu')
        self.dense3    = tfkl.Dense(tfpl.IndependentNormal(self.K).params_size(self.K))        
        self.lambda1   = tfkl.Lambda(lambda x: tf.abs(x)+0.001)
        self.ind_norm1 = tfpl.DistributionLambda(
                            make_distribution_fn=lambda v: tfd.TruncatedNormal(
                                loc=v[...,0], scale=v[...,1],low = 0,high = 1000))
        
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.lambda1(x)
        x = self.ind_norm1(x)
        return x
    
class Std_VAE(tfk.Model):
    def __init__(self):      
        super(Std_VAE,self).__init__()
        self.encoder = Std_Encoder()
        self.decoder = Std_Decoder()
    
    def call(self,inputs):
        return self.decoder(self.encoder(inputs))
    
    
    
    
#VAE for radius-conditioned angular distribution
class Sphere_Encoder(tfk.Model):
    
    def __init__(self):      
        super(Sphere_Encoder,self).__init__()
        self.encoded_size = 4
        self.prior        =  tfd.Independent(tfd.Normal(loc=tf.zeros(self.encoded_size), scale=1),
                                reinterpreted_batch_ndims=1)
        self.concat       = tfkl.Concatenate()
        self.dense1       = tfkl.Dense(8,activation='relu')
        self.dense2       = tfkl.Dense(8,activation='relu')
        self.lambda1      = tfkl.Lambda(lambda x: tf.abs(x)+0.001)
        self.dense3       = tfkl.Dense(tfpl.IndependentNormal.params_size(self.encoded_size))
        self.ind_norm1    = tfpl.IndependentNormal(self.encoded_size,
                                                   activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior,
                                                                                                     weight=1.0))
        
    def call(self, inputs):
        inputs1,inputs2 = inputs
        x = self.concat([inputs1,inputs2/10])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.lambda1(x)
        x = self.dense3(x)
        x = self.ind_norm1(x)
        return x
    

class Sphere_Decoder(tfk.Model):
    def __init__(self):
        super(Sphere_Decoder,self).__init__()
        self.K          = 5
        self.dense1     = tfkl.Dense(5, use_bias=True, activation='relu')
        self.dense2     = tfkl.Dense(10, use_bias=True, activation='relu')
        self.dense31    = tfkl.Dense(tfpl.IndependentNormal(self.K).params_size(self.K))        
        self.ind_norm11 = tfpl.IndependentNormal(self.K)
        self.dense32    = tfkl.Dense(5,activation = 'relu', bias_initializer =
                                     tfk.initializers.RandomUniform(minval=2, maxval=3))
        #self.lambda12   = tfkl.Lambda(lambda x: tf.abs(x),activation='softmax')
        self.concat     = tfkl.Concatenate()
        self.lambda1    = tfkl.Lambda(lambda x : 1/(1+x))
        
        self.diri12 = tfpl.DistributionLambda(
                            make_distribution_fn=lambda c: tfd.Dirichlet(
                                c))
        
    def call(self, inputs):
        inputs1,inputs2 = inputs
        inputs2         = self.lambda1(inputs2)
        x               = self.dense1(inputs1)
        x               = self.concat([x,inputs2])
        x               = self.dense2(x)
        #normal without forcing mu to be on the sphere
        x               = self.dense31(x)
        x               = self.ind_norm11(x)
        #dirichlet output 
        #x               = self.dense32(x)
        #x = self.lambda12(x)
        #x = self.diri12(x)
        return x
    
class Sphere_VAE(tfk.Model):
    def __init__(self):      
        super(Sphere_VAE,self).__init__()
        self.encoder = Sphere_Encoder()
        self.decoder = Sphere_Decoder()
    
    def call(self,inputs):
        inputs1,inputs2 = inputs
        return self.decoder([self.encoder([inputs1,inputs2]),inputs2])
    
