# Machine-learning
Linear Regression/机器学习案例分析/线性回归  
step1: 给定数据集，需要先进行抽取特征：  
                                  exam_X = examDf.loc[:,'xxx']  
                                  exam_y = examDf.loc[:,'yyy']  
step2: 基于已给的数据集，分出来训练集和测试集：    
                                  from sklearn.model_selection import train_split  
                                  X_train, X_test, y_train, t_test = train_split(exam_X,exam_y,train_size = .8) #分类百分比80%    
step3:建立训练模型：  
                                  from sklearn.linear_model import LinearRegression  
                                  model = LinearRegression  
                                  model.fit(X_train , y_train) #训练模型,注意reshape()函数的使用 -->此时已获得模型 model  
step4:基于已求得模型，计算截距a和回归系数b:(也可以不求)  
                                  a = model.intercept_  
                                  b = model.coef_  
step5:训练数据的预测值,做出拟合曲线：  
                                 y_train_pred = model.predict(X_train)  
                                  plt.plot(X_train, y_train_pred)  
step6:模型评估：  
                                  model.score(X_test , y_test) #注意模型评估是用测试集来做。就好比我们用模拟考试的表现来进行模型建立，参数求解。  
                                                                但是想要真正的测试模型的好坏，还是需要用真正的考试来进行验证。  

I also used Linear Regression method to predict the Boston housing price tonight(2022/2/20):  
The problem I found is here:https://www.cnblogs.com/wish-together/p/14764148.html.Following the author's step just to learn efficiently.  
Besides, there is a little difference from precious problem in Github. There are 13 variables among this problem and we can't plot them in scatter plot. Instead, we could plot them in form of the curve，which is shown in attachment(jpg). Now I have already familarize the linear regression which is good for me to understand deeply in mathematic perspective in the future. Besides, I should spend some time to learn data vision which is very helpful to analysis the problem.  

Edit on Feb/21,2022
Reviewed the logistic regression. It is important to understand ROC,AUC in problem of certain classes/groups.  
I attach it in here which is very helpful to understand the concept: https://zh.wikipedia.org/zh-cn/ROC曲线. 
The main difference between logistic regression and linear regression is that: linear regressin mainly deals with the problem without clear boundary.  
Logistic regression deals with 0/1 problems, which is useful at the problem of classification.  
I found out a nice blog which clearly clarified the difference between two above: https://blog.csdn.net/likewind1993/article/details/84196135. 

