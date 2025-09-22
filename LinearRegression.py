# =============================================
# 线性回归示例：基于芝加哥出租车数据预测车费（FARE）
# =============================================
# 你现在看到的是一个最简单的“结构化数据 + 深度学习线性模型”的例子。
# 整体流程：
# 1. 读取数据集（CSV）
# 2. 选取需要的特征列（输入）和标签列（输出）
# 3. 做一些简单的探索（查看前几行、描述统计、相关性、可视化）
# 4. 定义一个线性回归模型（Keras 实现）
# 5. 训练模型（model.fit）
# 6. 观察训练指标（损失 / RMSE）、画图查看拟合效果
# =============================================
# 你需要先理解：
# - “特征”(feature)：模型的输入，如 TRIP_MILES（行程里程）
# - “标签”(label/target)：模型要预测的值，如 FARE（车费）
# - “超参数”(hyperparameters)：不是模型学出来的，而是你手动设定的值，如学习率、批量大小、训练轮数
# - “epoch”：整份训练数据被完整训练一遍叫 1 个 epoch
# - “batch”：一次喂给模型的一小部分数据，大小由 batch_size 决定
# - “loss”：训练时优化的目标（这里是均方误差 MSE）
# - “rmse”：均方根误差，是 MSE 开平方，更直观（单位与原目标一致）
# =============================================

import numpy as np          # 数值计算库（这里暂时只用到很少功能，可选）
import pandas as pd         # 处理表格数据的核心库（DataFrame）

# 机器学习 / 深度学习相关
import keras                # Keras 是一个高层深度学习框架接口，这里用它构建和训练模型
import ml_edu.experiment    # 自定义（或外部提供）的模块：貌似封装了实验设置与结果对象（不是标准库）
import ml_edu.results       # 同上：用于结果可视化（画训练曲线、预测散点等）

# 数据可视化（交互式图表）
import plotly.express as px # Plotly Express 简化版 API，用于快速生成图

# 1. 读取原始数据集 CSV 文件。假设文件与当前脚本在同一目录下。
# read_csv 会返回一个 DataFrame。
chicago_taxi_dataset = pd.read_csv("chicago_taxi_train.csv")

# 2. 只选择我们关心的列，生成一个新的 DataFrame：training_df。
# loc[:, (列名集合)] 表示：选取所有行，后面括号里这些指定的列。
# 这些列包含：
# TRIP_MILES     行程里程（数值型特征）
# TRIP_SECONDS   行程时间（秒）（数值型特征）
# FARE           车费（我们要预测的标签）
# COMPANY        出租车公司（类别型特征，当前模型中未使用，只是保留）
# PAYMENT_TYPE   支付方式（类别型特征）
# TIP_RATE       小费比例（数值型或衍生特征）
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))  # 打印行数（样本量）

# training_df.head(200) 在脚本里单独写不会打印，需要 print(...)。
# 在 Jupyter Notebook 里最后一行会自动显示，这里如果你想看内容可改为：print(training_df.head(200))
training_df.head(200)  # 不影响执行，可留着（无输出）。

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
# describe(include='all') 给出数值/类别列的统计信息（计数、均值、标准差、最值、频次等）
# 在脚本中同样需要 print 才显示；如想输出可写：print(training_df.describe(include='all'))
training_df.describe(include='all')

# 相关性矩阵：只对数值列计算（numeric_only=True）。矩阵中值 ∈ [-1,1]，越靠近 ±1 相关性越强。
# 这里没有保存也没有 print，所以脚本不会显示；需要的话加 print(training_df.corr(numeric_only=True))
training_df.corr(numeric_only = True)

# 使用 Plotly 画散点矩阵（FARE vs TRIP_MILES vs TRIP_SECONDS）。
# 在纯终端运行时不会自动弹出（除非打开渲染器），在 Notebook 中会显示交互图。
px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])

# =============================
# 定义两个核心函数：create_model / train_model
# =============================

# create_model: 根据输入特征构建一个“线性回归”结构的 Keras 模型。
# 注意：这里虽然用了 Dense 层（全连接层），但由于只有一层 + 无激活，就等价于线性回归。
# settings:     实验设置对象（包含学习率、输入特征列表等）
# metrics:      训练时额外想观察的指标列表（如 rmse）
# 返回：编译完成的 keras.Model

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """创建并编译一个简单的线性回归模型。

  步骤说明：
  1. 为每个输入特征创建一个 Keras 输入层（shape=(1,) 表示该特征是一个标量数值）。
  2. 将所有输入通过 Concatenate 拼接成一个向量（如果只有 1 个特征，拼接后还是 1 维）。
  3. 使用 Dense(units=1) 得到单输出（预测 FARE）。没有激活函数 => 线性输出。
  4. 使用 RMSprop 优化器（学习率来自 settings），loss 设为均方误差（MSE）。
  5. 额外 metrics（如 rmse）用于观测，不用于反向传播优化（优化只看 loss）。
  """
  # 构建输入：字典推导式，key=特征名，value=对应的 Input 层
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  # 将多个输入（如果有）按顺序合并成一个张量
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  # 输出层：units=1 => 预测一个标量（车费）。等价公式：y_hat = w1*x1 + w2*x2 + ... + b
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  # 构建模型对象（多输入 -> 单输出）
  model = keras.Model(inputs=inputs, outputs=outputs)

  # 编译：指定优化器、损失函数、监控指标
  # loss = "mean_squared_error" 等价于 keras.losses.MeanSquaredError()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)

  return model


# train_model: 负责真正调用 model.fit 进行训练。
# experiment_name: 实验名称（用于区分不同配置训练的结果）
# model:           已经编译好的 Keras 模型
# dataset:         训练用的 DataFrame（包含特征列与标签列）
# label_name:      标签列名（这里是 'FARE'）
# settings:        超参数配置对象
# 返回：ml_edu.experiment.Experiment（封装训练信息的对象，便于后续画图或分析）

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """训练模型：将 DataFrame 中的列转为 numpy 数组喂给 Keras。

  关键点：
  1. Keras 支持多输入：这里用一个 dict：{特征名: ndarray}
  2. label 也取出为 ndarray
  3. model.fit 里指定 batch_size（每次更新使用多少样本）与 epochs（总轮数）
  4. history.history 是一个字典：键=指标名（loss, rmse），值=list（每个 epoch 的值）
  """

  # features: dict，key=特征名，value=对应列的值（.values -> numpy array）
  features = {name: dataset[name].values for name in settings.input_features}
  # label: 目标值数组（与 features 中行数对应）
  label = dataset[label_name].values
  # 训练：返回 History 对象
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)

  # 将训练过程包装成 Experiment 对象返回（便于统一处理 / 画图）
  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),  # 把指标字典转为 DataFrame（每列是一个指标）
  )

print("SUCCESS: defining linear regression functions complete.")

# =============================
# 运行一个实验（Experiment 1）
# =============================

# 定义实验的超参数（Hyperparameters）：
# learning_rate  学习率：控制权重更新步子大小（过大震荡，过小收敛慢）
# number_epochs  总训练轮数：数据被完整迭代多少次
# batch_size     每个 batch 样本数：影响梯度估计噪声与显存占用
# input_features 输入特征列表：这里只选一个 'TRIP_MILES'，相当于单变量线性回归
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES']
)

# 定义评估指标列表：这里只加一个均方根误差 rmse
# rmse = sqrt(mse)，值越小表示预测越接近真实车费
metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

# 基于上述设置创建模型
model_1 = create_model(settings_1, metrics)

# 训练：特征 = TRIP_MILES；标签 = FARE
experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

# 可视化训练过程中的 rmse 随 epoch 变化（学习曲线）
ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
# 可视化预测结果 vs 真实值（例如散点图）
ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')

# =============================
# 你可以做的后续练习：
# 1. 把 input_features 改成 ['TRIP_MILES', 'TRIP_SECONDS'] 看是否提升。
# 2. 调整 learning_rate：试 0.01 / 0.0005 对比收敛速度和稳定性。
# 3. 增加 epochs（比如 50）观察 loss 曲线是否继续下降或过拟合。
# 4. 尝试添加类别特征：需要先做独热编码（one-hot），否则不能直接作为数值输入。
# 5. 把 loss 改成 MAE (mean_absolute_error) 比较与 MSE 的差异。
# =============================
