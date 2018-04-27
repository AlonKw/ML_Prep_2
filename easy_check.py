import pandas as pd
import scale_data

location = 'datasets\\1\\{}.csv'

df = pd.read_csv(location.format("X_train1No_Nan"), header=0)
dy = pd.read_csv(location.format("Y_train1"), header=0)

head_tag = "Vote"
df[head_tag] = dy[head_tag]

df.corr().to_csv("Correlation_befor_scale.csv")
df.to_csv("Data_before_scale.csv")
sd = scale_data.ScaleData()
sd.scale_train(df)

df.corr().to_csv("Correlation_after_scale.csv")
df.to_csv("Data_after_scale.csv")