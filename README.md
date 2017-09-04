# MultiDGAN

## 主目录下创分别把图片放到
```
dataset/faces
dataset/celeba
```

## 运行时有这么几个参数
```
python train.py device 0 d1 0.9 d2 01
```
device 表示在哪个 GPU 上跑， d1 表示 D1 loss 的比例， d2 同理

## log
运行之后会在主目录下创建 log 文件夹
譬如 
```
python train.py device 0 d1 0.9 d2 0.1
```
会创建一个 log/0.9_0.1/xx_xx_xx_xx/ 路径 xx 表示时间




