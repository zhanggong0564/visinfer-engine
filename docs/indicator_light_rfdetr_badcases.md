# 指示灯 RF-DETR Bad Case

## 全量评测结果

- 评测时间：2026-08-05
- 数据目录：`demo/data/light`
- 型号-机位组：57组
- 注册图：57张
- 待测图：391张
- 总软链接：448个
- 断链：0
- 检测模型：`weights/indicator_light/rfdetr-small_v1.1.onnx`
- 识别模型：`weights/indicator_light/rec_v3.onnx`
- 检测置信度：`0.8`
- embedding 相似度阈值：`0.65`

全部待测图均按正样本统计，预期结果为 PASS：

- PASS：382张
- FAIL：9张
- ERROR：0张
- 正确率：97.70%
- 评测耗时：227.73秒
- 机器可读报告：`/tmp/indicator_light_layout_eval.json`

## 当前 Bad Case

| 编号 | 型号-机位 | 注册图 | 待测图 | 原因 |
|---:|---|---|---|---|
| 1 | A0HR1017-1 | `1783931100632.jpg` | `1783931221412.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 2 | A0HR1017-1 | `1783931100632.jpg` | `1783931225270.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 3 | A0HR1017-1 | `1783931100632.jpg` | `1783933377593.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 4 | A0HR1017-1 | `1783931100632.jpg` | `1783933380650.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 5 | A0HR1017-1 | `1783931100632.jpg` | `1783933382808.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 6 | A0HR1017-1 | `1783931100632.jpg` | `1783933385125.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 7 | A0HR1017-1 | `1783931100632.jpg` | `1783933387269.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 8 | A0HR1017-1 | `1783931100632.jpg` | `1783934763267.jpg` | 注册7 / 待测6，缺少注册灯位 |
| 9 | A0SW1685-1 | `1784096552184.jpg` | `1784099273095.jpg` | 注册13 / 待测12，缺少注册灯位 |

9张失败均为检测数量不足，没有 embedding 不匹配、布局歧义或推理错误。

## 逐项调试命令

从仓库根目录执行，参数顺序为“待测图、注册图”。可视化结果保存为根目录下的
`indicator_light_result.jpg`。

### 1. A0HR1017-1 / 1783931221412.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783931221412.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 2. A0HR1017-1 / 1783931225270.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783931225270.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 3. A0HR1017-1 / 1783933377593.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783933377593.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 4. A0HR1017-1 / 1783933380650.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783933380650.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 5. A0HR1017-1 / 1783933382808.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783933382808.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 6. A0HR1017-1 / 1783933385125.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783933385125.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 7. A0HR1017-1 / 1783933387269.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783933387269.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 8. A0HR1017-1 / 1783934763267.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0HR1017-1/current/1783934763267.jpg \
  demo/data/light/A0HR1017-1/registered/1783931100632.jpg
```

### 9. A0SW1685-1 / 1784099273095.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0SW1685-1/current/1784099273095.jpg \
  demo/data/light/A0SW1685-1/registered/1784096552184.jpg
```
