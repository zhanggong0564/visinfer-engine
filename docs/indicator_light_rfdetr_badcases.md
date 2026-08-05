# 指示灯 RF-DETR Bad Case

## 当前数据集

- 数据目录：`demo/data/light`
- 注册图：57张
- 待测图：428张
- 总软链接：485个
- 断链：0
- 图片均为正样本，预期结果为 PASS。
- 已排除30张内容旋转的图片，不纳入当前评测。

已调整的注册图：

- `A0ST7262-1`：`1783413346459.jpg`
- `A0SW1685-1`：`1784105342639.jpg`

`A0SW1685-1` 原注册图 `1784099222396.jpg` 和同批11张横向旋转待测图均已
移除软链接。使用新注册图复测该组14张正向待测图：13张 PASS、1张 FAIL。

## 评测配置

- 检测模型：`weights/indicator_light/rfdetr-small.onnx`
- 识别模型：`weights/indicator_light/rec_v3.onnx`
- 检测置信度：`0.8`
- embedding 相似度阈值：`0.65`

## 当前 Bad Case

| 编号 | 型号-机位 | 注册图 | 待测图 | 原因 |
|---:|---|---|---|---|
| 1 | A0SW0415-1 | `1784075381455.jpg` | `1783599061028.jpg` | 灯位匹配后状态 embedding 不匹配 |
| 2 | A0SW1685-1 | `1784105342639.jpg` | `1784105351623.jpg` | 注册12 / 待测11，缺少注册灯位 |
| 3 | A0SW2053-1 | `1783937659702.jpg` | `1783922766526.jpg` | 灯位匹配后状态 embedding 不匹配 |
| 4 | A0SW2195-2 | `1783914556729.jpg` | `1783905346654.jpg` | 检测数均为6，但布局无法可靠匹配 |
| 5 | A0SW2195-2 | `1783914556729.jpg` | `1783905348085.jpg` | 检测数均为6，但布局无法可靠匹配 |

失败类型统计：

- 缺少注册灯位：1张
- 状态 embedding 不匹配：2张
- 灯位布局无法可靠匹配：2张

当前数据集尚未完整复测全部428张待测图，因此本文不提供整体正确率。

## 逐项调试命令

从仓库根目录执行，参数顺序为“待测图、注册图”。可视化结果保存为根目录下的
`indicator_light_result.jpg`。

### 1. A0SW0415-1 / 1783599061028.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0SW0415-1/current/1783599061028.jpg \
  demo/data/light/A0SW0415-1/registered/1784075381455.jpg
```

### 2. A0SW1685-1 / 1784105351623.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0SW1685-1/current/1784105351623.jpg \
  demo/data/light/A0SW1685-1/registered/1784105342639.jpg
```

### 3. A0SW2053-1 / 1783922766526.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0SW2053-1/current/1783922766526.jpg \
  demo/data/light/A0SW2053-1/registered/1783937659702.jpg
```

### 4. A0SW2195-2 / 1783905346654.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0SW2195-2/current/1783905346654.jpg \
  demo/data/light/A0SW2195-2/registered/1783914556729.jpg
```

### 5. A0SW2195-2 / 1783905348085.jpg

```bash
INDICATOR_DET_CONF_THRESHOLD=0.8 conda run -n mobile_vision python \
  plugins/vie-plugin-indicator-light/examples/run.py \
  demo/data/light/A0SW2195-2/current/1783905348085.jpg \
  demo/data/light/A0SW2195-2/registered/1783914556729.jpg
```
