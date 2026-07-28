# 模型权重命名规范与清单

> 本目录不入 git（体积大），但本 README 描述的规范随仓库文档维护。
> 部署同步方式见 [PACKAGING.md](../PACKAGING.md) 第 5 节（热更新部署）。

## 命名规范

```
weights/{场景}/{任务}_{架构}_v{N}[.onnx]    # 自训模型
weights/common/official/{官方原名}/          # 官方原版模型，保留官方名便于溯源
```

- **场景**：与检测工厂注册名一致（`dc_fuse` / `indicator_light` / `lap_surf` /
  `line_squeeze` / `mvs` / `panel_label` / `plate_screw`）；跨场景共享资产放 `common/`。
- **任务**：`det`（检测）、`rec`（识别）、`label_det`（线标区域检测）、`text_det`（文本检测）、
  `textline_ori`（文本行方向）、`text_rec`（文本识别）、`text_rec_plane`（平面文本识别，仅标注脚本用）。
- **架构**：简短小写——`yolo`、`ppocrv5m`（PP-OCRv5 mobile）、`ppocrv5s`（PP-OCRv5 server）、
  `lcnet`（PP-LCNet）；架构不明确或自定义结构时省略（如 `line_squeeze/det_v3.onnx`）。
- **版本**：`v{N}` 单调递增。训练实验细节（diff_lr、merged 等）**不进文件名**，记入下方清单。
  例外：同版本的实验分支无法归并时保留后缀（如 `text_rec_plane_ppocrv5s_v2_self_word`）。
- **效果指标不进文件名**：指标随评测集变化，统一记录在清单的「指标」列。

## 管理约定

- 新模型一律**新增版本号目录/文件，不覆盖旧版**；config 切换路径即上线，改回旧路径即回滚。
- 模型路径在各插件 `plugins/vie-plugin-*/vie_plugin_*/config.py` 中配置，相对启动 cwd 解析。
- 上新模型同时在下方清单补一行（架构、训练数据/备注、指标）。

## 模型清单

| 路径 | 架构 | 用途 | 训练备注 | 指标 |
|---|---|---|---|---|
| `dc_fuse/det_yolo_v5.onnx` | YOLO | 直流熔丝检测 | | 待补 |
| `indicator_light/det_yolo_v2.onnx` | YOLO | 指示灯检测 | | 待补 |
| `indicator_light/rec_v2.onnx` | ONNX embedding | 指示灯状态比对（与注册图 embedding 比对） | | 待补 |
| `indicator_light/standard_embeddings.json` | — | 标准 embedding 库 | | — |
| `lap_surf/rec_yolo_v2.onnx` | YOLO | 搭接面识别 | | 待补 |
| `line_squeeze/det_v3.onnx` | 自定义 RoiDet | 线序 ROI 检测 | | 待补 |
| `mvs/doc_ori_lcnet_v1/` | PP-LCNet x1.0 doc ori | MVS 文档方向分类 | Paddle BOS 官方模型 | — |
| `mvs/doc_unwarp_uvdoc_v1/` | UVDoc | MVS 文档矫正 | Paddle BOS 官方模型 | — |
| `mvs/text_det_ppocrv5s_v1/` | PP-OCRv5 server det | MVS 文本检测 | Paddle BOS 官方模型 | — |
| `mvs/textline_ori_lcnet_v1/` | PP-LCNet x1.0 textline ori | MVS 文本行方向分类 | Paddle BOS 官方模型 | — |
| `mvs/text_rec_ppocrv5s_v1/` | PP-OCRv5 server rec | MVS 文本识别 | Paddle BOS 官方模型 | — |
| `plate_screw/det_yolo_v2.onnx` | YOLO | 铁片螺丝检测 | | 待补 |
| `panel_label/label_det_yolo_v1~3.onnx` | YOLO | 线标区域检测 | | 待补 |
| `panel_label/text_det_plane_ppocrv5m_v1/` | PP-OCRv5 mobile det | 线标文本检测（panel 数据微调） | panel 数据微调 | 待补 |
| `panel_label/textline_ori_lcnet_v1~3/` | PP-LCNet x1.0 | 文本行方向分类 | | 待补 |
| `panel_label/text_rec_ppocrv5s_v1~3/` | PP-OCRv5 server rec | 线标文本识别 | 合并字典训练（merged） | 待补 |
| `panel_label/text_rec_ppocrv5s_v4/` | PP-OCRv5 server rec | 线标文本识别（当前线上） | merged + 差异学习率（diff_lr） | 待补 |
| `panel_label/text_rec_plane_ppocrv5s_v1~3/` | PP-OCRv5 server rec | 平面文本识别（auto_annotate 标注辅助） | plane_infer 系列 | 待补 |
| `panel_label/text_rec_plane_ppocrv5s_v2_self_word/` | PP-OCRv5 server rec | 同上，v2 自建词表实验分支 | self_word | 待补 |
| `common/official/PP-OCRv5_server_rec/` | PP-OCRv5 server rec | 官方原版（微调基座） | 官方发布 | — |
| `common/official/PP-en_rec_ppocr_v5/` | PP-OCRv5 en rec | 官方英文识别（line_squeeze OCR 用） | 官方发布 | — |

## 新旧名称映射（2026-06-10 重组）

| 旧 | 新 |
|---|---|
| `dc_fuse_v5.onnx` | `dc_fuse/det_yolo_v5.onnx` |
| `IndicatorLightDet_v2.onnx` | `indicator_light/det_yolo_v2.onnx` |
| `IndicatorLightRec_v2.onnx` | `indicator_light/rec_v2.onnx` |
| `jsons/standard_embeddings.json` | `indicator_light/standard_embeddings.json` |
| `LapJointSurfRec_v2.onnx` | `lap_surf/rec_yolo_v2.onnx` |
| `LineSqueeze_v3.onnx` | `line_squeeze/det_v3.onnx` |
| `mobile_vision_plate_v2.onnx` | `plate_screw/det_yolo_v2.onnx` |
| `official_models/*` | `common/official/*`（原名保留） |
| `panel_label/best_v{1,2,3}.onnx` | `panel_label/label_det_yolo_v{1,2,3}.onnx` |
| `panel_label/PP-OCRv5_mobile_det_panel_v1` | `panel_label/text_det_plane_ppocrv5m_v1` |
| `panel_label/PP-LCNet_x1_0_textline_ori[,_v2,_v3]` | `panel_label/textline_ori_lcnet_v{1,2,3}` |
| `panel_label/PP-OCRv5_server_rec_merged_v{1,2,3}` | `panel_label/text_rec_ppocrv5s_v{1,2,3}` |
| `panel_label/PP-OCRv5_server_rec_merged_v4_diff_lr` | `panel_label/text_rec_ppocrv5s_v4` |
| `panel_label/PP-OCRv5_server_rec_plane_infer[,_v2,_v3]` | `panel_label/text_rec_plane_ppocrv5s_v{1,2,3}` |
| `panel_label/PP-OCRv5_server_rec_plane_infer_v2_self_word` | `panel_label/text_rec_plane_ppocrv5s_v2_self_word` |

> 注意：PaddleX 代码中 `model_name="PP-LCNet_x1_0_textline_ori"` 等是**架构名参数**，
> 不是文件路径，与目录改名无关，保持官方名不变。
