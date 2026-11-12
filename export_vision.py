"""
export_vision.py - xuất bộ mã hoá ảnh (SigLIP + gộp mảnh + Projector đã train) thành MỘT file ONNX cho app Y-App.

    python export_vision.py --out <thư_mục_model> --dst yui-vision.onnx [--no_quant]

Vào "pixel_values" [1,3,224,224] (RGB, đã chuẩn hoá về [-1,1]); ra "image_embeds" [N, D]  (N = (14/pool)^2, D = chiều embedding của LM).
Mặc định lượng tử hoá int8 (~90 MB thay vì ~350 MB). Sau khi xuất tự so với PyTorch (cần onnxruntime) để chắc không lệch.
Chép yui-vision.onnx vào điện thoại rồi nhập trong app (nút ảnh trong khung chat). Chỉ dùng với file .yui xuất từ vision_best.pt.
"""
import argparse, os
import torch
import torch.nn as nn
import vision_yui as V


class Wrap(nn.Module):
    def __init__(self, tower, proj):
        super().__init__()
        self.tower, self.proj = tower, proj

    def forward(self, pixel_values):
        return self.proj(self.tower(pixel_values))[0]        # [N, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục có vision_proj.pt")
    ap.add_argument("--dst", default="yui-vision.onnx")
    ap.add_argument("--no_quant", action="store_true")
    a = ap.parse_args()
    meta = torch.load(os.path.join(a.out, "vision_proj.pt"), map_location="cpu")
    tower = V.VisionTower(meta["pool"]).eval()
    proj = V.Projector(tower.out_dim, meta["d_model"]).eval()
    proj.load_state_dict(meta["proj"])
    net = Wrap(tower, proj).eval()
    x = torch.randn(1, 3, 224, 224)
    raw = a.dst if a.no_quant else a.dst + ".fp32.onnx"
    kw = dict(input_names=["pixel_values"], output_names=["image_embeds"], opset_version=17)
    try:
        torch.onnx.export(net, x, raw, dynamo=False, **kw)
    except TypeError:                       # torch cũ không có tham số dynamo
        torch.onnx.export(net, x, raw, **kw)
    if not a.no_quant:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(raw, a.dst, weight_type=QuantType.QInt8)
        os.remove(raw)
    print(f"đã ghi {a.dst} ({os.path.getsize(a.dst) / 1e6:.1f} MB), {meta['n_img']} token ảnh x {meta['d_model']} chiều")
    try:
        import onnxruntime as ort
        ref = net(x).detach().numpy()
        got = ort.InferenceSession(a.dst).run(None, {"pixel_values": x.numpy()})[0]
        cos = (ref * got).sum() / (((ref ** 2).sum() ** 0.5) * ((got ** 2).sum() ** 0.5) + 1e-9)
        print(f"so với PyTorch: cos {cos:.4f}, lệch tối đa {abs(ref - got).max():.4f}" + ("   ⚠ lệch nhiều, thử --no_quant" if cos < 0.98 else ""))
    except ImportError:
        print("(không có onnxruntime nên bỏ qua bước so sánh)")


if __name__ == "__main__":
    main()
