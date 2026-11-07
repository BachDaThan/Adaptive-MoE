"""Test nhanh cho persona_fix.py, teencode.py, vi_filter.py (không cần mạng/torch)."""
import json
import random

import persona_fix
import teencode
import vi_filter

GOOD = ("Đồng bằng sông Cửu Long là vùng đất rộng lớn ở phía nam của Việt Nam, nơi sông Mê Kông chia thành nhiều nhánh nhỏ trước khi đổ ra biển. "
        "Phù sa mang về mỗi mùa nước nổi giúp ruộng lúa luôn màu mỡ, còn hệ thống kênh rạch chằng chịt là con đường giao thương quen thuộc của người dân.\n"
        "Vào mùa khô, nước mặn từ biển lấn sâu vào đất liền khiến nhiều hộ phải đổi giống cây trồng hoặc chuyển sang nuôi tôm. "
        "Các nhà khoa học cho rằng việc xây đập ở thượng nguồn và khai thác cát quá mức đã làm lượng phù sa giảm đi đáng kể.\n"
        "Chính quyền địa phương đang thử nhiều mô hình thích ứng như trồng lúa chịu mặn, làm hồ trữ nước ngọt và kết hợp lúa với nuôi tôm. "
        "Dù vậy, câu chuyện lâu dài vẫn là cân bằng giữa phát triển kinh tế và giữ gìn hệ sinh thái mong manh của vùng châu thổ này, "
        "nơi hàng triệu người đang sống và làm việc dựa vào nhịp lên xuống của dòng nước.")


def test_persona_converts_address_only():
    f = persona_fix.fix
    assert f("Là Canberra đó anh, không phải Sydney đâu.") == "Là Canberra đó Boss, không phải Sydney đâu."
    assert f("Anh ôn lại phần hay sai nhất nha.").startswith("Boss ôn")
    assert f("Yui kể chuyện cười cho anh nghe nha.") == "Yui kể chuyện cười cho Boss nghe nha."
    assert f("Mèo của bạn tên Mun nè.") == "Mèo của Boss tên Mun nè."
    # không đụng các nghĩa khác
    for keep in ("Anh hùng Nguyễn Huệ đánh quân Thanh.", "Anh chị em trong nhà ai cũng thích.", "Nước Anh nằm ở châu Âu.",
                 "London là thủ đô của Anh.", "Bạn bè của mình đông lắm.", "Anh ấy đến muộn.", "Các bạn nhỏ chơi vui nha."):
        assert f(keep) == keep, keep


def test_persona_only_touches_assistant_turns():
    line = json.dumps({"messages": [{"role": "user", "content": "anh hỏi nghiêm túc nè"}, {"role": "assistant", "content": "Dạ anh hỏi đi nha"}]}, ensure_ascii=False)
    new, ch = persona_fix.fix_conv_line(line)
    m = json.loads(new)["messages"]
    assert ch and m[0]["content"] == "anh hỏi nghiêm túc nè" and m[1]["content"] == "Dạ Boss hỏi đi nha"
    assert persona_fix.fix_conv_line("không phải json") == ("không phải json", False)
    assert persona_fix.fix_conv_line(new)[1] is False   # chạy lại không đổi gì nữa (idempotent)


def test_teencode_and_no_diacritics():
    assert teencode.strip_diacritics("Đường Hồ Chí Minh") == "Duong Ho Chi Minh"
    out = teencode.teencodify("Hôm nay mình không biết làm gì luôn", random.Random(1), p_word=1.0, p_nodau=0.0, p_lower=0.0, p_end=0.0, p_stretch=0.0)
    assert out != "Hôm nay mình không biết làm gì luôn" and len(out.split()) <= 8
    docs = [GOOD] * 5
    aug = teencode.augment_docs(docs, 5, seed=1)
    assert aug and all(20 < len(a) < 600 for a in aug)
    assert teencode.augment_docs(docs, 5, seed=1) == aug   # có seed => lặp lại được


def test_quality_filter():
    assert vi_filter.keep(GOOD) == (True, None)
    assert not vi_filter.keep("Nhà cái uy tín số 1 nạp tiền rút tiền nhanh, cá cược bóng đá, xổ số, casino. Liên hệ hotline zalo nhận khuyến mãi sốc. " * 8)[0]
    assert not vi_filter.keep("\n".join(["Trang chủ", "Giới thiệu", "Tin tức", "Liên hệ", "Sản phẩm", "Đăng nhập"] * 8))[0]
    assert not vi_filter.keep("Mua hàng giá rẻ tại cửa hàng của chúng tôi hôm nay " * 40)[0]
    assert not vi_filter.keep("This is an English paragraph about nothing in particular. " * 20)[0]
