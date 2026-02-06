"""
批量图片识别测试工具

使用示例:
    # TemplateMatch 模式 - 单个模板
    python main.py -i ./screenshots -t ./template.png

    # TemplateMatch 模式 - 模板文件夹（对撞模式）
    python main.py -i ./screenshots -t ./templates/

    # OCR 模式
    python main.py -i ./screenshots --mode OCR --expected "目标文本"

    # 自定义参数
    python main.py -i ./screenshots --mode OCR -p '{"only_rec": true}'
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from maa.controller import CustomController
from maa.pipeline import (
    JRecognitionType,
    JTemplateMatch,
    JOCR,
    JColorMatch,
    JAnd,
    JOr,
    JDirectHit,
    JRecognitionParam,
)
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit


# 支持的识别模式
RECOGNITION_MODES = {
    "TemplateMatch": JRecognitionType.TemplateMatch,
    "OCR": JRecognitionType.OCR,
    "ColorMatch": JRecognitionType.ColorMatch,
    "DirectHit": JRecognitionType.DirectHit,
    "And": JRecognitionType.And,
    "Or": JRecognitionType.Or,
}


class DummyController(CustomController):
    """用于初始化 Tasker 的空 Controller"""

    def connect(self) -> bool:
        return True

    def request_uuid(self) -> str:
        return "dummy"

    def screencap(self) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)


def load_image_bgr(path: Path) -> np.ndarray:
    """加载图片并转换为 BGR 格式的 numpy 数组"""
    from PIL import ImageOps

    # 读取图片
    pil_img = Image.open(path)

    # 处理 EXIF 方向信息（手机拍摄的照片可能有旋转标记）
    pil_img = ImageOps.exif_transpose(pil_img)

    # 确保是 RGB 模式（处理灰度图、RGBA 等情况）
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 转为 numpy 数组并确保连续
    rgb = np.ascontiguousarray(np.array(pil_img))

    # RGB -> BGR
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr


@dataclass
class TestResult:
    """测试结果"""

    image_path: str
    template_name: str = ""  # 模板名称（对撞模式）
    succeeded: bool = False
    hit: bool = False
    box: Optional[Tuple[int, int, int, int]] = None
    algorithm: str = ""
    detail: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    draw_images: Optional[List[np.ndarray]] = None


def find_images(path: Path, recursive: bool = False) -> List[Path]:
    """查找图片文件"""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []

    if not path.is_dir():
        return []

    if recursive:
        return sorted([f for f in path.rglob("*") if f.suffix.lower() in extensions])
    else:
        return sorted([f for f in path.glob("*") if f.suffix.lower() in extensions])


def run_test(
    image_path: Path,
    tasker: Tasker,
    reco_type: JRecognitionType,
    reco_param: JRecognitionParam,
    template_name: str = "",
) -> TestResult:
    """对单张图片运行识别测试"""
    result = TestResult(image_path=str(image_path), template_name=template_name)

    # 加载图片
    try:
        image = load_image_bgr(image_path)
    except Exception as e:
        result.error = f"Failed to load image: {e}"
        return result

    # 执行识别
    try:
        job = tasker.post_recognition(reco_type, reco_param, image)
        task_detail = job.get(wait=True)

        if task_detail and task_detail.nodes:
            node = task_detail.nodes[0]
            reco = node.recognition
            if reco:
                result.succeeded = True
                result.hit = reco.hit
                result.algorithm = str(reco.algorithm)
                result.detail = reco.raw_detail
                result.draw_images = reco.draw_images if reco.draw_images else None
                if reco.box:
                    result.box = (
                        reco.box.x,
                        reco.box.y,
                        reco.box.w,
                        reco.box.h,
                    )
    except Exception as e:
        result.error = f"Recognition failed: {e}"

    return result


def save_draw_images(result: TestResult, output_dir: Path) -> List[str]:
    """保存绘制的结果图片，返回保存的文件路径列表"""
    saved_paths = []
    if not result.draw_images:
        return saved_paths

    # 获取原图片的基础名
    base_name = Path(result.image_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, draw_img in enumerate(result.draw_images):
        if draw_img is None or draw_img.size == 0:
            continue

        # 生成输出文件名
        if len(result.draw_images) == 1:
            out_name = f"{base_name}_draw.png"
        else:
            out_name = f"{base_name}_draw_{idx}.png"

        out_path = output_dir / out_name

        # BGR -> RGB 然后保存
        # 确保数组连续并明确指定为 RGB 模式
        if len(draw_img.shape) == 3 and draw_img.shape[2] >= 3:
            rgb_img = np.ascontiguousarray(draw_img[:, :, ::-1])
            pil_img = Image.fromarray(rgb_img, mode="RGB")
        else:
            rgb_img = np.ascontiguousarray(draw_img)
            pil_img = Image.fromarray(rgb_img)

        pil_img.save(out_path)
        saved_paths.append(str(out_path))

    return saved_paths


def build_reco_param(
    mode: str,
    template_name: Optional[str],
    threshold: float,
    expected: List[str],
    extra_params: Dict[str, Any],
) -> Tuple[JRecognitionType, JRecognitionParam]:
    """根据模式构建识别参数"""
    reco_type = RECOGNITION_MODES[mode]

    if mode == "TemplateMatch":
        if not template_name:
            raise ValueError("TemplateMatch 模式需要 --template 参数")
        base_params = {"template": [template_name], "threshold": [threshold]}
        base_params.update(extra_params)
        return reco_type, JTemplateMatch(**base_params)

    elif mode == "OCR":
        base_params = {"expected": expected, "threshold": threshold}
        base_params.update(extra_params)
        return reco_type, JOCR(**base_params)

    elif mode == "ColorMatch":
        # ColorMatch 必须通过 -p 提供 lower 和 upper
        if "lower" not in extra_params or "upper" not in extra_params:
            raise ValueError("ColorMatch 模式需要通过 -p 提供 lower 和 upper 参数")
        return reco_type, JColorMatch(**extra_params)

    elif mode == "DirectHit":
        return reco_type, JDirectHit()

    elif mode == "And":
        # And 模式需要通过 -p 提供 all_of
        if "all_of" not in extra_params:
            raise ValueError("And 模式需要通过 -p 提供 all_of 参数")
        return reco_type, JAnd(**extra_params)

    elif mode == "Or":
        # Or 模式需要通过 -p 提供 any_of
        if "any_of" not in extra_params:
            raise ValueError("Or 模式需要通过 -p 提供 any_of 参数")
        return reco_type, JOr(**extra_params)

    else:
        raise ValueError(f"不支持的识别模式: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="MaaBatchTest - 批量图片识别测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # TemplateMatch 模式（默认）
    python main.py -i ./screenshots -t ./template.png

    # OCR 模式
    python main.py -i ./screenshots --mode OCR --expected "目标文本"

    # OCR 模式（仅识别，不匹配）
    python main.py -i ./screenshots --mode OCR -p '{"only_rec": true}'

    # ColorMatch 模式
    python main.py -i ./screenshots --mode ColorMatch -p '{"lower": [[0,0,0]], "upper": [[255,255,255]]}'

    # And 模式（组合多个识别）
    python main.py -i ./screenshots --mode And -p '{"all_of": [{"recognition": "OCR", "expected": ["文本"]}, {"recognition": "TemplateMatch", "template": ["btn.png"]}]}'
""",
    )

    parser.add_argument(
        "-i",
        "--images",
        type=Path,
        required=True,
        help="目标图片文件夹或单个图片路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RECOGNITION_MODES.keys()),
        default="TemplateMatch",
        help="识别模式（默认 TemplateMatch）",
    )
    parser.add_argument(
        "-t",
        "--template",
        type=Path,
        help="模板图片路径（TemplateMatch 模式必需）",
    )
    parser.add_argument(
        "--expected",
        type=str,
        nargs="+",
        default=[],
        help="OCR 期望匹配的文本列表",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="匹配阈值（默认 0.7）",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default="{}",
        help="额外的识别参数（JSON 格式）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.json"),
        help="输出 JSON 结果文件（默认 results.json）",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="禁止递归子目录（默认递归）",
    )

    args = parser.parse_args()
    recursive = not args.no_recursive

    # 解析额外参数
    try:
        extra_params = json.loads(args.params)
        if not isinstance(extra_params, dict):
            raise ValueError("params 必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: 无效的 JSON 参数: {e}", file=sys.stderr)
        sys.exit(1)

    # 验证模板文件（TemplateMatch 模式）
    template_paths: List[Path] = []
    if args.mode == "TemplateMatch":
        if not args.template:
            print("Error: TemplateMatch 模式需要 --template 参数", file=sys.stderr)
            sys.exit(1)
        if not args.template.exists():
            print(f"Error: Template not found: {args.template}", file=sys.stderr)
            sys.exit(1)
        # 支持单个模板或模板文件夹
        template_paths = find_images(args.template, recursive=recursive)
        if not template_paths:
            print(
                f"Error: No template images found in {args.template}", file=sys.stderr
            )
            sys.exit(1)

    # 查找图片
    image_paths = find_images(args.images, recursive=recursive)
    if not image_paths:
        print(f"Error: No images found in {args.images}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s)")
    print(f"Mode: {args.mode}")
    if args.mode == "TemplateMatch":
        print(f"Templates: {len(template_paths)} template(s)")
        for tp in template_paths[:5]:  # 只显示前5个
            print(f"  - {tp.name}")
        if len(template_paths) > 5:
            print(f"  ... and {len(template_paths) - 5} more")
    if args.mode == "OCR" and args.expected:
        print(f"Expected: {args.expected}")
    print(f"Threshold: {args.threshold}")
    if extra_params:
        print(f"Extra params: {json.dumps(extra_params, ensure_ascii=False)}")
    print("-" * 50)

    # 初始化 MAA
    Toolkit.init_option("./")
    Tasker.set_debug_mode(True)
    Tasker.set_save_draw(True)

    # 创建临时资源目录
    resource_dir = Path("./temp_resource")
    resource_dir.mkdir(exist_ok=True)
    (resource_dir / "pipeline").mkdir(exist_ok=True)
    (resource_dir / "image").mkdir(exist_ok=True)

    # 复制所有模板图片
    for tpl_path in template_paths:
        dest = resource_dir / "image" / tpl_path.name
        shutil.copy(tpl_path, dest)

    # 创建空的 pipeline 配置
    (resource_dir / "pipeline" / "default.json").write_text("{}", encoding="utf-8")

    # 加载资源
    resource = Resource()
    if not resource.post_bundle(resource_dir).wait().succeeded:
        print("Error: Failed to load resource", file=sys.stderr)
        sys.exit(1)

    # 创建 Tasker 并绑定资源
    controller = DummyController()
    controller.set_screenshot_target_short_side(720)
    controller.post_connection().wait()

    tasker = Tasker()
    tasker.bind(resource, controller)

    if not tasker.inited:
        print("Error: Tasker not initialized", file=sys.stderr)
        sys.exit(1)

    # 输出目录
    output_dir = args.output.parent if args.output.parent != Path(".") else Path(".")

    # 运行测试
    all_results: Dict[str, List[TestResult]] = {}  # 按模板名分组
    total_hit_count = 0
    total_test_count = 0

    if args.mode == "TemplateMatch" and template_paths:
        # 对撞模式：每个模板与每张图片匹配
        for tpl_idx, tpl_path in enumerate(template_paths):
            tpl_name = tpl_path.stem  # 模板名（不含扩展名）
            print(f"\n[Template {tpl_idx+1}/{len(template_paths)}] {tpl_path.name}")
            print("-" * 40)

            # 构建该模板的识别参数
            try:
                reco_type, reco_param = build_reco_param(
                    args.mode,
                    tpl_path.name,
                    args.threshold,
                    args.expected,
                    extra_params,
                )
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                continue

            template_results: List[TestResult] = []
            hit_count = 0

            for i, img_path in enumerate(image_paths):
                print(f"  [{i+1}/{len(image_paths)}] {img_path.name}", end=" ... ")

                result = run_test(
                    img_path, tasker, reco_type, reco_param, tpl_path.name
                )
                template_results.append(result)
                total_test_count += 1

                # 保存命中的绘制结果到对应模板目录
                if result.hit:
                    hit_count += 1
                    total_hit_count += 1
                    draw_dir = output_dir / "hits" / tpl_name
                    saved_imgs = save_draw_images(result, draw_dir)
                    box_str = f" @ {result.box}" if result.box else ""
                    img_str = f" [{len(saved_imgs)} img(s)]" if saved_imgs else ""
                    print(f"HIT{box_str}{img_str}")
                elif result.error:
                    print(f"ERROR: {result.error}")
                else:
                    print("MISS")

            all_results[tpl_path.name] = template_results
            print(f"  Template [{tpl_path.name}]: {hit_count}/{len(image_paths)} hit")

    else:
        # 非对撞模式：单模板或其他识别模式
        template_name = template_paths[0].name if template_paths else None
        try:
            reco_type, reco_param = build_reco_param(
                args.mode,
                template_name,
                args.threshold,
                args.expected,
                extra_params,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        results: List[TestResult] = []
        hit_count = 0

        for i, img_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] {img_path.name}", end=" ... ")

            result = run_test(
                img_path, tasker, reco_type, reco_param, template_name or ""
            )
            results.append(result)
            total_test_count += 1

            draw_output_dir = output_dir / "draw_results"
            saved_imgs = save_draw_images(result, draw_output_dir)

            if result.error:
                print(f"ERROR: {result.error}")
            elif result.hit:
                hit_count += 1
                total_hit_count += 1
                box_str = f" @ {result.box}" if result.box else ""
                img_str = f" [{len(saved_imgs)} img(s)]" if saved_imgs else ""
                print(f"HIT{box_str}{img_str}")
            else:
                print("MISS")

        all_results[template_name or "default"] = results

    # 统计
    print("\n" + "=" * 50)
    print(f"Total Results: {total_hit_count}/{total_test_count} hit")

    # 输出 JSON（简洁格式：按模板分组，只记录分数和命中图片）
    output_data = {}
    for tpl_name, results in all_results.items():
        hit_images = [r.image_path for r in results if r.hit]
        total = len(results)
        hit_count = len(hit_images)
        output_data[tpl_name] = {
            "score": f"{hit_count}/{total}",
            "hit_images": hit_images,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {args.output}")

    hits_dir = output_dir / "hits"
    if hits_dir.exists():
        print(f"Hit images saved to: {hits_dir}")

    # 打印分数列表
    print("\n" + "=" * 50)
    print("Score Summary:")
    for tpl_name, data in output_data.items():
        print(f"  {tpl_name}: {data['score']}")

    sys.exit(0 if total_hit_count > 0 else 1)


if __name__ == "__main__":
    main()
