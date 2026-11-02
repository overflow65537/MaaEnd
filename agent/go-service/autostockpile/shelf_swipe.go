package autostockpile

import (
	"fmt"
	"image"

	maa "github.com/MaaXYZ/maa-framework-go/v4"
	"github.com/rs/zerolog/log"
)

// shelfFrameCompareStep 是判断「货架有没有动」时的采样步长（像素）。
// 逐像素比较整屏开销过大，而滚动一屏必然改变大量采样点，隔点采样已足够。
const shelfFrameCompareStep = 4

// shelfSwipeDownNodes 是货架逐屏下滑的节点顺序：先小幅下滑露出次屏，再一直滑到底。
// 下标即「露出该屏需要下滑的次数」，与 mergeGoodsPages 产出的 SwipeCountToReveal 一致。
var shelfSwipeDownNodes = []string{swipeShelfDownNodeName, swipeShelfToBottomNodeName}

func runShelfSwipe(ctx *maa.Context, nodeName string) error {
	if ctx == nil {
		return fmt.Errorf("context is nil")
	}

	_, err := ctx.RunAction(nodeName, maa.Rect{0, 0, 0, 0}, "", nil)
	return err
}

func swipeShelfUp(ctx *maa.Context) error {
	return runShelfSwipe(ctx, swipeShelfUpNodeName)
}

func screencapShelf(ctx *maa.Context) (image.Image, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is nil")
	}
	tasker := ctx.GetTasker()
	if tasker == nil {
		return nil, fmt.Errorf("tasker is nil")
	}
	ctrl := tasker.GetController()
	if ctrl == nil {
		return nil, fmt.Errorf("controller is nil")
	}

	ctrl.PostScreencap().Wait()
	img, err := ctrl.CacheImage()
	if err != nil {
		return nil, err
	}
	return img, nil
}

// shelfFramesLookIdentical 判断两帧的商品区域是否完全一致。
// 当前控制器不需要滑动（如 Win32 一屏即可看全）或货架已到尽头时，
// 下滑不会改变画面，此时再识别只会重复扫描同一屏，可以直接结束翻页。
func shelfFramesLookIdentical(previous, current image.Image) bool {
	if previous == nil || current == nil {
		return false
	}

	rect := image.Rect(
		goodsRecognitionBaseROI[0],
		goodsRecognitionBaseROI[1],
		goodsRecognitionBaseROI[0]+goodsRecognitionBaseROI[2],
		goodsRecognitionBaseROI[1]+goodsRecognitionBaseROI[3],
	).Intersect(previous.Bounds()).Intersect(current.Bounds())
	if rect.Empty() {
		return false
	}

	for y := rect.Min.Y; y < rect.Max.Y; y += shelfFrameCompareStep {
		for x := rect.Min.X; x < rect.Max.X; x += shelfFrameCompareStep {
			if !samePixel(previous, current, x, y) {
				return false
			}
		}
	}

	return true
}

// samePixel 比较两个画面上同一坐标的像素，避免直接比较 color.Color 接口值。
func samePixel(a, b image.Image, x, y int) bool {
	ar, ag, ab, aa := a.At(x, y).RGBA()
	br, bg, bb, ba := b.At(x, y).RGBA()
	return ar == br && ag == bg && ab == bb && aa == ba
}

// restoreShelf 把货架滑回首屏：下滑过几屏就上滑几次，逐次抵消。
// 首屏是后续点击商品的前提，因此失败只记录日志，不阻断本次识别结果。
func restoreShelf(ctx *maa.Context, times int) {
	for i := 0; i < times; i++ {
		if err := swipeShelfUp(ctx); err != nil {
			log.Warn().
				Err(err).
				Str("component", autoStockpileComponent).
				Str("step", "shelf_swipe_up").
				Int("restore_index", i).
				Int("restore_times", times).
				Msg("failed to restore shelf to first page")
			return
		}
	}
}

// scanGoodsAcrossShelfPages 从首屏开始逐屏下滑扫描，按商品 ID 合并所有屏结果，
// 返回合并后的商品与「露出该商品需要下滑的次数」，并尽量把货架滑回首屏。
func scanGoodsAcrossShelfPages(
	ctx *maa.Context,
	firstImg image.Image,
	region string,
	itemMap *ItemMap,
) (goods []GoodsItem, swipeCountToReveal map[string]int, abortReason AbortReason, err error) {
	if ctx == nil || firstImg == nil {
		return nil, nil, AbortReasonGoodsOCRUnavailableWarn, fmt.Errorf("ctx or image is nil")
	}

	firstPage, abortReason, err := scanGoodsOnImage(ctx, firstImg, region, itemMap)
	if abortReason != AbortReasonNone || err != nil {
		return nil, nil, abortReason, err
	}

	pages := [][]GoodsItem{firstPage}
	swiped := 0
	restored := false

	// 下游要按首屏结果点击商品，因此无论后面是否降级，都要尽量把货架滑回首屏。
	defer func() {
		if restored || swiped == 0 {
			return
		}
		restoreShelf(ctx, swiped)
	}()

	previousImg := firstImg
	for index, nodeName := range shelfSwipeDownNodes {
		pageIndex := index + 1

		if swipeErr := runShelfSwipe(ctx, nodeName); swipeErr != nil {
			log.Warn().
				Err(swipeErr).
				Str("component", autoStockpileComponent).
				Str("step", "shelf_swipe_down").
				Str("node", nodeName).
				Int("page", pageIndex).
				Int("scanned_pages", len(pages)).
				Msg("shelf swipe down failed, keep scanned pages only")
			break
		}

		currentImg, capErr := screencapShelf(ctx)
		if capErr != nil {
			log.Warn().
				Err(capErr).
				Str("component", autoStockpileComponent).
				Str("step", "shelf_screencap").
				Int("page", pageIndex).
				Int("scanned_pages", len(pages)).
				Msg("shelf screencap failed, keep scanned pages only")
			break
		}

		if shelfFramesLookIdentical(previousImg, currentImg) {
			log.Info().
				Str("component", autoStockpileComponent).
				Str("step", "shelf_swipe_skip").
				Str("node", nodeName).
				Int("page", pageIndex).
				Msg("shelf did not move after swipe, stop scanning further pages")
			break
		}
		swiped++

		page, pageAbort, pageErr := scanGoodsOnImage(ctx, currentImg, region, itemMap)
		if pageAbort != AbortReasonNone || pageErr != nil {
			log.Warn().
				Err(pageErr).
				Str("component", autoStockpileComponent).
				Str("step", "scan_page").
				Str("abort_reason", string(pageAbort)).
				Int("page", pageIndex).
				Int("scanned_pages", len(pages)).
				Msg("page scan failed, keep scanned pages only")
			break
		}

		pages = append(pages, page)
		previousImg = currentImg
	}

	goods, swipeCountToReveal = mergeGoodsPages(pages)

	log.Info().
		Str("component", autoStockpileComponent).
		Str("step", "merge_pages").
		Int("page_count", len(pages)).
		Ints("page_goods_counts", pageGoodsCounts(pages)).
		Int("merged_count", len(goods)).
		Int("swipe_count_to_reveal", len(swipeCountToReveal)).
		Msg("shelf goods scan merged")

	restoreShelf(ctx, swiped)
	restored = true

	return goods, swipeCountToReveal, AbortReasonNone, nil
}

func pageGoodsCounts(pages [][]GoodsItem) []int {
	counts := make([]int, 0, len(pages))
	for _, page := range pages {
		counts = append(counts, len(page))
	}
	return counts
}
