import torch
from torchvision.ops._utils import _upcast


def box_cxcywh_to_xyxy(box):
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
    :param box: tensor of shape (N, 4), where each row is (cx, cy, w, h)
    :return: tensor of shape (N, 4), where each row is (x1, y1, x2, y2)
    """
    cx, cy, w, h = box.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(box):
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h)
    :param box: tensor of shape (N, 4), where each row is (x1, y1, x2, y2)
    :return: tensor of shape (N, 4), where each row is (cx, cy, w, h)
    """
    x1, y1, x2, y2 = box.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of bounding boxes.
    :param boxes: tensor of boxes, of shape (N, 4), format (x1, y1, x2, y2)
    :return: tensor of shape (N) containing the area of each box
    """
    boxes = _upcast(boxes)
    widths = (boxes[:, 2] - boxes[:, 0])
    heights = (boxes[:, 3] - boxes[:, 1])
    return widths * heights


def box_iou(boxes1, boxes2):
    """
    Compute the IoU between two boxes.
    :param boxes1: list of boxes 1, of shape (N, 4), format (x1, y1, x2, y2), 0 <= x1 < x2, 0 <= y1 < y2
    :param boxes2: list of boxes 2, of shape (M, 4), format (x1, y1, x2, y2), 0 <= x1 < x2, 0 <= y1 < y2
    :return: (iou, union) where iou is the IoU and union the union area
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Left top and right bottom coordinates of intersection boxes
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    # width and height of intersection boxes
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    # area of intersection boxes
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # area of union (A + B - A ∩ B)
    union = area1[:, None] + area2 - inter

    # IoU
    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU between two boxes.
    Each box is represented as a tensor of shape (4) where the last dimension corresponds to (x1, y1, x2, y2).
    :param boxes1: list of boxes 1
    :param boxes2: list of boxes 2
    :return: a (N, M) where N is the number of boxes in b1 and M the number of boxes in b2
    """
    # Avoid degenerate boxes
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # Compute IoU and union
    iou, union = box_iou(boxes1, boxes2)

    # Left top and right bottom coordinates of intersection boxes
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # width and height of intersection boxes
    wh = (rb - lt).clamp(min=0)

    # area is the area of the intersection
    area = wh[:, :, 0] * wh[:, :, 1]

    # Return the generalized IoU
    return iou - (area - union) / area
