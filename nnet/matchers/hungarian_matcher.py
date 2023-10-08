import torch
from torch import nn

from scipy.optimize import linear_sum_assignment

from nnet.utils.boxes import generalized_box_iou, box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Hungarian matcher module. It matches all the targets with all the anchors.
        :param cost_class: Relative weight of classification
        :param cost_bbox: Relative weight of L1 error of bounding box
        :param cost_giou: Relative weight of giou
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, target_classes: [torch.Tensor],
                target_boxes: [torch.Tensor]) -> [(int, int)]:
        """
        Match all the targets with all the anchors.
        :param pred_logits: predicted logits of shape (batch_size, num_anchors, num_classes)
        :param pred_boxes: predicted boxes of shape (batch_size, num_anchors, 4), in format (cx, cy, w, h)
        :param target_classes: list of tensors of shape (num_targets) containing the class of each target
        :param target_boxes: list of tensors of shape (num_targets, 4) containing the bounding box of each target in format (cx, cy, w, h)
        :return: matched indices, list of tuples (is, js) where is are the indices of the prediction and js is the indices of the
                 target
        """
        batch_size, num_pred = pred_logits.shape[:2]

        # Flatten over batch dimension
        out_prob = pred_logits.flatten(0, 1).softmax(-1)
        out_bbox = pred_boxes.flatten(0, 1)

        # Do the same for target
        tgt_ids = torch.cat(target_classes)
        tgt_bbox = torch.cat(target_boxes)

        # Classification cost
        cost_class = -out_prob[:, tgt_ids]  # Approximate 1 - P[y] with -P[y]

        # L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Cost matrix
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(batch_size, num_pred, -1).cpu()

        # Match
        sizes_per_batch = [len(t) for t in target_boxes]
        # Solve the linear sum assignment problem (best pairing) for each batch separately
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes_per_batch, -1))]
        # Format the output as list of tensors pair
        output = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return output
