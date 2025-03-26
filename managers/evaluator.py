from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, average_precision_score


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    # Evaluate model performance.
    def eval(self, save=False):
        # y_pred = []
        y_preds = []
        targets = []
        pred_labels = []
        all_loss = 0
        g_reps = []
        all_targets = []
        all_y_preds = []

        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=True,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                output, g_rep = self.graph_classifier(data_pos)

                # Save pair-wise representation.
                g_reps += g_rep.cpu().tolist()
                m = nn.Sigmoid()
                log = torch.squeeze(m(output), dim=-1)

                criterion = nn.BCELoss(reduce=False)
                loss_eval = criterion(log, r_labels_pos)
                loss = torch.sum(loss_eval)

                all_loss += loss.cpu().detach().numpy().item() / len(r_labels_pos)

                target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                targets += target

                y_pred = output.cpu().flatten().tolist()
                y_preds += y_pred

                all_targets.extend(target)
                all_y_preds.extend(y_pred)

            # auc_1 = roc_auc_score(targets, y_preds)
            all_auc = roc_auc_score(all_targets, all_y_preds)
            p, r, t = precision_recall_curve(all_targets, all_y_preds)
            all_aupr = auc(r, p)

            all_pred_labels = [1 if i >= 0.5 else 0 for i in all_y_preds]

            all_f1 = f1_score(all_targets, all_pred_labels)

        return {'loss': all_loss / b_idx, 'auc': all_auc, 'aupr': all_aupr, 'f1_score': all_f1}, (
            g_reps, pred_labels, targets)
