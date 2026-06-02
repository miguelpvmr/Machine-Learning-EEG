import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

class ClassificationScorer:
    """
    Herramienta funcional para el cálculo de métricas multiclase.
    Retorna exclusivamente valores escalares para compatibilidad con el Trainer.
    """
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.labels = list(range(num_classes))

    def _get_cm_and_stats(self, y_true, y_pred):
        """Base de cálculo: Matriz de confusión y estadísticas por clase."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t in self.labels and p in self.labels:
                cm[int(t), int(p)] += 1
        
        stats = []
        total = cm.sum()
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = total - (tp + fp + fn)
            stats.append({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
        return cm, stats

    def _get_recalls_list(self, y_true, y_pred):
        """Método privado para obtener recalls sin promediar (uso interno)."""
        _, stats = self._get_cm_and_stats(y_true, y_pred)
        return [s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0.0 for s in stats]

    def get_sensitivity(self, y_true, y_pred):
        """Retorna Macro-Recall (promedio simple)."""
        recalls = self._get_recalls_list(y_true, y_pred)
        return np.mean(recalls)

    def get_g_index(self, y_true, y_pred):
        """Media geométrica de los recalls de cada clase: $G = (\prod Recall_i)^{1/n}$."""
        recalls = self._get_recalls_list(y_true, y_pred)
        # Se asegura que el producto no sea negativo antes de la raíz
        return np.power(np.prod(recalls), 1/self.num_classes)

    def get_specificity(self, y_true, y_pred):
        """Retorna Macro-Specificity."""
        _, stats = self._get_cm_and_stats(y_true, y_pred)
        specs = [s['tn'] / (s['tn'] + s['fp']) if (s['tn'] + s['fp']) > 0 else 0.0 for s in stats]
        return np.mean(specs)

    def get_fbeta_score(self, y_true, y_pred, beta=2):
        """Retorna Macro F-beta score."""
        _, stats = self._get_cm_and_stats(y_true, y_pred)
        recalls = [s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0.0 for s in stats]
        precs = [s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0.0 for s in stats]
        
        beta_sq = beta ** 2
        fbeta_scores = []
        for p, r in zip(precs, recalls):
            denom = (beta_sq * p) + r
            fbeta = (1 + beta_sq) * (p * r) / denom if denom > 0 else 0.0
            fbeta_scores.append(fbeta)
            
        return np.mean(fbeta_scores)

    def get_all_metrics(self, y_true, y_pred, y_prob=None):
            """Diccionario de métricas escalares ajustado para multiclase."""
            cm, stats = self._get_cm_and_stats(y_true, y_pred)
            recalls = self._get_recalls_list(y_true, y_pred)
            precs = [s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0.0 for s in stats]
            specs = [s['tn'] / (s['tn'] + s['fp']) if (s['tn'] + s['fp']) > 0 else 0.0 for s in stats]
            
            metrics = {
                'accuracy': cm.trace() / cm.sum() if cm.sum() > 0 else 0.0,
                'precision': np.mean(precs),
                'sensitivity': np.mean(recalls),
                'specificity': np.mean(specs),
                'g_index': self.get_g_index(y_true, y_pred),
                'f1_score': np.mean([(2*p*r)/(p+r) if (p+r)>0 else 0.0 for p, r in zip(precs, recalls)]),
                'f2_score': self.get_fbeta_score(y_true, y_pred, beta=2)
            }

            if y_prob is not None:
                if self.num_classes > 2:
                    # Caso Multiclase: y_prob debe ser una matriz (N, num_classes)
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    # Binarizamos y_true para calcular el PR-AUC multiclase (Macro)
                    y_true_bin = label_binarize(y_true, classes=self.labels)
                    metrics['pr_auc'] = average_precision_score(y_true_bin, y_prob, average='macro')
                else:
                    # Caso Binario: y_prob debe ser un vector 1D (clase positiva)
                    # Si y_prob viene como matriz (N, 2), tomamos la columna 1
                    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                        y_prob = y_prob[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                
            return metrics