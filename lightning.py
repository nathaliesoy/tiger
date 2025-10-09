import torch
from pytorch_lightning import LightningModule
from dataloader import tiger_dataset, collate_fn, TigerSampler
from loss import TigerLoss
from torch.utils.data import DataLoader
from models.tiger import TigerModel
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from numpy.random import default_rng

from sklearn import metrics

SEED = 123456
RNG = default_rng(SEED)

class Tiger_Lit(LightningModule):

    def __init__(self, config, comet_logger = None):

        super().__init__()

        self.config = config
        self.comet_logger = comet_logger

        self.model = TigerModel(self.config)

        self.stage2_modules = []

        for name, module in self.model.named_modules():
            if 'stage2' in name:
                self.stage2_modules.append(module)


        self.loss = TigerLoss()
        self.binary = self.config['binary_class']

        self.aux_task = self.config.get('aux_task', False)
        self.aux_task_fin = self.config.get('aux_task_fin', False)
        self.sb_class = self.config.get('sb_class', False)
        self.freeze_stage2 = self.config.get('freeze_stage2', False)

        self.use_sampler = self.config.get('use_sampler', True)

        if self.config['freeze_stage2']:
            self.freeze_stage2 = True
            self._freeze_stage2()
            self.unfreeze_stage_2_epoch = self.config['stage2_unfreeze_epoch']

        self.outputs = []

        self.tth_data = self.config.get('tth_data', False)

    def _freeze_stage2(self):
        print(f"Freezing Stage 2 parameters at epoch {self.current_epoch if hasattr(self, 'current_epoch') else 'initialization'}...")
        for module in self.stage2_modules:
            for param in module.parameters():
                param.requires_grad = False
        self.is_stage2_frozen = True

    def _unfreeze_stage2(self):
        print(f"Unfreezing Stage 2 parameters at epoch {self.current_epoch}...")
        for module in self.stage2_modules:
            for param in module.parameters():
                param.requires_grad = True
        self.is_stage2_frozen = False


    def forward(self, data):
        return self.model(data)
    
    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'])#, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.config['T_max'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def train_dataloader(self):
        dataset = tiger_dataset(self.config['train_data_path'], self.config, n_events= self.config['n_events_train'])
        print('Training dataset size:', len(dataset))
        if self.use_sampler:
            sampler = TigerSampler(dataset.data_dict['n_objects'], batch_size = self.config['batch_size'])
            return DataLoader(dataset, batch_sampler = sampler, collate_fn = collate_fn, num_workers=self.config['n_workers'])
        else:
            return DataLoader(dataset, batch_size = self.config['batch_size'], collate_fn = lambda batch: collate_fn(batch, sampler=False), num_workers=self.config['n_workers'], shuffle=True)
    
    def val_dataloader(self):
        dataset = tiger_dataset(self.config['val_data_path'], self.config, n_events= self.config['n_events_val'], event_start_idx= self.config['val_index_start'])
        print('Validation dataset size:', len(dataset))
        if self.use_sampler:
            sampler = TigerSampler(dataset.data_dict['n_objects'], batch_size = self.config['batch_size'])
            return DataLoader(dataset, batch_sampler = sampler, collate_fn = collate_fn, num_workers=self.config['n_workers'])
        else:
            return DataLoader(dataset, batch_size = self.config['batch_size'], collate_fn = lambda batch: collate_fn(batch, sampler=False), num_workers=self.config['n_workers'], shuffle=False)
        
    def on_train_epoch_start(self):
        if self.freeze_stage2:
            if self.current_epoch == self.unfreeze_stage_2_epoch and self.is_stage2_frozen:
                self._unfreeze_stage2()
                print(f"Unfreezing Stage 2 parameters at epoch {self.current_epoch}...")
    
    
    def training_step(self, batch, batch_idx):
        
        data = self.model(batch)
        
        edge_mask = data.get('edge_mask', None)
        
        loss_stage1 = self.loss.stage1_edge_loss(data['edge_pred'], data['adj_matrix'], binary = self.binary, edge_mask = edge_mask)
        
        if self.config['stage1_only']:
            total_loss = loss_stage1
        
        else:
            loss_stage2, _ = self.loss.stage2_edge_loss(data['edge_pred2'], 
                                                        data['meta_data_jetidx'], 
                                                        data['edge_prob_mask'], 
                                                        data['top_idx'],
                                                        node_mask = data.get('node_mask', None))
            
            if self.freeze_stage2:
                epoch_offset = self.unfreeze_stage_2_epoch
            else:
                epoch_offset = 0
            interpolation_weight = torch.sigmoid(
                torch.tensor([
                (self.current_epoch - epoch_offset) / self.config['balance_epoch'] * self.config['balance_slope'] - self.config['balance_offset']
                ])
                ).item()
            total_loss = (1 - 0.5 * interpolation_weight) * loss_stage1 + 0.5 * interpolation_weight * loss_stage2
            
            self.log('train_stage2', loss_stage2, sync_dist=True)

            if self.aux_task_fin:
                aux_loss_fin = self.loss.aux_loss_fin(data['aux_pred_fin'], data['aux_tag_fin'])
                self.log('train_aux_fin', aux_loss_fin, sync_dist=True)
                total_loss += aux_loss_fin / self.config['aux_task_fin_scale']

            if self.sb_class:
                sb_loss = self.loss.sb_class_loss(data['sb_pred'], data['signal'])
                self.log('train_sb_class', sb_loss, sync_dist=True)
                total_loss += sb_loss 

            
        if self.aux_task:
            aux_loss = self.loss.aux_loss(data['aux_pred'], data['aux_tag'], node_mask = data.get('node_mask', None))
            self.log('train_aux', aux_loss, sync_dist=True)
            total_loss += aux_loss / self.config['aux_task_scale']
        
        self.log('train_stage1', loss_stage1, sync_dist=True)
        self.log('train_total', total_loss, sync_dist=True)

        return total_loss
    
 
    def validation_step(self, batch, batch_idx):
        
        data = self.model(batch)
        edge_mask = data.get('edge_mask', None)

        loss_stage1 = self.loss.stage1_edge_loss(data['edge_pred'], data['adj_matrix'], binary = self.binary, edge_mask = edge_mask)
        
        if self.config['stage1_only']:
            total_loss = loss_stage1
            return_dict = {'pred_labels': data['edge_pred'],
                            'adj_matrix': data['adj_matrix'],
                            'node_mask': data.get('node_mask', None),
                            'edge_mask': edge_mask
                            }
            
        else:
            loss_stage2, target2 = self.loss.stage2_edge_loss(data['edge_pred2'],
                                                            data['meta_data_jetidx'], 
                                                            data['edge_prob_mask'], 
                                                            data['top_idx'],
                                                            node_mask = data.get('node_mask', None))
            self.log('val_stage2', loss_stage2, sync_dist=True)

            total_loss = loss_stage1 + loss_stage2


            return_dict = {'pred_labels': data['edge_pred'],
                            'adj_matrix': data['adj_matrix'],
                            'pred_labels2': data['edge_pred2'],
                            'edge_prob_mask': data['edge_prob_mask'],
                            'node_mask': data.get('node_mask', None),
                            'target2': target2,
                            'edge_mask': edge_mask}
            
            if self.aux_task_fin:
                aux_loss_fin = self.loss.aux_loss_fin(data['aux_pred_fin'], data['aux_tag_fin'])
                self.log('val_aux_fin', aux_loss_fin, sync_dist=True)
                total_loss += aux_loss_fin / self.config['aux_task_fin_scale']
                return_dict['aux_pred_fin'] = data['aux_pred_fin']
                return_dict['aux_tag_fin'] = data['aux_tag_fin']

            if self.sb_class:
                sb_loss = self.loss.sb_class_loss(data['sb_pred'], data['signal'])
                self.log('val_sb_class', sb_loss, sync_dist=True)
                total_loss += sb_loss
                return_dict['sb_pred'] = data['sb_pred']
                return_dict['signal'] = data['signal']
            
        if self.aux_task:
            aux_loss = self.loss.aux_loss(data['aux_pred'], data['aux_tag'], node_mask = data.get('node_mask', None))
            self.log('val_aux', aux_loss, sync_dist=True)
            total_loss += aux_loss / self.config['aux_task_scale']
            return_dict['aux_pred'] = data['aux_pred']
            return_dict['aux_tag'] = data['aux_tag']
            
        self.outputs.append(return_dict)
        self.log('val_total', total_loss, sync_dist=True)
        self.log('val_stage1', loss_stage1, sync_dist=True)

        return return_dict
    
    def on_train_epoch_end(self):
        self.lr_schedulers().step()
        self.log('lr', self.lr_schedulers().get_last_lr()[0], sync_dist=True)
    
    def on_validation_epoch_end(self):

        # reshuffle edge logits 
        edge_logits = []
        target_matrix = []
        edge_logits2 = []
        target2 = []
        mask = []
        edge_mask1 = []
        node_mask = []
        for out in self.outputs:
            edge_logits.append(out['pred_labels'].flatten(0,1))
            target_matrix.append(out['adj_matrix'].flatten())
            edge_mask1.append(out['edge_mask'].flatten() if out['edge_mask'] is not None else None)
            if 'pred_labels2' in out:
                edge_logits2.append(out['pred_labels2'].flatten())
                target2.append(out['target2'].flatten())
                edge_mask = out['edge_prob_mask'].unsqueeze(1).repeat(1, out['pred_labels2'].size(1),1).flatten()
                
                if out['node_mask'] is not None:
                    node_mask = out['node_mask'].unsqueeze(2).repeat(1, 1, out['pred_labels2'].size(2)).flatten()
                    mask_ = edge_mask * node_mask
                else:
                    mask_ = edge_mask
                mask.append(mask_)


        if self.aux_task:
            aux_pred = torch.cat([torch.argmax(out['aux_pred'], dim = -1).flatten() for out in self.outputs], dim=0)
            aux_tag = torch.cat([out['aux_tag'].flatten() for out in self.outputs], dim=0)
            node_mask = torch.cat([out['node_mask'].flatten() for out in self.outputs], dim=0) if self.outputs[0].get('node_mask', None) is not None else None
        if self.aux_task_fin:
            aux_pred_fin = torch.cat([torch.argmax(out['aux_pred_fin'], dim = -1) for out in self.outputs], dim=0)
            aux_tag_fin = torch.cat([out['aux_tag_fin'] for out in self.outputs], dim=0)
        if self.sb_class:
            sb_pred = torch.cat([torch.sigmoid(out['sb_pred']).flatten() for out in self.outputs], dim=0)
            signal = torch.cat([out['signal'].flatten() for out in self.outputs], dim=0)
        edge_logits = torch.cat(edge_logits, dim=0)
        target_matrix = torch.cat(target_matrix, dim=0).detach().cpu().numpy()
        edge_mask1 = torch.cat(edge_mask1, dim=0).detach().cpu().numpy() if edge_mask1[0] is not None else None

        if self.binary:
            edge_sigmoid = torch.sigmoid(edge_logits).detach().cpu().numpy()

        if len(edge_logits2) > 0:
            edge_logits2 = torch.cat(edge_logits2, dim=0)
            target2 = torch.cat(target2, dim=0).detach().cpu().numpy()
            mask = torch.cat(mask, dim=0).detach().cpu().numpy()
            edge_sigmoid2 = torch.sigmoid(edge_logits2).detach().cpu().numpy()

        if self.config['binary_class']:
            # make a roc curve
            if edge_mask1 is not None:
                fpr, tpr, _ = metrics.roc_curve(target_matrix[edge_mask1], edge_logits.detach().cpu().numpy()[edge_mask1])
            else:
                fpr, tpr, _ = metrics.roc_curve(target_matrix, edge_logits.detach().cpu().numpy())

            roc_auc = metrics.auc(fpr, tpr)
            # plot the roc curve
            fig, ax = plt.subplots(3,2, figsize=(10, 15), dpi=100, tight_layout=True)
            ax[0,0].plot(tpr, 1 / (fpr + 1e-8), color='blue', label=f'ROC curve (area = {roc_auc:.3f})')
            ax[0,0].set_xlim(1e-2,1)
            ax[0,0].set_ylim(1e-0, 1e5)
            ax[0,0].set_yscale('log')
            ax[0,0].legend(loc='upper right')

            ax[0,1].hist(edge_sigmoid[target_matrix == 0], bins = np.linspace(0, 1, 50), histtype='stepfilled', alpha = 0.5, color='blue', label='Negative samples')
            ax[0,1].hist(edge_sigmoid[target_matrix == 1], bins = np.linspace(0, 1, 50), histtype='stepfilled', alpha = 0.5, color='red', label='Positive samples')
            ax[0,1].set_xlabel('Edge probability')
            ax[0,1].set_ylabel('Count')
            ax[0,1].set_yscale('log')
            ax[0,1].legend(loc='upper right')

            
            if (len(edge_logits2) > 0):
                if (len(np.unique(target2[mask])) > 1):
                    fpr2, tpr2, _ = metrics.roc_curve(target2[mask], edge_logits2[mask].detach().cpu().numpy())
                    roc_auc2 = metrics.auc(fpr2, tpr2)
                    ax[1,0].plot(tpr2, 1 / (fpr2 + 1e-8), color='blue', label=f'ROC curve 2 (area = {roc_auc2:.3f})')
                    ax[1,0].set_xlim(1e-2,1)
                    ax[1,0].set_ylim(1e-0, 1e5)
                    ax[1,0].set_yscale('log')
                    ax[1,0].legend(loc='upper right')

                    ax[1,1].hist(edge_sigmoid2[mask][target2[mask] == 0], bins = np.linspace(0, 1, 50), histtype='stepfilled', alpha = 0.5, color='blue', label='Negative samples')
                    ax[1,1].hist(edge_sigmoid2[mask][target2[mask] == 1], bins = np.linspace(0, 1, 50), histtype='stepfilled', alpha = 0.5, color='red', label='Positive samples')
                    ax[1,1].set_xlabel('Edge probability')
                    ax[1,1].set_ylabel('Count')
                    ax[1,1].set_yscale('log')
                    ax[1,1].legend(loc='upper right')

                if self.aux_task_fin:
                    all_labels = ['0t', '1t', '2t']
                    classes_fin = np.unique(np.concatenate((aux_tag_fin.detach().cpu().numpy(), aux_pred_fin.detach().cpu().numpy())))
                    aux_tick_labels_fin = [all_labels[int(i)] for i in classes_fin]
                    aux_confusion_matrix_fin = metrics.confusion_matrix(aux_tag_fin.detach().cpu().numpy(), aux_pred_fin.detach().cpu().numpy(), normalize='true')
                    aux_cm_display_fin = metrics.ConfusionMatrixDisplay(confusion_matrix=aux_confusion_matrix_fin, display_labels=aux_tick_labels_fin)
                    ax[2,1].set_title('Auxiliary task confusion matrix')
                    aux_cm_display_fin.plot(ax=ax[2,1])

                
        else:
            # make a confusion matrix
            classes = torch.argmax(edge_logits, dim=1).detach().cpu().numpy()
            confusion_matrix = metrics.confusion_matrix(target_matrix, classes, normalize='true')
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)

            fig, ax = plt.subplots(3,2, figsize=(10, 15), dpi=100, tight_layout=True)

            cm_display.plot(ax=ax[0,0])

            tick_labels = ['#', 'W', 'H']

            if len(tick_labels) == confusion_matrix.shape[0]:
                ax[0,0].set_xticklabels(tick_labels)
                ax[0,0].set_yticklabels(tick_labels)
            else:
                print("Warning: Number of tick labels does not match the size of the confusion matrix.")

            ax[0,0].set_title('W matrix')

            if len(edge_logits2) > 0:
                if (len(np.unique(target2[mask])) > 1):
                    fpr2, tpr2, _ = metrics.roc_curve(target2[mask], edge_logits2[mask].detach().cpu().numpy())
                    roc_auc2 = metrics.auc(fpr2, tpr2)
                    ax[1,0].plot(tpr2, 1 / (fpr2 + 1e-8), color='blue', label=f'ROC curve 2 (area = {roc_auc2:.3f})')
                    ax[1,0].set_xlim(1e-2,1)
                    ax[1,0].set_ylim(1e-0, 1e5)
                    ax[1,0].set_yscale('log')
                    ax[1,0].legend(loc='upper right')

                    ax[1,1].hist(edge_sigmoid2[mask][target2[mask] == 0], bins = np.linspace(0, 1, 50), histtype='step', color='blue', label='Negative samples')
                    ax[1,1].hist(edge_sigmoid2[mask][target2[mask] == 1], bins = np.linspace(0, 1, 50), histtype='step', color='red', label='Positive samples')
                    ax[1,1].set_xlabel('Edge probability')
                    ax[1,1].set_ylabel('Count')
                    ax[1,1].set_yscale('log')
                    ax[1,1].legend(loc='upper right')

                if self.aux_task_fin:
                    all_labels = ['0h0t', '0h1t', '0h2t', '1h0t', '1h1t', '1h2t']
                    classes_fin = np.unique(np.concatenate((aux_tag_fin.detach().cpu().numpy(), aux_pred_fin.detach().cpu().numpy())))
                    aux_tick_labels_fin = [all_labels[int(i)] for i in classes_fin]
                    aux_confusion_matrix_fin = metrics.confusion_matrix(aux_tag_fin.detach().cpu().numpy(), aux_pred_fin.detach().cpu().numpy(), normalize='true')
                    aux_cm_display_fin = metrics.ConfusionMatrixDisplay(confusion_matrix=aux_confusion_matrix_fin, display_labels=aux_tick_labels_fin)
                    ax[2,1].set_title('Auxiliary task confusion matrix')
                    aux_cm_display_fin.plot(ax=ax[2,1])

                if self.sb_class:
                    sb_fpr, sb_tpr, _ = metrics.roc_curve(signal.detach().cpu().numpy(), sb_pred.detach().cpu().numpy())
                    sb_roc_auc = metrics.auc(sb_fpr, sb_tpr)
                    ax[0,1].plot(sb_tpr, 1 / (sb_fpr + 1e-8), color='blue', label=f'SB ROC curve (area = {sb_roc_auc:.3f})')
                    ax[0,1].set_xlim(1e-2,1)
                    ax[0,1].set_ylim(1e-0, 1e5)
                    ax[0,1].set_yscale('log')
                    ax[0,1].legend(loc='upper right')
                    ax[0,1].set_title('SB ROC curve')


                


        if self.aux_task:
            # all_labels = ['0h0t', '0h1t', '0h2t', '1h0t', '1h1t', '1h2t']
            if self.tth_data:
                all_labels = ['fisr', 'recW', 'nrecW', 'recT', 'nrecT', 'recH', 'nrecH', 'lep', 'met']
            else:
                all_labels = ['fisr', 'recW', 'nrecW', 'recT', 'nrecT']
            aux_tag = aux_tag.detach().cpu().numpy()
            aux_pred = aux_pred.detach().cpu().numpy()
            if node_mask is not None:
                node_mask = node_mask.detach().cpu().numpy()
                aux_tag = aux_tag[node_mask]
                aux_pred = aux_pred[node_mask]

            classes = np.unique(np.concatenate((aux_tag, aux_pred)))
            aux_tick_labels = [all_labels[int(i)] for i in classes]
            
            aux_confusion_matrix = metrics.confusion_matrix(aux_tag, aux_pred, normalize='true')
            aux_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=aux_confusion_matrix, display_labels=aux_tick_labels)

            ax[2,0].set_title('Auxiliary task confusion matrix')
            aux_cm_display.plot(ax=ax[2,0])

            
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

        if self.comet_logger is not None:
            self.comet_logger.experiment.log_image(image, name='roc_curve', step=self.current_epoch)

        else:
            fig.savefig('validation_plot.png')

        plt.close(fig)

        self.outputs.clear()



        


    
    