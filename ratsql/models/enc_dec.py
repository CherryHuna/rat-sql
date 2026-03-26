import torch
import torch.utils.data
import json

from ratsql.models import abstract_preproc
from ratsql.utils import registry
import torch.nn.functional as F

class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(lengths[0] == other for other in lengths[1:]), f"Lengths don't match: {lengths}"
        self.components = components
    
    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)
    
    def __len__(self):
        return len(self.components[0])


@registry.register('model', 'EncDec')
class EncDecModel(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                encoder,
                decoder,
                encoder_preproc,
                decoder_preproc):
            super().__init__()

            self.enc_preproc = registry.lookup('encoder', encoder['name']).Preproc(**encoder_preproc)
            self.dec_preproc = registry.lookup('decoder', decoder['name']).Preproc(**decoder_preproc)
        
        def validate_item(self, item, section):
            enc_result, enc_info = self.enc_preproc.validate_item(item, section)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section)
            
            return enc_result and dec_result, (enc_info, dec_info)
        
        def add_item(self, item, section, validation_info):
            enc_info, dec_info = validation_info
            self.enc_preproc.add_item(item, section, enc_info)
            self.dec_preproc.add_item(item, section, dec_info)
        
        def clear_items(self):
            self.enc_preproc.clear_items()
            self.dec_preproc.clear_items()

        def save(self):
            self.enc_preproc.save()
            self.dec_preproc.save()

        def save_pred(self):
            self.dec_preproc.save_pred()
        
        def load(self):
            self.enc_preproc.load()
            self.dec_preproc.load()
        
        def dataset(self, section):
            return ZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))

        
    def __init__(self, preproc, device, encoder, decoder, args = None):
        super().__init__()
        self.args = args
        self.preproc = preproc
        conifg_select = json.loads(self.args.config_args)
        self.encoder = registry.construct(
                'encoder', encoder, device=device, preproc=preproc.enc_preproc,
            train_select_step = conifg_select['train_select_step'])
        self.decoder = registry.construct(
                'decoder', decoder, device=device, preproc=preproc.dec_preproc,
            select_where_train_step=conifg_select['train_select_step'])


        if getattr(self.encoder, 'batched'):
            self.compute_loss = self._compute_loss_enc_batched
        else:
            self.compute_loss = self._compute_loss_unbatched

        self.args = args

    def get_sql_component_param_groups(self):

        param_groups = []
        named_params = list(self.named_parameters())

        # ========== 1. SELECT 子句参数 ==========
        select_params = []
        # 匹配 SELECT 相关参数名（根据 RAT-SQL 解码器命名规则）
        select_keywords = [
            'select', 'agg', 'column', 'alias', 'distinct',
            'decoder.nt_select', 'decoder.t_select',
        ]
        conifg_select = json.loads(self.args.config_args)
        for name, param in named_params:
            if any(kw in name.lower() for kw in select_keywords) and param.requires_grad:
                select_params.append(param)
        select_group = {
            'params': select_params,
            'lr': conifg_select["select_lr"] or conifg_select["lr"] * 1.2,  # SELECT 子句学习率（默认1.2倍全局）
            'weight_decay': 0.01,
            'name': 'select_clause'
        }
        param_groups.append(select_group)

        # ========== 2. FROM 子句参数 ==========
        from_params = []
        from_keywords = [
            'from', 'table', 'join', 'schema', 'database',
            'decoder.nt_from', 'decoder.t_from'
        ]
        for name, param in named_params:
            if any(kw in name.lower() for kw in from_keywords) and param.requires_grad:
                from_params.append(param)

        from_group = {
            'params': from_params,
            'lr': conifg_select["from_lr"] or conifg_select["lr"] * 1.0,  # FROM 子句学习率（默认等于全局）
            'weight_decay':  0.01,
            'name': 'from_clause'
        }
        param_groups.append(from_group)

        # ========== 3. UNION 子句参数 ==========
        union_params = []
        union_keywords = [
            'union', 'intersect', 'except',
            'decoder.nt_union', 'decoder.t_union'
        ]
        for name, param in named_params:
            if any(kw in name.lower() for kw in union_keywords) and param.requires_grad:
                union_params.append(param)

        # UNION 子句样本少，设置更高学习率
        union_group = {
            'params': union_params,
            'lr': conifg_select["union_lr"] or conifg_select["lr"] * 2.0,  # UNION 子句学习率（默认2倍全局）
            'weight_decay': 0.01,
            'name': 'union_clause'
        }
        param_groups.append(union_group)

        # ========== 4. 其他参数 ==========
        # 排除已分组的参数名
        grouped_names = set()
        for g in [select_params, from_params, union_params]:
            for p in g:
                for name, param in named_params:
                    if param is p:
                        grouped_names.add(name)

        other_params = []
        for name, param in named_params:
            if name not in grouped_names and param.requires_grad:
                other_params.append(param)

        other_group = {
            'params': other_params,
            'lr': conifg_select["lr"],  # 全局基础学习率
            'weight_decay': 0.01,
            'name': 'other_clauses'
        }
        param_groups.append(other_group)

        return param_groups

    def _compute_loss_enc_batched(self, batch, debug=False, step=None):
        losses = []
        enc_states = self.encoder([enc_input for enc_input, dec_output in batch],
                                  descs_first=[dec_output for enc_input, dec_output in batch], step= step)

        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug, step = step)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_enc_batched2(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state, = self.encoder([enc_input])
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_unbatched(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state = self.encoder(enc_input)
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def eval_on_batch(self, batch, step=None):
        mean_loss = self.compute_loss(batch, step = step).item()
        batch_size = len(batch)
        result = {'loss': mean_loss * batch_size, 'total': batch_size}
        return result

    def begin_inference(self, orig_item, preproc_item):
        enc_input, _ = preproc_item
        if getattr(self.encoder, 'batched'):
            enc_state, = self.encoder([enc_input])
        else:
            enc_state = self.encoder(enc_input)
        return self.decoder.begin_inference(enc_state, orig_item)