from mmdet.registry import MODELS
from .data_preprocessor import DetDataPreprocessor
from typing import Sequence, Union
from numbers import Number
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.utils import is_seq_of
import torch
from mmengine.model.utils import stack_batch
import torch.nn.functional as F
import math


@MODELS.register_module()
class RGBTDetDataPreprocessor(DetDataPreprocessor):

    def __init__(self,
                 rgb_mean: Sequence[Number] = None,
                 thermal_mean: Sequence[Number] = None,
                 rgb_std: Sequence[Number] = None,
                 thermal_std: Sequence[Number] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._enable_normalize = True
        self.register_buffer('rgb_mean',
                             torch.tensor(rgb_mean).view(-1, 1, 1), False)
        self.register_buffer('thermal_mean',
                             torch.tensor(thermal_mean).view(-1, 1, 1), False)
        self.register_buffer('rgb_std',
                             torch.tensor(rgb_std).view(-1, 1, 1), False)
        self.register_buffer('thermal_std',
                             torch.tensor(thermal_std).view(-1, 1, 1), False)

    def forwardnormalize(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        _rgb_batch_inputs = data['rgb_inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            assert (_rgb_batch_inputs, torch.Tensor)
            batch_inputs = []
            rgb_batch_inputs = []
            for _batch_input, _rgb_batch_input in zip(_batch_inputs, _rgb_batch_inputs):
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                    _rgb_batch_input = _rgb_batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                _rgb_batch_input = _rgb_batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.rgb_mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.thermal_mean) / self.thermal_std
                    _rgb_batch_input = (_rgb_batch_input - self.rgb_mean) / self.rgb_std
                batch_inputs.append(_batch_input)
                rgb_batch_inputs.append(_rgb_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
            rgb_batch_inputs = stack_batch(rgb_batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert isinstance(_rgb_batch_inputs, torch.Tensor)
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
                _rgb_batch_inputs = _rgb_batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            _rgb_batch_inputs = _rgb_batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.thermal_mean) / self.thermal_std
                _rgb_batch_inputs = (_rgb_batch_inputs - self.rgb_mean) / self.rgb_std
                assert _batch_inputs.shape == _rgb_batch_inputs.shape
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
            rgb_batch_inputs = F.pad(_rgb_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data['rgb_inputs'] = rgb_batch_inputs
        data.setdefault('data_samples', None)
        return data

    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)
        data = self.forwardnormalize(data, training)
        inputs, rgb_inputs, data_samples = data['inputs'], data['rgb_inputs'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            raise NotImplementedError

        return {'inputs': torch.cat([rgb_inputs, inputs], 1), 'data_samples': data_samples}
