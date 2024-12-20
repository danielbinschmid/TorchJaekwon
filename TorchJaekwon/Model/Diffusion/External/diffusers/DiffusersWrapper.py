from typing import Optional
from torch import Tensor, device

import torch
from tqdm import tqdm
from TorchJaekwon.Util.UtilTorch import UtilTorch
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM, DDPMOutput
from typing import Literal

class DiffusersWrapper:
    @staticmethod
    def get_diffusers_output_type_name(ddpm_module: DDPM) -> str:
        output_type_dict = {
            'v_prediction': 'v_prediction',
            'noise': 'epsilon',
            'x_start': 'sample'
        }
        return output_type_dict[ddpm_module.model_output_type]
    
    @staticmethod
    def get_diffusers_scheduler_config(ddpm_module: DDPM, scheduler_args: dict):
        config:dict = {
            'num_train_timesteps': ddpm_module.timesteps,
            'trained_betas': ddpm_module.betas.to('cpu').detach().numpy(),
            'prediction_type': DiffusersWrapper.get_diffusers_output_type_name(ddpm_module),
        }
        config.update(scheduler_args)
        return config
    
    @staticmethod
    def infer(
        ddpm_module: DDPM, 
        diffusers_scheduler_class,
        x_shape:tuple,
        cond:Optional[dict] = None,
        is_cond_unpack:bool = False,
        num_steps: int = 20,
        scheduler_args: dict = {'timestep_spacing': 'trailing'},
        cfg_scale: float = None,
        device:device = None,
        x_start: Optional[torch.Tensor] = None,
        delta_h: Optional[torch.nn.Module] = None
        ) -> DDPMOutput:
        
        noise_scheduler = diffusers_scheduler_class(**DiffusersWrapper.get_diffusers_scheduler_config(ddpm_module, scheduler_args))
        noise_scheduler.set_timesteps(num_steps)
        
        _, cond, additional_data_dict = ddpm_module.preprocess(x_start = None, cond=cond)
        if x_shape is None: x_shape = ddpm_module.get_x_shape(cond=cond)
        model_device: "device" = UtilTorch.get_model_device(ddpm_module) if device is None else device
        
        x:Tensor = torch.randn(x_shape, device = model_device) if x_start is None else x_start
        x = x * noise_scheduler.init_noise_sigma
        for t in tqdm(noise_scheduler.timesteps, desc='sample time step'):
            
            t_tensor = torch.full((x_shape[0],), t, device=model_device, dtype=torch.long)
            
            # hspace steering
            if delta_h is not None and cond is not None:
                delta_h_cond = delta_h.forward(t_tensor)
                cond["delta_h"] = delta_h_cond
    
            denoiser_input = noise_scheduler.scale_model_input(x, t)
            model_output = ddpm_module.apply_model(denoiser_input, 
                                                   t_tensor, 
                                                   cond, 
                                                   is_cond_unpack, 
                                                   cfg_scale = ddpm_module.cfg_scale if cfg_scale is None else cfg_scale)
            x = noise_scheduler.step( model_output, t, x, return_dict=False)[0]
        
        return ddpm_module.postprocess(x, additional_data_dict)
        

        